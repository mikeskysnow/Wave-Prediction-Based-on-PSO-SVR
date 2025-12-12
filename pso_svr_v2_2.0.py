import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')


# -------------------------- 1. 模拟波浪数据生成（修复NaN问题）--------------------------
def generate_wave_data(n_samples=1000):
    """
    生成模拟数据：基于风速-波高经验公式（H_s = 0.0208*V²），修复负数导致的NaN问题
    """
    np.random.seed(42)  # 固定随机种子，保证可复现
    # 1. 生成风速（5-25 m/s，符合近海风场特性，确保风速不会过小）
    wind_speed = np.random.uniform(6, 25, n_samples) + np.random.normal(0, 0.8, n_samples)
    # 2. 生成风向（0-360°，转换为余弦值归一化）
    wind_dir = np.random.uniform(0, 360, n_samples)
    wind_dir_cos = np.cos(np.radians(wind_dir))  # 归一化到[-1,1]
    # 3. 生成有效波高（基于经验公式+噪声，用np.maximum确保非负，避免NaN）
    sig_wave_height = 0.0208 * wind_speed ** 2 + np.random.normal(0, 0.2, n_samples)
    sig_wave_height = np.maximum(sig_wave_height, 1e-6)  # 限制最小值为1e-6，杜绝负数和零
    # 4. 生成波周期（与波高正相关，3-15s，此时sqrt无NaN）
    wave_period = 2.5 + 0.8 * np.sqrt(sig_wave_height) + np.random.normal(0, 0.4, n_samples)

    # 构建输入特征（时间序列滞后特征）
    X = []
    y = []

    # 定义特征名称
    feature_names = [
        'wind_speed_t-3', 'wind_speed_t-2', 'wind_speed_t-1',
        'wave_height_t-3', 'wave_height_t-2', 'wave_height_t-1',
        'wave_period_t-2', 'wave_period_t-1',
        'wind_dir_cos_t'
    ]

    for i in range(3, n_samples):  # 前3个时刻的特征预测当前波高
        features = [
            wind_speed[i - 3], wind_speed[i - 2], wind_speed[i - 1],  # 滞后3/2/1时刻风速
            sig_wave_height[i - 3], sig_wave_height[i - 2], sig_wave_height[i - 1],  # 滞后3/2/1时刻波高
            wave_period[i - 2], wave_period[i - 1],  # 滞后2/1时刻波周期
            wind_dir_cos[i]  # 当前时刻风向余弦
        ]
        X.append(features)
        y.append(sig_wave_height[i])

    # 转换为数组并过滤NaN（双重保险）
    X = np.array(X)
    y = np.array(y)
    # 删除包含NaN的样本（若仍存在）
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, feature_names


# -------------------------- 2. PSO优化SVR参数（增加早停机制）--------------------------
class PSO_SVR:
    def __init__(self, n_particles=20, max_iter=50, c1=2, c2=2, w_max=0.9, w_min=0.4, patience=10, min_delta=1e-4):
        self.n_particles = n_particles  # 粒子数
        self.max_iter = max_iter  # 最大迭代次数
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 全局学习因子
        self.w_max = w_max  # 最大惯性权重
        self.w_min = w_min  # 最小惯性权重
        self.patience = patience  # 早停耐心值
        self.min_delta = min_delta  # 最小改进阈值
        self.best_iteration = 0  # 最佳迭代次数

        # SVR参数搜索范围（优化范围，避免极端值）
        self.param_bounds = [
            [1e-1, 1e2],  # C的范围（缩小下限，避免过拟合）
            [1e-3, 1e1],  # gamma的范围（缩小范围，提升稳定性）
            [1e-3, 5e-2]  # epsilon的范围（优化区间，适配数据）
        ]

    def fitness_func(self, params, X_train, y_train, n_splits=5):
        """适应度函数：使用时间序列交叉验证的RMSE"""
        C, gamma, epsilon = params
        try:
            svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)

            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                # 分割数据
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                # 训练和验证
                svr.fit(X_train_fold, y_train_fold)
                y_pred = svr.predict(X_val_fold)

                # 计算RMSE
                fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                scores.append(fold_rmse)

            # 返回平均RMSE
            return np.mean(scores)

        except Exception as e:
            # 若参数异常导致训练失败，返回极大值（淘汰该粒子）
            return 1e10

    def optimize(self, X_train, y_train):
        """PSO优化过程（带早停机制）"""
        n_params = len(self.param_bounds)

        # 1. 初始化粒子位置和速度
        particles = np.random.uniform(
            low=[b[0] for b in self.param_bounds],
            high=[b[1] for b in self.param_bounds],
            size=(self.n_particles, n_params)
        )
        velocities = np.random.uniform(-0.5, 0.5, size=(self.n_particles, n_params))

        # 2. 初始化个体最优和全局最优
        p_best = particles.copy()  # 个体最优位置
        p_best_fitness = np.array([self.fitness_func(p, X_train, y_train) for p in particles])
        g_best_idx = np.argmin(p_best_fitness)  # 全局最优索引
        g_best = particles[g_best_idx].copy()  # 全局最优位置
        g_best_fitness = p_best_fitness[g_best_idx]

        # 记录迭代过程的适应度值
        fitness_history = [g_best_fitness]
        best_iteration = 0  # 记录最佳适应度出现的迭代次数
        patience_counter = 0  # 早停计数器

        # 3. 迭代优化（带早停机制）
        for t in range(self.max_iter):
            # 计算当前惯性权重（线性递减）
            w = self.w_max - (self.w_max - self.w_min) * (t / self.max_iter)

            for i in range(self.n_particles):
                # 更新速度
                r1, r2 = np.random.rand(n_params), np.random.rand(n_params)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (p_best[i] - particles[i]) +
                                 self.c2 * r2 * (g_best - particles[i]))

                # 更新位置（限制在参数范围内）
                particles[i] = np.clip(particles[i],
                                       [b[0] for b in self.param_bounds],
                                       [b[1] for b in self.param_bounds])

                # 计算当前粒子的适应度
                current_fitness = self.fitness_func(particles[i], X_train, y_train)

                # 更新个体最优
                if current_fitness < p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = current_fitness

                # 更新全局最优
                if current_fitness < g_best_fitness:
                    improvement = g_best_fitness - current_fitness
                    g_best = particles[i].copy()
                    g_best_fitness = current_fitness
                    best_iteration = t
                    patience_counter = 0  # 重置早停计数器
                    print(f"迭代第{t + 1}次，全局最优RMSE：{g_best_fitness:.4f}，改进：{improvement:.6f}")
                else:
                    patience_counter += 1

            # 记录每代的全局最优适应度
            fitness_history.append(g_best_fitness)

            # 检查早停条件
            if patience_counter >= self.patience:
                print(f"\n早停触发！连续{self.patience}代无改进。")
                print(f"最佳RMSE：{g_best_fitness:.4f} 出现在第{best_iteration + 1}代")
                break

        self.best_iteration = best_iteration
        return g_best, fitness_history


# -------------------------- 3. 特征重要性分析 --------------------------
def analyze_feature_importance(model, X_test, y_test, feature_names, save_dir=None):
    """
    使用排列特征重要性分析
    通过打乱每个特征的值，观察模型性能下降程度来衡量特征重要性
    """
    print("\n" + "=" * 60)
    print("特征重要性分析")
    print("=" * 60)

    # 基准性能
    y_pred = model.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    importance_scores = []

    for i, feature_name in enumerate(feature_names):
        # 复制测试数据
        X_test_shuffled = X_test.copy()

        # 打乱第i个特征的值
        np.random.shuffle(X_test_shuffled[:, i])

        # 使用打乱后的数据预测
        y_pred_shuffled = model.predict(X_test_shuffled)

        # 计算打乱后的性能
        shuffled_rmse = np.sqrt(mean_squared_error(y_test, y_pred_shuffled))

        # 计算重要性分数（性能下降程度）
        importance = shuffled_rmse - baseline_rmse
        importance_scores.append((feature_name, importance))

        print(f"特征 '{feature_name}': 基准RMSE={baseline_rmse:.4f}, "
              f"打乱后RMSE={shuffled_rmse:.4f}, 重要性={importance:.4f}")

    # 按重要性排序
    importance_scores.sort(key=lambda x: x[1], reverse=True)

    # 可视化特征重要性
    fig, ax = plt.subplots(figsize=(12, 8))
    features = [score[0] for score in importance_scores]
    scores = [score[1] for score in importance_scores]

    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax.barh(range(len(features)), scores, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('重要性分数（RMSE增加量）', fontsize=12)
    ax.set_title('特征重要性分析', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 在条形图上添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()

    # 保存图像
    if save_dir:
        save_path = os.path.join(save_dir, 'feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")

    plt.show()

    return importance_scores


# -------------------------- 4. 波形可视化与保存功能 --------------------------
def visualize_wave_predictions(y_true, y_pred, save_dir=None, title="波浪预测波形图",
                               sample_size=200, show_waveform=True):
    """
    可视化并保存波浪预测波形图

    参数:
    - y_true: 真实值数组
    - y_pred: 预测值数组
    - save_dir: 保存目录
    - title: 图表标题
    - sample_size: 显示的样本数量
    - show_waveform: 是否显示波形图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 确保样本数量不超过数据长度
    sample_size = min(sample_size, len(y_true))

    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 1. 完整测试集预测对比
    x_full = np.arange(len(y_true))
    axes[0].plot(x_full, y_true, label='真实波浪高度', color='blue', linewidth=1.5, alpha=0.8)
    axes[0].plot(x_full, y_pred, label='预测波浪高度', color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    axes[0].set_xlabel('样本索引', fontsize=11)
    axes[0].set_ylabel('波高 (m)', fontsize=11)
    axes[0].set_title('完整测试集波浪预测对比', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 2. 局部波形对比（前sample_size个样本）
    x_sample = np.arange(sample_size)
    axes[1].plot(x_sample, y_true[:sample_size], label='真实波形', color='green',
                 linewidth=2, alpha=0.9, marker='o', markersize=3, markevery=10)
    axes[1].plot(x_sample, y_pred[:sample_size], label='预测波形', color='orange',
                 linewidth=2, alpha=0.8, linestyle='--', marker='s', markersize=3, markevery=10)
    axes[1].fill_between(x_sample, y_true[:sample_size], y_pred[:sample_size],
                         alpha=0.2, color='gray', label='误差区域')
    axes[1].set_xlabel('样本索引', fontsize=11)
    axes[1].set_ylabel('波高 (m)', fontsize=11)
    axes[1].set_title(f'前{sample_size}个样本波浪波形对比', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 添加误差统计
    mae = mean_absolute_error(y_true[:sample_size], y_pred[:sample_size])
    rmse = np.sqrt(mean_squared_error(y_true[:sample_size], y_pred[:sample_size]))
    axes[1].text(0.02, 0.98, f'MAE: {mae:.3f}m\nRMSE: {rmse:.3f}m',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 3. 残差分布图
    residuals = y_true - y_pred
    axes[2].scatter(np.arange(len(residuals)), residuals, alpha=0.6, color='purple', s=20)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[2].fill_between(x_full, -rmse, rmse, alpha=0.2, color='gray', label=f'±RMSE ({rmse:.3f}m)')
    axes[2].set_xlabel('样本索引', fontsize=11)
    axes[2].set_ylabel('残差 (m)', fontsize=11)
    axes[2].set_title('预测残差分布', fontsize=13, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # 添加残差统计信息
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    axes[2].text(0.02, 0.98, f'残差均值: {residual_mean:.4f}m\n残差标准差: {residual_std:.4f}m',
                 transform=axes[2].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存图像
    if save_dir:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'wave_prediction_{timestamp}.png')

        # 保存高分辨率图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n波形图已保存到: {save_path}")

        # 同时保存为PDF格式（矢量图）
        pdf_path = os.path.join(save_dir, f'wave_prediction_{timestamp}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"波形图PDF版本已保存到: {pdf_path}")

    # 显示图形
    if show_waveform:
        plt.show()
    else:
        plt.close()

    return fig


def create_waveform_animation(y_true, y_pred, save_dir=None, frames=50):
    """
    创建波形预测动画（可选功能）
    """
    try:
        from matplotlib.animation import FuncAnimation
        import matplotlib.animation as animation

        print("\n正在生成波形预测动画...")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0, 100)
        ax.set_ylim(min(np.min(y_true), np.min(y_pred)) - 0.5,
                    max(np.max(y_true), np.max(y_pred)) + 0.5)
        ax.set_xlabel('样本索引')
        ax.set_ylabel('波高 (m)')
        ax.set_title('波浪预测动态演示')
        ax.grid(True, alpha=0.3)

        line_true, = ax.plot([], [], 'b-', label='真实波形', linewidth=2)
        line_pred, = ax.plot([], [], 'r--', label='预测波形', linewidth=2)
        ax.legend()

        def init():
            line_true.set_data([], [])
            line_pred.set_data([], [])
            return line_true, line_pred

        def update(frame):
            x = np.arange(frame)
            line_true.set_data(x, y_true[:frame])
            line_pred.set_data(x, y_pred[:frame])
            return line_true, line_pred

        ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                            blit=True, interval=100)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(save_dir, f'wave_animation_{timestamp}.gif')
            ani.save(gif_path, writer='pillow', fps=10)
            print(f"动画已保存到: {gif_path}")

        plt.show()
        return ani

    except ImportError:
        print("警告: 无法生成动画，需要安装matplotlib.animation")
        return None


# -------------------------- 5. 模型保存和加载功能 --------------------------
def save_model(model, scaler_X, scaler_y, best_params, feature_names,
               performance_metrics=None, save_dir='saved_models'):
    """
    保存训练好的模型和归一化器
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f'model_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(model_dir, 'pso_svr_model.pkl')
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")

    # 保存归一化器
    scaler_X_path = os.path.join(model_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(model_dir, 'scaler_y.pkl')
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"归一化器已保存到: {scaler_X_path}, {scaler_y_path}")

    # 保存特征名称
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    joblib.dump(feature_names, feature_names_path)

    # 保存元数据
    metadata = {
        'best_params': best_params,
        'feature_names': feature_names,
        'timestamp': timestamp,
        'model_path': model_path,
        'scaler_X_path': scaler_X_path,
        'scaler_y_path': scaler_y_path,
        'performance_metrics': performance_metrics,
        'model_dir': model_dir
    }

    metadata_path = os.path.join(model_dir, 'metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"元数据已保存到: {metadata_path}")

    # 保存性能指标为文本文件
    if performance_metrics:
        metrics_path = os.path.join(model_dir, 'performance_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("模型性能指标\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"最优参数: {best_params}\n\n")

            for set_name, metrics in performance_metrics.items():
                f.write(f"{set_name}集评估指标:\n")
                f.write(f"  RMSE: {metrics['rmse']:.4f} m\n")
                f.write(f"  MAE: {metrics['mae']:.4f} m\n")
                f.write(f"  R²: {metrics['r2']:.4f}\n\n")

        print(f"性能指标已保存到: {metrics_path}")

    return metadata


def load_model(metadata_path):
    """
    加载保存的模型和归一化器
    """
    # 加载元数据
    metadata = joblib.load(metadata_path)

    # 加载模型和归一化器
    model = joblib.load(metadata['model_path'])
    scaler_X = joblib.load(metadata['scaler_X_path'])
    scaler_y = joblib.load(metadata['scaler_y_path'])

    print(f"模型和归一化器已加载")
    print(f"模型参数: {metadata['best_params']}")
    print(f"模型目录: {metadata.get('model_dir', 'N/A')}")

    return model, scaler_X, scaler_y, metadata


# -------------------------- 6. 结果可视化综合函数 --------------------------
def visualize_results(fitness_history, y_test_true, y_test_pred, pso_svr,
                      save_dir=None, show_plots=True):
    """
    综合可视化所有结果并保存
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    # 子图1：PSO适应度收敛曲线
    axes[0].plot(fitness_history, label='全局最优RMSE', color='blue', linewidth=2)
    axes[0].axvline(x=pso_svr.best_iteration, color='red', linestyle='--',
                    label=f'最佳迭代 ({pso_svr.best_iteration + 1})', alpha=0.7)
    axes[0].set_xlabel('迭代次数', fontsize=11)
    axes[0].set_ylabel('RMSE（m）', fontsize=11)
    axes[0].set_title('PSO优化收敛曲线（带早停机制）', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：测试集真实值vs预测值（散点图）
    axes[1].scatter(y_test_true, y_test_pred, alpha=0.6, color='green', s=30)
    # 添加理想预测线
    max_val = max(np.max(y_test_true), np.max(y_test_pred))
    min_val = min(np.min(y_test_true), np.min(y_test_pred))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                 label='理想预测线')
    axes[1].set_xlabel('真实有效波高（m）', fontsize=11)
    axes[1].set_ylabel('预测有效波高（m）', fontsize=11)
    axes[1].set_title('真实值 vs 预测值散点图', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 添加R²值
    r2 = r2_score(y_test_true, y_test_pred)
    axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图3：测试集真实值vs预测值（线图）
    t = np.arange(len(y_test_true))
    axes[2].plot(t, y_test_true, label='真实有效波高', color='green', linewidth=2)
    axes[2].plot(t, y_test_pred, label='预测有效波高', color='red',
                 linewidth=2, linestyle='--', alpha=0.8)
    axes[2].set_xlabel('测试样本序号', fontsize=11)
    axes[2].set_ylabel('有效波高（m）', fontsize=11)
    axes[2].set_title('测试集真实值与预测值对比', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 子图4：残差分析
    residuals = y_test_true - y_test_pred
    axes[3].scatter(y_test_pred, residuals, alpha=0.6, color='purple', s=30)
    axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[3].set_xlabel('预测值（m）', fontsize=11)
    axes[3].set_ylabel('残差（真实值-预测值）', fontsize=11)
    axes[3].set_title('残差分析图', fontsize=13, fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    # 添加残差统计信息
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    axes[3].text(0.05, 0.95, f'残差均值: {residual_mean:.4f}m\n残差标准差: {residual_std:.4f}m',
                 transform=axes[3].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图5：残差直方图
    axes[4].hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[4].axvline(x=residual_mean, color='red', linestyle='--', linewidth=2,
                    label=f'均值: {residual_mean:.4f}')
    axes[4].set_xlabel('残差（m）', fontsize=11)
    axes[4].set_ylabel('频数', fontsize=11)
    axes[4].set_title('残差分布直方图', fontsize=13, fontweight='bold')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    # 子图6：误差分布箱线图
    error_data = [residuals]
    bp = axes[5].boxplot(error_data, patch_artist=True, labels=['预测误差'])
    # 设置箱线图颜色
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    axes[5].set_ylabel('误差值（m）', fontsize=11)
    axes[5].set_title('预测误差箱线图', fontsize=13, fontweight='bold')
    axes[5].grid(True, alpha=0.3, axis='y')

    # 添加误差统计到箱线图
    q1, q3 = np.percentile(residuals, [25, 75])
    iqr = q3 - q1
    axes[5].text(0.05, 0.95, f'Q1: {q1:.4f}\nQ3: {q3:.4f}\nIQR: {iqr:.4f}',
                 transform=axes[5].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('PSO-SVR波浪预测模型结果综合分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存图像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'comprehensive_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n综合分析图已保存到: {save_path}")

    # 显示图形
    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


# -------------------------- 7. 主流程 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PSO-SVR波浪预测模型优化版本 - 带波形图保存功能")
    print("=" * 60)

    # 创建结果保存目录
    results_dir = "wave_prediction_results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到目录: {results_dir}")

    # 1. 生成数据（包含特征名称）
    X, y, feature_names = generate_wave_data(n_samples=1000)
    print(f"\n数据维度：输入特征{X.shape}，输出目标{y.shape}")
    print(f"特征名称：{feature_names}")

    # 2. 数据归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 3. 划分训练集/测试集（时间序列划分）
    train_size = int(0.7 * len(X_scaled))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    # 4. PSO优化SVR参数（带早停机制）
    print("\n" + "-" * 60)
    print("开始PSO参数优化（带早停机制）")
    print("-" * 60)

    pso_svr = PSO_SVR(n_particles=20, max_iter=100, patience=15)  # 增加最大迭代次数
    best_params, fitness_history = pso_svr.optimize(X_train, y_train)
    C_opt, gamma_opt, epsilon_opt = best_params

    print(f"\n优化结果：")
    print(f"最优参数：C={C_opt:.4f}, gamma={gamma_opt:.4f}, epsilon={epsilon_opt:.4f}")
    print(f"最佳适应度（RMSE）：{fitness_history[-1]:.4f}")
    print(f"收敛迭代次数：{pso_svr.best_iteration + 1}")

    # 5. 训练最终的PSO-SVR模型
    print("\n" + "-" * 60)
    print("训练最终模型")
    print("-" * 60)

    svr_model = SVR(kernel='rbf', C=C_opt, gamma=gamma_opt, epsilon=epsilon_opt)
    svr_model.fit(X_train, y_train)

    # 6. 预测（反归一化得到真实值）
    y_train_pred_scaled = svr_model.predict(X_train)
    y_test_pred_scaled = svr_model.predict(X_test)

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()


    # 7. 模型评估
    def evaluate(y_true, y_pred, set_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n{set_name}集评估指标：")
        print(f"RMSE（均方根误差）：{rmse:.4f} m")
        print(f"MAE（平均绝对误差）：{mae:.4f} m")
        print(f"R²（决定系数）：{r2:.4f}")
        return {'rmse': rmse, 'mae': mae, 'r2': r2}


    print("\n" + "=" * 60)
    print("模型性能评估")
    print("=" * 60)
    train_metrics = evaluate(y_train_true, y_train_pred, "训练")
    test_metrics = evaluate(y_test_true, y_test_pred, "测试")

    # 保存性能指标
    performance_metrics = {
        'train': train_metrics,
        'test': test_metrics
    }

    # 8. 特征重要性分析
    importance_scores = analyze_feature_importance(svr_model, X_test, y_test,
                                                   feature_names, save_dir=results_dir)

    # 9. 波形图可视化与保存
    print("\n" + "=" * 60)
    print("生成波浪预测波形图")
    print("=" * 60)

    # 生成并保存波形图
    wave_fig = visualize_wave_predictions(
        y_test_true,
        y_test_pred,
        save_dir=results_dir,
        title="PSO-SVR波浪预测波形对比",
        sample_size=200,
        show_waveform=True
    )

    # 可选：生成波形动画（如果需要）
    # create_waveform_animation(y_test_true[:100], y_test_pred[:100], save_dir=results_dir)

    # 10. 综合结果可视化
    print("\n" + "=" * 60)
    print("生成综合分析图表")
    print("=" * 60)

    comprehensive_fig = visualize_results(
        fitness_history,
        y_test_true,
        y_test_pred,
        pso_svr,
        save_dir=results_dir,
        show_plots=True
    )

    # 11. 保存模型
    print("\n" + "=" * 60)
    print("保存模型和归一化器")
    print("=" * 60)
    metadata = save_model(svr_model, scaler_X, scaler_y, best_params,
                          feature_names, performance_metrics, save_dir='saved_models')

    # 12. 演示模型加载功能
    print("\n" + "=" * 60)
    print("演示模型加载功能")
    print("=" * 60)
    try:
        loaded_model, loaded_scaler_X, loaded_scaler_y, loaded_metadata = load_model(metadata['metadata_path'])

        # 使用加载的模型进行预测
        X_sample = X_test[:5]
        y_sample_true = y_test_true[:5]

        # 注意：新数据需要先进行归一化
        X_sample_scaled = loaded_scaler_X.transform(X[:5])  # 使用原始X的样本进行演示

        y_sample_pred_scaled = loaded_model.predict(X_sample_scaled)
        y_sample_pred = loaded_scaler_y.inverse_transform(y_sample_pred_scaled.reshape(-1, 1)).flatten()

        print(f"\n样本预测测试（前5个样本）：")
        for i in range(5):
            print(f"样本{i + 1}: 真实值={y_sample_true[i]:.4f}, 预测值={y_sample_pred[i]:.4f}, "
                  f"误差={abs(y_sample_true[i] - y_sample_pred[i]):.4f}")
    except Exception as e:
        print(f"模型加载演示失败: {e}")

    # 13. 生成最终报告
    print("\n" + "=" * 60)
    print("优化总结")
    print("=" * 60)
    print(f"1. 模型性能: 测试集R² = {test_metrics['r2']:.4f}, RMSE = {test_metrics['rmse']:.4f}m")
    print(f"2. 特征重要性: 最重要的特征是'{importance_scores[0][0]}'")
    print(f"3. 波形图已保存到: {results_dir}/ 目录")
    print(f"4. 模型文件已保存到: saved_models/ 目录")
    print(f"5. 早停机制: 当连续{pso_svr.patience}代无改进时停止，实际迭代{pso_svr.best_iteration + 1}次")
    print("=" * 60)