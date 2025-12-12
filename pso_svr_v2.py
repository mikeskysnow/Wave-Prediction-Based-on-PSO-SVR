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
    feature_names = []  # 特征名称列表

    # 定义特征名称
    for i in range(3, n_samples):  # 前3个时刻的特征预测当前波高
        features = [
            wind_speed[i - 3], wind_speed[i - 2], wind_speed[i - 1],  # 滞后3/2/1时刻风速
            sig_wave_height[i - 3], sig_wave_height[i - 2], sig_wave_height[i - 1],  # 滞后3/2/1时刻波高
            wave_period[i - 2], wave_period[i - 1],  # 滞后2/1时刻波周期
            wind_dir_cos[i]  # 当前时刻风向余弦
        ]
        X.append(features)
        y.append(sig_wave_height[i])

    # 创建特征名称（只在第一次迭代时）
    if not feature_names:
        feature_names = [
            'wind_speed_t-3', 'wind_speed_t-2', 'wind_speed_t-1',
            'wave_height_t-3', 'wave_height_t-2', 'wave_height_t-1',
            'wave_period_t-2', 'wave_period_t-1',
            'wind_dir_cos_t'
        ]

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
def analyze_feature_importance(model, X_test, y_test, feature_names):
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
    fig, ax = plt.subplots(figsize=(10, 6))
    features = [score[0] for score in importance_scores]
    scores = [score[1] for score in importance_scores]

    bars = ax.barh(range(len(features)), scores, color='skyblue')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('重要性分数（RMSE增加量）')
    ax.set_title('特征重要性分析')

    # 在条形图上添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', ha='left', va='center')

    plt.tight_layout()
    plt.show()

    return importance_scores


# -------------------------- 4. 模型保存和加载功能 --------------------------
def save_model(model, scaler_X, scaler_y, best_params, feature_names, save_dir='saved_models'):
    """
    保存训练好的模型和归一化器
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存模型
    model_path = os.path.join(save_dir, f'pso_svr_model_{timestamp}.pkl')
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")

    # 保存归一化器
    scaler_X_path = os.path.join(save_dir, f'scaler_X_{timestamp}.pkl')
    scaler_y_path = os.path.join(save_dir, f'scaler_y_{timestamp}.pkl')
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"归一化器已保存到: {scaler_X_path}, {scaler_y_path}")

    # 保存元数据
    metadata = {
        'best_params': best_params,
        'feature_names': feature_names,
        'timestamp': timestamp,
        'model_path': model_path,
        'scaler_X_path': scaler_X_path,
        'scaler_y_path': scaler_y_path
    }

    metadata_path = os.path.join(save_dir, f'metadata_{timestamp}.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"元数据已保存到: {metadata_path}")

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

    return model, scaler_X, scaler_y, metadata


# -------------------------- 5. 主流程 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PSO-SVR波浪预测模型优化版本")
    print("=" * 60)

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
        return rmse, mae, r2


    print("\n" + "=" * 60)
    print("模型性能评估")
    print("=" * 60)
    evaluate(y_train_true, y_train_pred, "训练")
    evaluate(y_test_true, y_test_pred, "测试")

    # 8. 特征重要性分析
    importance_scores = analyze_feature_importance(svr_model, X_test, y_test, feature_names)

    # 9. 保存模型
    print("\n" + "=" * 60)
    print("保存模型和归一化器")
    print("=" * 60)
    metadata = save_model(svr_model, scaler_X, scaler_y, best_params, feature_names)

    # 10. 演示模型加载功能
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

    # 11. 结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 支持负号
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    # 子图1：PSO适应度收敛曲线
    axes[0].plot(fitness_history, label='全局最优RMSE', color='blue', linewidth=2)
    axes[0].axvline(x=pso_svr.best_iteration, color='red', linestyle='--',
                    label=f'最佳迭代 ({pso_svr.best_iteration + 1})', alpha=0.7)
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('RMSE（m）')
    axes[0].set_title('PSO优化收敛曲线（带早停机制）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：测试集真实值vs预测值
    t = np.arange(len(y_test_true))
    axes[1].plot(t, y_test_true, label='真实有效波高', color='green', linewidth=2)
    axes[1].plot(t, y_test_pred, label='预测有效波高', color='red', linewidth=2, linestyle='--')
    axes[1].set_xlabel('测试样本序号')
    axes[1].set_ylabel('有效波高（m）')
    axes[1].set_title('测试集真实值与预测值对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3：残差分析
    residuals = y_test_true - y_test_pred
    axes[2].scatter(y_test_pred, residuals, alpha=0.6, color='purple')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[2].set_xlabel('预测值（m）')
    axes[2].set_ylabel('残差（真实值-预测值）')
    axes[2].set_title('残差分析图')
    axes[2].grid(True, alpha=0.3)

    # 添加残差统计信息
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    axes[2].text(0.05, 0.95, f'残差均值: {residual_mean:.4f}\n残差标准差: {residual_std:.4f}',
                 transform=axes[2].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("优化总结")
    print("=" * 60)
    print("1. 时间序列交叉验证：在PSO适应度函数中使用TimeSeriesSplit")
    print("2. 特征重要性分析：使用排列特征重要性方法")
    print(f"3. 早停机制：当连续{pso_svr.patience}代无改进时停止")
    print(f"4. 模型保存：模型和相关文件保存在'saved_models'目录")
    print("=" * 60)