import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

    return X, y


# -------------------------- 2. PSO优化SVR参数 --------------------------
class PSO_SVR:
    def __init__(self, n_particles=20, max_iter=50, c1=2, c2=2, w_max=0.9, w_min=0.4):
        self.n_particles = n_particles  # 粒子数
        self.max_iter = max_iter  # 最大迭代次数
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 全局学习因子
        self.w_max = w_max  # 最大惯性权重
        self.w_min = w_min  # 最小惯性权重

        # SVR参数搜索范围（优化范围，避免极端值）
        self.param_bounds = [
            [1e-1, 1e2],  # C的范围（缩小下限，避免过拟合）
            [1e-3, 1e1],  # gamma的范围（缩小范围，提升稳定性）
            [1e-3, 5e-2]  # epsilon的范围（优化区间，适配数据）
        ]

    def fitness_func(self, params, X_train, y_train):
        """适应度函数：SVR的5折交叉验证RMSE"""
        C, gamma, epsilon = params
        try:
            svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            # 交叉验证（负RMSE，sklearn默认最大化）
            scores = cross_val_score(svr, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
            return -np.mean(scores)  # 转换为最小化目标
        except Exception as e:
            # 若参数异常导致训练失败，返回极大值（淘汰该粒子）
            return 1e10

    def optimize(self, X_train, y_train):
        """PSO优化过程"""
        n_params = len(self.param_bounds)
        # 1. 初始化粒子位置和速度
        particles = np.random.uniform(
            low=[b[0] for b in self.param_bounds],
            high=[b[1] for b in self.param_bounds],
            size=(self.n_particles, n_params)
        )
        velocities = np.random.uniform(-0.5, 0.5, size=(self.n_particles, n_params))  # 缩小速度范围，提升稳定性

        # 2. 初始化个体最优和全局最优
        p_best = particles.copy()  # 个体最优位置
        p_best_fitness = np.array([self.fitness_func(p, X_train, y_train) for p in particles])
        g_best_idx = np.argmin(p_best_fitness)  # 全局最优索引
        g_best = particles[g_best_idx].copy()  # 全局最优位置
        g_best_fitness = p_best_fitness[g_best_idx]

        # 记录迭代过程的适应度值（用于绘图）
        fitness_history = [g_best_fitness]

        # 3. 迭代优化
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
                    g_best = particles[i].copy()
                    g_best_fitness = current_fitness

            # 记录每代的全局最优适应度
            fitness_history.append(g_best_fitness)
            print(f"迭代第{t + 1}次，全局最优RMSE：{g_best_fitness:.4f}")

        return g_best, fitness_history


# -------------------------- 3. 主流程：数据预处理→PSO优化→模型训练→预测评估 --------------------------
if __name__ == "__main__":
    # 1. 生成数据（修复后无NaN）
    X, y = generate_wave_data(n_samples=1000)
    print(f"数据维度：输入特征{X.shape}，输出目标{y.shape}")
    print(f"输入特征是否含NaN：{np.isnan(X).any()}")  # 验证无NaN
    print(f"输出目标是否含NaN：{np.isnan(y).any()}")  # 验证无NaN

    # 2. 数据归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 3. 划分训练集/测试集（时间序列划分）
    train_size = int(0.7 * len(X_scaled))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    # 4. PSO优化SVR参数
    pso_svr = PSO_SVR(n_particles=20, max_iter=50)
    best_params, fitness_history = pso_svr.optimize(X_train, y_train)
    C_opt, gamma_opt, epsilon_opt = best_params
    print(f"\nPSO优化得到的最优参数：C={C_opt:.4f}, gamma={gamma_opt:.4f}, epsilon={epsilon_opt:.4f}")

    # 5. 训练PSO-SVR模型
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


    evaluate(y_train_true, y_train_pred, "训练")
    evaluate(y_test_true, y_test_pred, "测试")

    # 8. 结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 支持负号
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：PSO适应度收敛曲线
    axes[0].plot(fitness_history, label='全局最优RMSE', color='blue', linewidth=2)
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('RMSE（m）')
    axes[0].set_title('PSO优化收敛曲线')
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

    plt.tight_layout()
    plt.show()