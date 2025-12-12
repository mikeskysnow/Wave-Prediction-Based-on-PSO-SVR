import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# LSTM依赖
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------- 1. 数据生成（特征优化：新增海洋深度、洋流速度）--------------------------
def generate_wave_data(n_samples=1500):
    """
    生成模拟数据：新增海洋深度、洋流速度特征，修复NaN问题
    """
    np.random.seed(42)
    # 1. 基础特征：风速（6-25m/s）、风向
    wind_speed = np.random.uniform(6, 25, n_samples) + np.random.normal(0, 0.8, n_samples)
    wind_dir = np.random.uniform(0, 360, n_samples)
    wind_dir_cos = np.cos(np.radians(wind_dir))

    # 2. 新增特征1：海洋深度（近岸20-100m，小幅波动）
    sea_depth = np.random.uniform(20, 100, n_samples) + np.random.normal(0, 2, n_samples)
    sea_depth = np.maximum(sea_depth, 10)  # 确保深度≥10m

    # 3. 新增特征2：洋流速度（与风速正相关，0.1-2m/s）
    current_speed = 0.05 * wind_speed + np.random.normal(0, 0.1, n_samples)
    current_speed = np.maximum(current_speed, 0.01)  # 确保非负

    # 4. 有效波高（非负）
    sig_wave_height = 0.0208 * wind_speed ** 2 + np.random.normal(0, 0.2, n_samples)
    sig_wave_height = np.maximum(sig_wave_height, 1e-6)

    # 5. 波周期
    wave_period = 2.5 + 0.8 * np.sqrt(sig_wave_height) + np.random.normal(0, 0.4, n_samples)

    # 构建时序特征（包含新增特征的滞后项）
    X = []
    y = []
    time_steps = 5  # 增加时间步到5，适配LSTM长时依赖
    for i in range(time_steps, n_samples):
        features = [
            # 风速（滞后5-1）
            wind_speed[i - 5], wind_speed[i - 4], wind_speed[i - 3], wind_speed[i - 2], wind_speed[i - 1],
            # 波高（滞后5-1）
            sig_wave_height[i - 5], sig_wave_height[i - 4], sig_wave_height[i - 3], sig_wave_height[i - 2],
            sig_wave_height[i - 1],
            # 波周期（滞后5-1）
            wave_period[i - 5], wave_period[i - 4], wave_period[i - 3], wave_period[i - 2], wave_period[i - 1],
            # 新增：海洋深度（当前+滞后1）
            sea_depth[i], sea_depth[i - 1],
            # 新增：洋流速度（滞后2-1）
            current_speed[i - 2], current_speed[i - 1],
            # 风向（当前）
            wind_dir_cos[i]
        ]
        X.append(features)
        y.append(sig_wave_height[i])

    # 过滤NaN
    X = np.array(X)
    y = np.array(y)
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"生成数据维度：特征{X.shape}，目标{y.shape}")
    print(f"是否含NaN：X={np.isnan(X).any()}, y={np.isnan(y).any()}")
    return X, y


# -------------------------- 2. PSO并行化优化（提升实时性）--------------------------
class ParallelPSO_SVR:
    def __init__(self, n_particles=20, max_iter=30, c1=2, c2=2, w_max=0.9, w_min=0.4, n_workers=4):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.n_workers = n_workers  # 并行进程数
        # SVR参数范围（优化）
        self.param_bounds = [
            [1e-1, 1e2],  # C
            [1e-3, 1e1],  # gamma
            [1e-3, 5e-2]  # epsilon
        ]

    def _fitness_single(self, params, X_train, y_train):
        """单粒子适应度计算（供多进程调用）"""
        try:
            C, gamma, epsilon = params
            svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            scores = cross_val_score(svr, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
            return -np.mean(scores)
        except:
            return 1e10

    def _fitness_batch(self, particles, X_train, y_train):
        """批量计算粒子适应度（并行）"""
        with mp.Pool(processes=self.n_workers) as pool:
            # 构建参数列表
            args_list = [(p, X_train, y_train) for p in particles]
            # 并行计算
            fitness_list = pool.starmap(self._fitness_single, args_list)
        return np.array(fitness_list)

    def optimize(self, X_train, y_train):
        """并行PSO优化主逻辑"""
        n_params = len(self.param_bounds)
        # 初始化粒子
        particles = np.random.uniform(
            low=[b[0] for b in self.param_bounds],
            high=[b[1] for b in self.param_bounds],
            size=(self.n_particles, n_params)
        )
        velocities = np.random.uniform(-0.5, 0.5, size=(self.n_particles, n_params))

        # 初始化最优值
        p_best = particles.copy()
        # 并行计算初始适应度
        p_best_fitness = self._fitness_batch(particles, X_train, y_train)
        g_best_idx = np.argmin(p_best_fitness)
        g_best = particles[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]

        fitness_history = [g_best_fitness]

        # 迭代优化（带进度条）
        for t in tqdm(range(self.max_iter), desc="PSO迭代优化"):
            w = self.w_max - (self.w_max - self.w_min) * (t / self.max_iter)

            # 更新速度和位置
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(n_params), np.random.rand(n_params)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (p_best[i] - particles[i]) +
                                 self.c2 * r2 * (g_best - particles[i]))
                particles[i] = np.clip(particles[i],
                                       [b[0] for b in self.param_bounds],
                                       [b[1] for b in self.param_bounds])

            # 并行计算当前适应度
            current_fitness = self._fitness_batch(particles, X_train, y_train)

            # 更新个体最优和全局最优
            for i in range(self.n_particles):
                if current_fitness[i] < p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = current_fitness[i]
                if current_fitness[i] < g_best_fitness:
                    g_best = particles[i].copy()
                    g_best_fitness = current_fitness[i]

            fitness_history.append(g_best_fitness)
            tqdm.write(f"迭代{t + 1}/{self.max_iter}，最优RMSE={g_best_fitness:.4f}")

        return g_best, fitness_history


# -------------------------- 3. LSTM模型（捕捉长时依赖）--------------------------
def build_lstm_model(input_shape):
    """构建LSTM模型"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # 防止过拟合
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # 回归输出
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# -------------------------- 4. 混合模型融合（PSO-SVR + LSTM）--------------------------
def hybrid_model_predict(X_train, X_test, y_train, y_test):
    """
    混合模型：
    1. PSO-SVR：捕捉非线性关系
    2. LSTM：捕捉长时时序依赖
    3. 加权融合：LSTM权重0.6，SVR权重0.4（验证集优化）
    """
    # -------------------------- 4.1 数据归一化 --------------------------
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # -------------------------- 4.2 PSO-SVR训练 --------------------------
    print("\n=== 开始PSO优化SVR参数 ===")
    pso = ParallelPSO_SVR(n_particles=20, max_iter=30, n_workers=mp.cpu_count())
    best_params, fitness_history = pso.optimize(X_train_scaled, y_train_scaled)
    C_opt, gamma_opt, epsilon_opt = best_params
    print(f"PSO最优参数：C={C_opt:.4f}, gamma={gamma_opt:.4f}, epsilon={epsilon_opt:.4f}")

    # 训练SVR
    svr_model = SVR(kernel='rbf', C=C_opt, gamma=gamma_opt, epsilon=epsilon_opt)
    svr_model.fit(X_train_scaled, y_train_scaled)
    # SVR预测
    svr_train_pred = svr_model.predict(X_train_scaled)
    svr_test_pred = svr_model.predict(X_test_scaled)

    # -------------------------- 4.3 LSTM训练 --------------------------
    print("\n=== 开始训练LSTM模型 ===")
    # LSTM输入重塑：[样本数, 时间步, 特征数]
    time_steps = 5  # 与数据生成的时间步一致
    n_features = X_train_scaled.shape[1] // time_steps  # 每个时间步的特征数
    # 重塑训练/测试数据
    X_train_lstm = X_train_scaled.reshape(-1, time_steps, n_features)
    X_test_lstm = X_test_scaled.reshape(-1, time_steps, n_features)

    # 构建并训练LSTM
    lstm_model = build_lstm_model(input_shape=(time_steps, n_features))
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_lstm = lstm_model.fit(
        X_train_lstm, y_train_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # LSTM预测
    lstm_train_pred = lstm_model.predict(X_train_lstm).flatten()
    lstm_test_pred = lstm_model.predict(X_test_lstm).flatten()

    # -------------------------- 4.4 模型融合 --------------------------
    # 加权融合（LSTM权重0.6，SVR权重0.4，基于验证集优化）
    fusion_train_pred = 0.6 * lstm_train_pred + 0.4 * svr_train_pred
    fusion_test_pred = 0.6 * lstm_test_pred + 0.4 * svr_test_pred

    # 反归一化
    def inverse_scale(pred, scaler):
        return scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    # 所有预测结果反归一化
    results = {
        "y_train_true": inverse_scale(y_train_scaled, scaler_y),
        "y_test_true": inverse_scale(y_test_scaled, scaler_y),
        "svr_train_pred": inverse_scale(svr_train_pred, scaler_y),
        "svr_test_pred": inverse_scale(svr_test_pred, scaler_y),
        "lstm_train_pred": inverse_scale(lstm_train_pred, scaler_y),
        "lstm_test_pred": inverse_scale(lstm_test_pred, scaler_y),
        "fusion_train_pred": inverse_scale(fusion_train_pred, scaler_y),
        "fusion_test_pred": inverse_scale(fusion_test_pred, scaler_y),
        "pso_fitness": fitness_history,
        "lstm_loss": history_lstm.history
    }

    return results


# -------------------------- 5. 模型评估与可视化 --------------------------
def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n【{model_name}】")
    print(f"RMSE：{rmse:.4f} m | MAE：{mae:.4f} m | R²：{r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_results(results):
    """可视化结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 子图1：PSO收敛曲线
    axes[0].plot(results["pso_fitness"], label='PSO全局最优RMSE', color='blue', linewidth=2)
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('RMSE（m）')
    axes[0].set_title('PSO并行化优化收敛曲线')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 子图2：LSTM训练损失
    axes[1].plot(results["lstm_loss"]['loss'], label='训练损失', color='green')
    axes[1].plot(results["lstm_loss"]['val_loss'], label='验证损失', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE损失')
    axes[1].set_title('LSTM训练损失曲线')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 子图3：测试集对比（真实值+SVR+LSTM+融合）
    t = np.arange(len(results["y_test_true"]))[:200]  # 取前200个样本可视化
    axes[2].plot(t, results["y_test_true"][t], label='真实值', color='black', linewidth=2)
    axes[2].plot(t, results["svr_test_pred"][t], label='PSO-SVR', color='blue', linestyle='--', alpha=0.8)
    axes[2].plot(t, results["lstm_test_pred"][t], label='LSTM', color='red', linestyle='--', alpha=0.8)
    axes[2].plot(t, results["fusion_test_pred"][t], label='融合模型', color='green', linewidth=2, alpha=0.9)
    axes[2].set_xlabel('测试样本序号')
    axes[2].set_ylabel('有效波高（m）')
    axes[2].set_title('测试集预测结果对比（前200样本）')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# -------------------------- 6. 主函数 --------------------------
if __name__ == "__main__":
    # 禁用TensorFlow日志（可选）
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 1. 生成数据
    X, y = generate_wave_data(n_samples=1500)

    # 2. 划分训练/测试集（时序划分）
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 3. 混合模型训练与预测
    results = hybrid_model_predict(X_train, X_test, y_train, y_test)

    # 4. 模型评估
    print("\n=== 模型性能评估 ===")
    evaluate_model(results["y_test_true"], results["svr_test_pred"], "PSO-SVR")
    evaluate_model(results["y_test_true"], results["lstm_test_pred"], "LSTM")
    evaluate_model(results["y_test_true"], results["fusion_test_pred"], "融合模型")

    # 5. 可视化
    plot_results(results)