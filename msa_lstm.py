import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ====================== 1. 模拟船舶运动数据生成 ======================
def generate_ship_motion_data(n_samples=5000, seq_length=100):
    """
    生成模拟的船舶运动数据
    假设船舶运动包含4个主要模态分量：
    1. 高频模态（波浪激励响应）：频率 ~ 0.5-1Hz
    2. 中频模态（船体刚体运动）：频率 ~ 0.1-0.3Hz
    3. 低频模态（慢漂运动）：频率 ~ 0.01-0.05Hz
    4. 噪声
    """
    t = np.linspace(0, 100, seq_length)

    # 1. 高频模态 (波浪激励)
    freq_high = 0.8 + 0.4 * np.random.randn(n_samples, 1)
    amp_high = 1.0 + 0.3 * np.random.randn(n_samples, 1)
    phase_high = 2 * np.pi * np.random.rand(n_samples, 1)
    high_freq_component = amp_high * np.sin(2 * np.pi * freq_high * t.reshape(1, -1) + phase_high)

    # 添加高频模态的幅度调制（模拟波浪群）
    envelope_high = 0.5 * (1 + np.sin(2 * np.pi * 0.05 * t.reshape(1, -1) + np.random.randn(n_samples, 1)))
    high_freq_component *= envelope_high

    # 2. 中频模态 (船体刚体运动)
    freq_mid = 0.2 + 0.1 * np.random.randn(n_samples, 1)
    amp_mid = 1.5 + 0.5 * np.random.randn(n_samples, 1)
    phase_mid = 2 * np.pi * np.random.rand(n_samples, 1)
    mid_freq_component = amp_mid * np.sin(2 * np.pi * freq_mid * t.reshape(1, -1) + phase_mid)

    # 3. 低频模态 (慢漂运动)
    freq_low = 0.03 + 0.02 * np.random.randn(n_samples, 1)
    amp_low = 0.8 + 0.2 * np.random.randn(n_samples, 1)
    phase_low = 2 * np.pi * np.random.rand(n_samples, 1)
    low_freq_component = amp_low * np.sin(2 * np.pi * freq_low * t.reshape(1, -1) + phase_low)

    # 4. 噪声 (高斯白噪声 + 冲击噪声)
    noise = 0.1 * np.random.randn(n_samples, seq_length)
    # 添加随机冲击（模拟异常波浪冲击）
    for i in range(n_samples):
        n_shocks = np.random.randint(0, 3)
        for _ in range(n_shocks):
            idx = np.random.randint(0, seq_length)
            noise[i, idx] += 2 * np.random.randn()

    # 合成船舶运动信号
    ship_motion = high_freq_component + mid_freq_component + low_freq_component + noise

    # 模态分量作为标签（用于监督学习）
    components = np.stack([high_freq_component, mid_freq_component, low_freq_component, noise], axis=1)

    return ship_motion, components, t


# 生成数据
print("生成模拟船舶运动数据...")
ship_motion, components, time_points = generate_ship_motion_data(n_samples=2000, seq_length=100)
print(f"数据形状: ship_motion={ship_motion.shape}, components={components.shape}")


# ====================== 2. 数据预处理 ======================
class ShipMotionDataset(Dataset):
    def __init__(self, signals, components):
        self.signals = signals
        self.components = components

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # 转换为float32并添加通道维度
        signal = torch.FloatTensor(self.signals[idx]).unsqueeze(0)  # [1, seq_len]
        component = torch.FloatTensor(self.components[idx])  # [4, seq_len]
        return signal, component


# 数据集划分
train_size = int(0.7 * len(ship_motion))
val_size = int(0.15 * len(ship_motion))

train_signals = ship_motion[:train_size]
train_components = components[:train_size]

val_signals = ship_motion[train_size:train_size + val_size]
val_components = components[train_size:train_size + val_size]

test_signals = ship_motion[train_size + val_size:]
test_components = components[train_size + val_size:]

# 创建数据集和数据加载器
train_dataset = ShipMotionDataset(train_signals, train_components)
val_dataset = ShipMotionDataset(val_signals, val_components)
test_dataset = ShipMotionDataset(test_signals, test_components)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")


# ====================== 3. MSA-LSTM模型定义 ======================
class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # 线性变换并重塑为多头
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(scores, dim=-1)

        # 应用注意力到V
        out = torch.matmul(attention, V)

        # 重塑并应用输出线性层
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)

        return out, attention


class MSA_LSTM_Decomposition(nn.Module):
    """MSA-LSTM船舶运动模态分解模型"""

    def __init__(self, input_dim=1, hidden_dim=128, num_heads=4, num_components=4, num_lstm_layers=2):
        super().__init__()
        self.num_components = num_components
        self.hidden_dim = hidden_dim

        # 编码器: 1D卷积用于初步特征提取
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM编码器
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # 多头自注意力层
        self.multihead_attention = MultiHeadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM
            num_heads=num_heads
        )

        # 模态特定的LSTM解码器（每个模态一个）
        self.component_decoders = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_dim * 2,  # 保持不变
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            ) for _ in range(num_components)
        ])

        # 模态特定的输出层
        self.component_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_components)
        ])

        # 残差连接 - 修复：输入输出维度保持一致
        self.residual_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        batch_size, _, seq_len = x.shape

        # 编码
        encoded = self.encoder(x)  # [batch_size, hidden_dim, seq_len]
        encoded = encoded.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]

        # LSTM编码
        lstm_out, _ = self.lstm_encoder(encoded)  # [batch_size, seq_len, hidden_dim*2]

        # 多头自注意力
        attention_out, attention_weights = self.multihead_attention(lstm_out)

        # 残差连接 - 修复：维度保持 hidden_dim*2
        combined = lstm_out + self.residual_projection(attention_out)

        # 模态分解（每个模态使用独立的解码器）
        components = []
        for i in range(self.num_components):
            # 模态特定的LSTM解码
            component_lstm, _ = self.component_decoders[i](combined)
            # 模态特定的输出层
            component = self.component_outputs[i](component_lstm)
            components.append(component)

        # 拼接所有模态
        decomposed = torch.cat(components, dim=-1)  # [batch_size, seq_len, num_components]
        decomposed = decomposed.permute(0, 2, 1)  # [batch_size, num_components, seq_len]

        # 确保所有模态之和等于原始信号（重构约束）
        reconstructed = torch.sum(decomposed, dim=1, keepdim=True)  # [batch_size, 1, seq_len]

        return decomposed, reconstructed, attention_weights


# ====================== 4. 训练函数 ======================
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """训练模型"""
    model = model.to(device)

    # 定义损失函数和优化器
    criterion_reconstruction = nn.MSELoss()  # 重构损失
    criterion_components = nn.MSELoss()  # 模态分解损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_idx, (signals, components) in enumerate(train_loader):
            signals, components = signals.to(device), components.to(device)

            # 前向传播
            decomposed, reconstructed, _ = model(signals)

            # 计算损失
            # 1. 重构损失：确保分解后的分量之和等于原始信号
            loss_reconstruction = criterion_reconstruction(reconstructed, signals)

            # 2. 模态分解损失：分解的分量应接近真实分量（如果有标签）
            loss_components = criterion_components(decomposed, components)

            # 3. 稀疏性约束：鼓励某些模态分量稀疏
            # 这里假设噪声分量应该是稀疏的
            sparsity_loss = torch.mean(torch.abs(decomposed[:, 3, :]))  # 第4个分量（噪声）

            # 4. 平滑性约束：鼓励低频模态平滑
            # 计算低频模态（第3个分量）的梯度
            low_freq_component = decomposed[:, 2, :]
            smoothness_loss = torch.mean(torch.abs(low_freq_component[:, 1:] - low_freq_component[:, :-1]))

            # 总损失
            total_loss = loss_reconstruction + 0.5 * loss_components + 0.1 * sparsity_loss + 0.05 * smoothness_loss

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 50 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {total_loss.item():.6f}')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signals, components in val_loader:
                signals, components = signals.to(device), components.to(device)
                decomposed, reconstructed, _ = model(signals)

                loss_reconstruction = criterion_reconstruction(reconstructed, signals)
                loss_components = criterion_components(decomposed, components)
                total_val_loss = loss_reconstruction + 0.5 * loss_components
                val_loss += total_val_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 学习率调整
        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    return model, train_losses, val_losses


# ====================== 5. 模型训练 ======================
# 创建模型
model = MSA_LSTM_Decomposition(
    input_dim=1,
    hidden_dim=128,
    num_heads=4,
    num_components=4,
    num_lstm_layers=2
)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练模型
trained_model, train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=30,
    learning_rate=0.001
)


# ====================== 6. 测试和可视化 ======================
def plot_decomposition_results(model, test_loader, time_points, num_samples=3):
    """绘制分解结果"""
    model.eval()

    # 获取测试样本
    signals_list = []
    components_list = []
    decomposed_list = []
    attention_list = []

    with torch.no_grad():
        for i, (signals, components) in enumerate(test_loader):
            if i >= 1:  # 只取一个批次
                break

            signals, components = signals.to(device), components.to(device)
            decomposed, reconstructed, attention_weights = model(signals)

            signals_list.append(signals.cpu().numpy())
            components_list.append(components.cpu().numpy())
            decomposed_list.append(decomposed.cpu().numpy())
            attention_list.append(attention_weights.cpu().numpy())

    # 转换为numpy数组
    if signals_list:
        signals = np.concatenate(signals_list, axis=0)
        true_components = np.concatenate(components_list, axis=0)
        decomposed = np.concatenate(decomposed_list, axis=0)
        attention = np.concatenate(attention_list, axis=0)
    else:
        print("测试集为空")
        return None, None, None, None

    # 绘制多个样本的分解结果
    num_samples_to_plot = min(num_samples, len(signals))

    if num_samples_to_plot == 0:
        print("没有可绘制的样本")
        return signals, true_components, decomposed, attention

    fig, axes = plt.subplots(num_samples_to_plot, 5, figsize=(20, 4 * num_samples_to_plot))

    if num_samples_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    component_names = ['高频模态', '中频模态', '低频模态', '噪声']

    for sample_idx in range(num_samples_to_plot):
        # 原始信号和重构信号
        axes[sample_idx, 0].plot(time_points, signals[sample_idx, 0, :], 'b-', label='原始信号', alpha=0.7)
        axes[sample_idx, 0].plot(time_points, np.sum(decomposed[sample_idx], axis=0), 'r--', label='重构信号',
                                 alpha=0.7)
        axes[sample_idx, 0].set_title(f'样本 {sample_idx + 1}: 原始信号 vs 重构信号')
        axes[sample_idx, 0].set_xlabel('时间')
        axes[sample_idx, 0].set_ylabel('幅值')
        axes[sample_idx, 0].legend()
        axes[sample_idx, 0].grid(True, alpha=0.3)

        # 模态分解对比（真实 vs 预测）
        for comp_idx in range(4):
            axes[sample_idx, comp_idx + 1].plot(time_points, true_components[sample_idx, comp_idx, :],
                                                'b-', label='真实', alpha=0.7)
            axes[sample_idx, comp_idx + 1].plot(time_points, decomposed[sample_idx, comp_idx, :],
                                                'r--', label='预测', alpha=0.7)
            axes[sample_idx, comp_idx + 1].set_title(f'{component_names[comp_idx]}')
            axes[sample_idx, comp_idx + 1].set_xlabel('时间')
            axes[sample_idx, comp_idx + 1].set_ylabel('幅值')
            axes[sample_idx, comp_idx + 1].legend()
            axes[sample_idx, comp_idx + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('modal_decomposition_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return signals, true_components, decomposed, attention


def plot_attention_analysis(attention_weights, time_points, sample_idx=0, head_idx=0):
    """绘制注意力权重分析"""
    if attention_weights is None:
        print("注意力权重为空")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 注意力热力图
    im = axes[0, 0].imshow(attention_weights[sample_idx, head_idx],
                           aspect='auto', cmap='viridis')
    axes[0, 0].set_title(f'样本 {sample_idx + 1}, 注意力头 {head_idx + 1} 的注意力权重')
    axes[0, 0].set_xlabel('查询位置')
    axes[0, 0].set_ylabel('键位置')
    plt.colorbar(im, ax=axes[0, 0])

    # 特定时间点的注意力分布
    time_point = min(50, attention_weights.shape[2] - 1)  # 选择中间时间点
    axes[0, 1].plot(time_points[:attention_weights.shape[3]], attention_weights[sample_idx, head_idx, time_point, :],
                    'b-')
    axes[0, 1].set_title(f'时间点 {time_point} 的注意力分布')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('注意力权重')
    axes[0, 1].grid(True, alpha=0.3)

    # 平均注意力权重
    mean_attention = np.mean(attention_weights[sample_idx, head_idx], axis=0)
    axes[1, 0].plot(time_points[:len(mean_attention)], mean_attention, 'r-')
    axes[1, 0].set_title('平均注意力权重')
    axes[1, 0].set_xlabel('时间')
    axes[1, 0].set_ylabel('平均权重')
    axes[1, 0].grid(True, alpha=0.3)

    # 不同注意力头的对比
    num_heads = min(4, attention_weights.shape[1])
    colors = ['b', 'r', 'g', 'm']
    for h in range(num_heads):
        axes[1, 1].plot(time_points[:attention_weights.shape[3]],
                        np.mean(attention_weights[sample_idx, h], axis=0),
                        color=colors[h], label=f'头{h + 1}', alpha=0.7)

    axes[1, 1].set_title('不同注意力头的对比')
    axes[1, 1].set_xlabel('时间')
    axes[1, 1].set_ylabel('平均注意力权重')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_frequency_analysis(signals, decomposed, time_points, sample_idx=0):
    """绘制频率分析"""
    from scipy import signal

    if signals is None or decomposed is None:
        print("信号数据为空")
        return

    # 计算采样频率
    if len(time_points) > 1:
        fs = 1 / (time_points[1] - time_points[0])
    else:
        fs = 10  # 默认采样频率

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # 原始信号的频谱
    f, Pxx = signal.welch(signals[sample_idx, 0, :], fs, nperseg=min(64, len(time_points) // 2))
    axes[0, 0].semilogy(f, Pxx, 'b-')
    axes[0, 0].set_title('原始信号的功率谱密度')
    axes[0, 0].set_xlabel('频率 [Hz]')
    axes[0, 0].set_ylabel('PSD')
    axes[0, 0].grid(True, alpha=0.3)

    # 各模态分量的频谱
    component_names = ['高频模态', '中频模态', '低频模态', '噪声']
    colors = ['r', 'g', 'b', 'm']

    for comp_idx in range(4):
        f, Pxx = signal.welch(decomposed[sample_idx, comp_idx, :], fs, nperseg=min(64, len(time_points) // 2))
        axes[0, 1].semilogy(f, Pxx, color=colors[comp_idx], label=component_names[comp_idx], alpha=0.7)

    axes[0, 1].set_title('各模态分量的功率谱密度')
    axes[0, 1].set_xlabel('频率 [Hz]')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 时频分析 - 原始信号
    frequencies, times, Sxx = signal.spectrogram(signals[sample_idx, 0, :], fs, nperseg=min(32, len(time_points) // 4))
    if Sxx.size > 0:
        im1 = axes[1, 0].pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        axes[1, 0].set_title('原始信号的时频分析')
        axes[1, 0].set_xlabel('时间 [s]')
        axes[1, 0].set_ylabel('频率 [Hz]')
        plt.colorbar(im1, ax=axes[1, 0], label='功率 [dB]')
    else:
        axes[1, 0].text(0.5, 0.5, '时频分析数据不足', ha='center', va='center')
        axes[1, 0].set_title('原始信号的时频分析')

    # 时频分析 - 重构信号
    reconstructed = np.sum(decomposed[sample_idx], axis=0)
    frequencies, times, Sxx = signal.spectrogram(reconstructed, fs, nperseg=min(32, len(time_points) // 4))
    if Sxx.size > 0:
        im2 = axes[1, 1].pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        axes[1, 1].set_title('重构信号的时频分析')
        axes[1, 1].set_xlabel('时间 [s]')
        axes[1, 1].set_ylabel('频率 [Hz]')
        plt.colorbar(im2, ax=axes[1, 1], label='功率 [dB]')
    else:
        axes[1, 1].text(0.5, 0.5, '时频分析数据不足', ha='center', va='center')
        axes[1, 1].set_title('重构信号的时频分析')

    # 累积能量分布
    axes[2, 0].plot(time_points, np.cumsum(np.abs(signals[sample_idx, 0, :])), 'b-', label='原始信号')
    axes[2, 0].plot(time_points, np.cumsum(np.abs(decomposed[sample_idx, 0, :])), 'r-', label='高频模态', alpha=0.7)
    axes[2, 0].plot(time_points, np.cumsum(np.abs(decomposed[sample_idx, 1, :])), 'g-', label='中频模态', alpha=0.7)
    axes[2, 0].plot(time_points, np.cumsum(np.abs(decomposed[sample_idx, 2, :])), 'm-', label='低频模态', alpha=0.7)
    axes[2, 0].set_title('累积能量分布')
    axes[2, 0].set_xlabel('时间')
    axes[2, 0].set_ylabel('累积能量')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 模态能量占比
    energies = [np.sum(np.abs(decomposed[sample_idx, i, :]) ** 2) for i in range(4)]
    total_energy = np.sum(energies)
    energy_ratios = [e / total_energy for e in energies]

    axes[2, 1].pie(energy_ratios, labels=component_names, autopct='%1.1f%%', colors=colors)
    axes[2, 1].set_title('模态能量占比')

    plt.tight_layout()
    plt.savefig('frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# 绘制训练损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', label='训练损失', alpha=0.7)
plt.plot(val_losses, 'r-', label='验证损失', alpha=0.7)
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.title('MSA-LSTM模型训练过程')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# 测试模型并可视化结果
print("\n测试模型并可视化结果...")
signals, true_components, decomposed, attention = plot_decomposition_results(
    trained_model, test_loader, time_points, num_samples=3
)

if signals is not None:
    # 注意力分析
    plot_attention_analysis(attention, time_points, sample_idx=0, head_idx=0)

    # 频率分析
    plot_frequency_analysis(signals, decomposed, time_points, sample_idx=0)


    # ====================== 7. 性能评估 ======================
    def evaluate_model(model, test_loader):
        """评估模型性能"""
        model.eval()

        reconstruction_errors = []
        component_errors = []
        sparsity_scores = []

        with torch.no_grad():
            for signals, components in test_loader:
                signals, components = signals.to(device), components.to(device)
                decomposed, reconstructed, _ = model(signals)

                # 重构误差
                reconstruction_error = torch.mean((reconstructed - signals) ** 2, dim=(1, 2))
                reconstruction_errors.extend(reconstruction_error.cpu().numpy())

                # 模态分解误差
                component_error = torch.mean((decomposed - components) ** 2, dim=(1, 2))
                component_errors.extend(component_error.cpu().numpy())

                # 稀疏性得分（噪声分量）
                noise_component = decomposed[:, 3, :]
                sparsity_score = torch.mean(torch.abs(noise_component), dim=1)
                sparsity_scores.extend(sparsity_score.cpu().numpy())

        if reconstruction_errors:
            # 计算统计指标
            reconstruction_mean = np.mean(reconstruction_errors)
            reconstruction_std = np.std(reconstruction_errors)

            component_mean = np.mean(component_errors)
            component_std = np.std(component_errors)

            sparsity_mean = np.mean(sparsity_scores)
            sparsity_std = np.std(sparsity_scores)

            print("=" * 50)
            print("模型性能评估结果:")
            print("=" * 50)
            print(f"重构误差: {reconstruction_mean:.6f} ± {reconstruction_std:.6f}")
            print(f"模态分解误差: {component_mean:.6f} ± {component_std:.6f}")
            print(f"噪声分量稀疏性: {sparsity_mean:.6f} ± {sparsity_std:.6f}")
            print("=" * 50)
        else:
            print("测试集为空")

        return reconstruction_errors, component_errors, sparsity_scores


    # 评估模型
    reconstruction_errors, component_errors, sparsity_scores = evaluate_model(trained_model, test_loader)
else:
    print("无法获取测试结果")


# ====================== 8. 实际应用示例 ======================
def apply_to_realistic_scenario():
    """模拟实际船舶运动场景"""
    print("\n" + "=" * 50)
    print("模拟实际船舶运动场景应用")
    print("=" * 50)

    # 生成更复杂的船舶运动场景
    np.random.seed(123)

    # 场景1: 恶劣海况（大幅值，多频率成分）
    print("场景1: 恶劣海况（大幅值运动）")
    severe_motion, _, _ = generate_ship_motion_data(n_samples=1, seq_length=200)

    # 场景2: 平静海况（小幅值，主要是低频运动）
    print("场景2: 平静海况（小幅值，低频主导）")

    t = np.linspace(0, 200, 200)
    calm_motion = (
            0.5 * np.sin(2 * np.pi * 0.02 * t) +  # 低频
            0.2 * np.sin(2 * np.pi * 0.15 * t) +  # 中频
            0.1 * np.sin(2 * np.pi * 0.8 * t) +  # 高频
            0.05 * np.random.randn(200)  # 噪声
    ).reshape(1, -1)

    # 应用模型进行分析
    for scenario_idx, (scenario_name, motion) in enumerate([("恶劣海况", severe_motion), ("平静海况", calm_motion)]):
        print(f"\n分析{scenario_name}:")

        # 预处理
        motion_tensor = torch.FloatTensor(motion).unsqueeze(0).unsqueeze(0).to(device)

        # 使用模型进行分解
        with torch.no_grad():
            decomposed, reconstructed, attention = trained_model(motion_tensor)

        decomposed_np = decomposed.cpu().numpy()[0]

        # 计算各模态能量
        energies = [np.sum(np.abs(decomposed_np[i, :]) ** 2) for i in range(4)]
        total_energy = np.sum(energies)
        energy_ratios = [e / total_energy for e in energies]

        component_names = ['高频模态', '中频模态', '低频模态', '噪声']

        print(f"各模态能量占比:")
        for i, (name, ratio) in enumerate(zip(component_names, energy_ratios)):
            print(f"  {name}: {ratio * 100:.1f}%")

        # 运动烈度评估
        motion_intensity = np.sqrt(np.mean(motion ** 2))
        print(f"运动烈度(RMS): {motion_intensity:.4f}")

        # 基于分解结果的航行安全评估
        high_freq_energy = energy_ratios[0]  # 高频能量占比（波浪冲击）
        low_freq_energy = energy_ratios[2]  # 低频能量占比（慢漂运动）

        if high_freq_energy > 0.4:
            print("警告: 高频波浪冲击能量较高，可能导致船舶局部结构应力增大")
        if low_freq_energy > 0.5:
            print("警告: 低频慢漂运动显著，可能导致船舶定位困难")

        # 可视化该场景
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 原始信号
        axes[0, 0].plot(motion[0], 'b-')
        axes[0, 0].set_title(f'{scenario_name} - 原始信号')
        axes[0, 0].set_xlabel('时间点')
        axes[0, 0].set_ylabel('幅值')
        axes[0, 0].grid(True, alpha=0.3)

        # 模态分解
        for i in range(4):
            axes[(i + 1) // 3, (i + 1) % 3].plot(decomposed_np[i, :], color=['r', 'g', 'b', 'm'][i])
            axes[(i + 1) // 3, (i + 1) % 3].set_title(f'{component_names[i]} (能量占比: {energy_ratios[i] * 100:.1f}%)')
            axes[(i + 1) // 3, (i + 1) % 3].set_xlabel('时间点')
            axes[(i + 1) // 3, (i + 1) % 3].set_ylabel('幅值')
            axes[(i + 1) // 3, (i + 1) % 3].grid(True, alpha=0.3)

        plt.suptitle(f'{scenario_name} - MSA-LSTM模态分解结果', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'scenario_{scenario_idx + 1}.png', dpi=300, bbox_inches='tight')
        plt.show()


# 运行实际场景模拟
apply_to_realistic_scenario()

print("\n" + "=" * 50)
print("MSA-LSTM船舶运动模态分解模拟完成!")
print("=" * 50)
print("\n输出文件总结:")
print("1. modal_decomposition_results.png - 模态分解结果")
print("2. attention_analysis.png - 注意力机制分析")
print("3. frequency_analysis.png - 频率分析")
print("4. training_loss.png - 训练损失曲线")
print("5. scenario_1.png - 恶劣海况分析")
print("6. scenario_2.png - 平静海况分析")
print("\n模型已成功演示了船舶运动信号的模态分解能力!")