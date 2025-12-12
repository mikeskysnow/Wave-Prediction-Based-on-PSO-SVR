import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ====================== 1. 简化的模拟船舶运动数据生成 ======================
def generate_ship_motion_data(n_samples=1000, seq_length=100):
    """生成简化的船舶运动数据"""
    t = np.linspace(0, 100, seq_length)

    # 1. 高频模态
    high_freq_component = np.zeros((n_samples, seq_length))
    for i in range(n_samples):
        freq = 0.8 + 0.4 * np.random.randn()
        amp = 1.0 + 0.3 * np.random.randn()
        phase = 2 * np.pi * np.random.rand()
        high_freq_component[i] = amp * np.sin(2 * np.pi * freq * t + phase)

    # 2. 中频模态
    mid_freq_component = np.zeros((n_samples, seq_length))
    for i in range(n_samples):
        freq = 0.2 + 0.1 * np.random.randn()
        amp = 1.5 + 0.5 * np.random.randn()
        phase = 2 * np.pi * np.random.rand()
        mid_freq_component[i] = amp * np.sin(2 * np.pi * freq * t + phase)

    # 3. 低频模态
    low_freq_component = np.zeros((n_samples, seq_length))
    for i in range(n_samples):
        freq = 0.03 + 0.02 * np.random.randn()
        amp = 0.8 + 0.2 * np.random.randn()
        phase = 2 * np.pi * np.random.rand()
        low_freq_component[i] = amp * np.sin(2 * np.pi * freq * t + phase)

    # 4. 噪声
    noise = 0.1 * np.random.randn(n_samples, seq_length)

    # 合成船舶运动信号
    ship_motion = high_freq_component + mid_freq_component + low_freq_component + noise

    # 模态分量作为标签
    components = np.stack([high_freq_component, mid_freq_component, low_freq_component, noise], axis=1)

    return ship_motion, components, t


# 生成更少的数据以减少内存使用
print("生成模拟船舶运动数据...")
ship_motion, components, time_points = generate_ship_motion_data(n_samples=800, seq_length=80)
print(f"数据形状: ship_motion={ship_motion.shape}, components={components.shape}")


# ====================== 2. 数据预处理 ======================
class ShipMotionDataset(Dataset):
    def __init__(self, signals, components):
        self.signals = signals
        self.components = components

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.signals[idx]).unsqueeze(0)
        component = torch.FloatTensor(self.components[idx])
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

# 创建数据集和数据加载器，减小批次大小
train_dataset = ShipMotionDataset(train_signals, train_components)
val_dataset = ShipMotionDataset(val_signals, val_components)
test_dataset = ShipMotionDataset(test_signals, test_components)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")


# ====================== 3. 简化的MSA-LSTM模型定义 ======================
class MultiHeadAttention(nn.Module):
    """简化的多头自注意力机制"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # 线性变换
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(scores, dim=-1)

        # 应用注意力
        out = torch.matmul(attention, V)

        # 重塑并输出
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)

        return out, attention


class Simple_MSA_LSTM(nn.Module):
    """简化的MSA-LSTM模型"""

    def __init__(self, input_dim=1, hidden_dim=64, num_heads=2, num_components=4):
        super().__init__()
        self.num_components = num_components

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM编码器（单层单向，减少复杂度）
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 多头自注意力
        self.multihead_attention = MultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads
        )

        # 模态解码器（共享参数，减少模型大小）
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 输出层
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_components)
        ])

    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        batch_size, _, seq_len = x.shape

        # 编码
        encoded = self.encoder(x)  # [batch_size, hidden_dim, seq_len]
        encoded = encoded.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]

        # LSTM编码
        lstm_out, _ = self.lstm_encoder(encoded)

        # 多头自注意力
        attention_out, attention_weights = self.multihead_attention(lstm_out)

        # 残差连接
        combined = lstm_out + attention_out

        # 模态分解
        decoded, _ = self.decoder(combined)

        components = []
        for i in range(self.num_components):
            component = self.output_layers[i](decoded)
            components.append(component)

        # 拼接所有模态
        decomposed = torch.cat(components, dim=-1)  # [batch_size, seq_len, num_components]
        decomposed = decomposed.permute(0, 2, 1)  # [batch_size, num_components, seq_len]

        # 重构信号
        reconstructed = torch.sum(decomposed, dim=1, keepdim=True)

        return decomposed, reconstructed, attention_weights


# ====================== 4. 简化的训练函数 ======================
def simple_train_model(model, train_loader, val_loader, num_epochs=20):
    """简化的训练函数"""
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    print("开始训练...")
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_idx, (signals, components) in enumerate(train_loader):
                signals, components = signals.to(device), components.to(device)

                # 前向传播
                decomposed, reconstructed, _ = model(signals)

                # 计算损失
                loss_reconstruction = criterion(reconstructed, signals)
                loss_components = criterion(decomposed, components)
                total_loss = loss_reconstruction + 0.5 * loss_components

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += total_loss.item()

                # 清理内存
                del decomposed, reconstructed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for signals, components in val_loader:
                    signals, components = signals.to(device), components.to(device)
                    decomposed, reconstructed, _ = model(signals)

                    loss_reconstruction = criterion(reconstructed, signals)
                    loss_components = criterion(decomposed, components)
                    total_val_loss = loss_reconstruction + 0.5 * loss_components
                    val_loss += total_val_loss.item()

                    # 清理内存
                    del decomposed, reconstructed

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("尝试继续执行...")

    return model, train_losses, val_losses


# ====================== 5. 模型训练 ======================
# 创建简化的模型
model = Simple_MSA_LSTM(
    input_dim=1,
    hidden_dim=64,
    num_heads=2,
    num_components=4
)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练模型
trained_model, train_losses, val_losses = simple_train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15
)


# ====================== 6. 基本可视化 ======================
def simple_visualization(model, test_loader, time_points):
    """简化的可视化函数"""
    try:
        model.eval()

        # 获取一个测试样本
        sample_signals, sample_components = None, None
        for signals, components in test_loader:
            sample_signals = signals[0:1]
            sample_components = components[0:1]
            break

        if sample_signals is None:
            print("没有测试数据")
            return

        # 前向传播
        with torch.no_grad():
            sample_signals = sample_signals.to(device)
            decomposed, reconstructed, attention = model(sample_signals)

        # 转换为numpy
        signals_np = sample_signals.cpu().numpy()[0, 0]
        decomposed_np = decomposed.cpu().numpy()[0]
        reconstructed_np = reconstructed.cpu().numpy()[0, 0]

        # 绘制训练损失
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='训练损失', alpha=0.7)
        plt.plot(val_losses, 'r-', label='验证损失', alpha=0.7)
        plt.xlabel('训练轮次')
        plt.ylabel('损失')
        plt.title('训练过程')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制信号重构
        plt.subplot(1, 2, 2)
        plt.plot(time_points, signals_np, 'b-', label='原始信号', alpha=0.7)
        plt.plot(time_points, reconstructed_np, 'r--', label='重构信号', alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('幅值')
        plt.title('信号重构')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_and_reconstruction.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 绘制模态分解
        plt.figure(figsize=(12, 8))
        component_names = ['高频模态', '中频模态', '低频模态', '噪声']
        colors = ['red', 'green', 'blue', 'purple']

        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(time_points, decomposed_np[i], color=colors[i], linewidth=1.5)
            plt.title(f'{component_names[i]}')
            plt.xlabel('时间')
            plt.ylabel('幅值')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('modal_decomposition.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 清理内存
        del decomposed, reconstructed, attention
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return signals_np, decomposed_np

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        return None, None


print("\n进行可视化...")
signals_np, decomposed_np = simple_visualization(trained_model, test_loader, time_points)


# ====================== 7. 简单的性能评估 ======================
def simple_evaluation(model, test_loader):
    """简单的性能评估"""
    try:
        model.eval()

        reconstruction_errors = []
        with torch.no_grad():
            for signals, _ in test_loader:
                signals = signals.to(device)
                decomposed, reconstructed, _ = model(signals)

                # 计算重构误差
                error = torch.mean((reconstructed - signals) ** 2).item()
                reconstruction_errors.append(error)

                # 清理内存
                del decomposed, reconstructed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if reconstruction_errors:
            avg_error = np.mean(reconstruction_errors)
            print(f"平均重构误差: {avg_error:.6f}")
            return avg_error
        else:
            print("无法计算误差")
            return None

    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        return None


print("\n评估模型性能...")
avg_error = simple_evaluation(trained_model, test_loader)


# ====================== 8. 实际应用示例 ======================
def simple_scenario_demo():
    """简化的场景演示"""
    try:
        print("\n" + "=" * 50)
        print("简化的船舶运动场景演示")
        print("=" * 50)

        # 生成测试场景
        t = np.linspace(0, 100, 80)

        # 场景1: 复杂海况
        print("场景1: 复杂海况")
        complex_motion = (
                1.2 * np.sin(2 * np.pi * 0.8 * t) +  # 高频
                1.5 * np.sin(2 * np.pi * 0.2 * t) +  # 中频
                0.8 * np.sin(2 * np.pi * 0.03 * t) +  # 低频
                0.15 * np.random.randn(80)  # 噪声
        )

        # 场景2: 平静海况
        print("场景2: 平静海况")
        calm_motion = (
                0.3 * np.sin(2 * np.pi * 0.6 * t) +  # 高频
                0.5 * np.sin(2 * np.pi * 0.15 * t) +  # 中频
                0.6 * np.sin(2 * np.pi * 0.02 * t) +  # 低频
                0.05 * np.random.randn(80)  # 噪声
        )

        scenarios = [("复杂海况", complex_motion), ("平静海况", calm_motion)]

        for scenario_name, motion in scenarios:
            print(f"\n分析{scenario_name}:")

            # 预处理
            motion_tensor = torch.FloatTensor(motion).unsqueeze(0).unsqueeze(0).to(device)

            # 使用模型进行分解
            with torch.no_grad():
                decomposed, reconstructed, _ = trained_model(motion_tensor)

            decomposed_np = decomposed.cpu().numpy()[0]

            # 计算各模态能量
            energies = [np.sum(np.abs(decomposed_np[i, :]) ** 2) for i in range(4)]
            total_energy = np.sum(energies)
            energy_ratios = [e / total_energy for e in energies]

            component_names = ['高频模态', '中频模态', '低频模态', '噪声']

            print(f"各模态能量占比:")
            for i, (name, ratio) in enumerate(zip(component_names, energy_ratios)):
                print(f"  {name}: {ratio * 100:.1f}%")

            # 简单的场景判断
            high_freq_ratio = energy_ratios[0]
            if high_freq_ratio > 0.4:
                print("  注意：高频波浪成分显著")
            elif high_freq_ratio < 0.1:
                print("  注意：高频波浪成分较弱")

            # 可视化
            plt.figure(figsize=(10, 6))

            plt.subplot(2, 1, 1)
            plt.plot(t, motion, 'b-', linewidth=2)
            plt.title(f'{scenario_name} - 船舶运动信号')
            plt.xlabel('时间')
            plt.ylabel('幅值')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            colors = ['red', 'green', 'blue', 'purple']
            for i in range(4):
                plt.plot(t, decomposed_np[i], color=colors[i], label=component_names[i], alpha=0.7)

            plt.title(f'模态分解结果')
            plt.xlabel('时间')
            plt.ylabel('幅值')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{scenario_name}_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()

            # 清理内存
            del decomposed, reconstructed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"场景演示过程中出现错误: {e}")


# 运行场景演示
simple_scenario_demo()

# ====================== 9. 保存模型 ======================
try:
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'msa_lstm_ship_motion_model.pth')
    print("\n模型已保存到: msa_lstm_ship_motion_model.pth")
except Exception as e:
    print(f"保存模型时出现错误: {e}")

print("\n" + "=" * 50)
print("MSA-LSTM船舶运动模态分解演示完成!")
print("=" * 50)
print("\n生成的文件:")
print("1. training_and_reconstruction.png - 训练过程和重构效果")
print("2. modal_decomposition.png - 模态分解结果")
print("3. 复杂海况_analysis.png - 复杂海况分析")
print("4. 平静海况_analysis.png - 平静海况分析")
print("5. msa_lstm_ship_motion_model.pth - 训练好的模型")
print("\n演示成功完成!")