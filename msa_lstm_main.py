import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
def generate_ship_motion_data(n_samples=800, seq_length=80):
    """生成模拟船舶运动数据"""
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


# 生成数据
print("生成模拟船舶运动数据...")
ship_motion, components, time_points = generate_ship_motion_data()
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

# 创建数据集和数据加载器
train_dataset = ShipMotionDataset(train_signals, train_components)
val_dataset = ShipMotionDataset(val_signals, val_components)
test_dataset = ShipMotionDataset(test_signals, test_components)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")


# ====================== 3. MSA-LSTM模型定义 ======================
class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

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


class SimpleMSALSTM(nn.Module):
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

        # LSTM编码器
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

        # 模态解码器
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


# ====================== 4. 训练函数 ======================
def train_model(model, train_loader, val_loader, num_epochs=15):
    """训练模型"""
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            loss_reconstruction = criterion(reconstructed, signals)
            loss_components = criterion(decomposed, components)
            total_loss = loss_reconstruction + 0.5 * loss_components

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

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

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    return model, train_losses, val_losses


# ====================== 5. 模型训练 ======================
# 创建模型
model = SimpleMSALSTM(
    input_dim=1,
    hidden_dim=64,
    num_heads=2,
    num_components=4
)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练模型
trained_model, train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15
)

# ====================== 6. 测试和评估 ======================
print("\n测试模型...")
trained_model.eval()

# 获取一个测试样本
test_signals_np = None
test_components_np = None
for signals, components in test_loader:
    test_signals_np = signals.numpy()
    test_components_np = components.numpy()
    break

if test_signals_np is not None:
    # 测试模型
    test_signals_tensor = torch.FloatTensor(test_signals_np).to(device)
    with torch.no_grad():
        decomposed, reconstructed, _ = trained_model(test_signals_tensor)

    decomposed_np = decomposed.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()

    # 计算重构误差
    reconstruction_error = np.mean((reconstructed_np - test_signals_np) ** 2)
    print(f"重构误差: {reconstruction_error:.6f}")

    # 计算模态分解误差
    component_error = np.mean((decomposed_np - test_components_np) ** 2)
    print(f"模态分解误差: {component_error:.6f}")

    # 保存结果用于后续分析
    np.savez('msa_lstm_results.npz',
             train_losses=train_losses,
             val_losses=val_losses,
             test_signals=test_signals_np,
             test_components=test_components_np,
             decomposed=decomposed_np,
             reconstructed=reconstructed_np,
             time_points=time_points)
    print("结果已保存到 msa_lstm_results.npz")
else:
    print("没有测试数据")

# ====================== 7. 保存模型 ======================
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses
}, 'msa_lstm_ship_motion_model.pth')
print("\n模型已保存到: msa_lstm_ship_motion_model.pth")

# ====================== 8. 创建简单的文本报告 ======================
print("\n" + "=" * 50)
print("MSA-LSTM船舶运动模态分解训练完成!")
print("=" * 50)
print("\n训练结果总结:")
print(f"- 最终训练损失: {train_losses[-1]:.6f}")
print(f"- 最终验证损失: {val_losses[-1]:.6f}")
if test_signals_np is not None:
    print(f"- 测试重构误差: {reconstruction_error:.6f}")
    print(f"- 测试模态分解误差: {component_error:.6f}")
print("\n生成的文件:")
print("1. msa_lstm_ship_motion_model.pth - 训练好的模型")
print("2. msa_lstm_results.npz - 训练和测试结果数据")
print("\n要可视化结果，请运行 visualization.py")