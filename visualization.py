# visualization.py
import numpy as np
import matplotlib

# 使用非交互式后端，避免Windows兼容性问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_results():
    """可视化训练结果"""
    try:
        # 加载保存的结果
        data = np.load('msa_lstm_results.npz', allow_pickle=True)

        train_losses = data['train_losses']
        val_losses = data['val_losses']
        test_signals = data['test_signals']
        test_components = data['test_components']
        decomposed = data['decomposed']
        reconstructed = data['reconstructed']
        time_points = data['time_points']

        print("成功加载结果数据")

        # 1. 绘制训练损失曲线
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='训练损失', alpha=0.7, linewidth=2)
        plt.plot(val_losses, 'r-', label='验证损失', alpha=0.7, linewidth=2)
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('损失', fontsize=12)
        plt.title('MSA-LSTM训练过程', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 2. 绘制信号重构对比
        sample_idx = 0
        plt.subplot(1, 2, 2)
        plt.plot(time_points, test_signals[sample_idx, 0, :],
                 'b-', label='原始信号', alpha=0.8, linewidth=2)
        plt.plot(time_points, reconstructed[sample_idx, 0, :],
                 'r--', label='重构信号', alpha=0.8, linewidth=2)
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('幅值', fontsize=12)
        plt.title('信号重构对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("已保存训练结果图: training_results.png")

        # 3. 绘制模态分解结果
        plt.figure(figsize=(12, 8))
        component_names = ['高频模态', '中频模态', '低频模态', '噪声']
        colors = ['red', 'green', 'blue', 'purple']

        for i in range(4):
            plt.subplot(4, 1, i + 1)
            # 真实分量
            plt.plot(time_points, test_components[sample_idx, i, :],
                     'k-', label='真实', alpha=0.6, linewidth=1.5)
            # 预测分量
            plt.plot(time_points, decomposed[sample_idx, i, :],
                     color=colors[i], label='预测', alpha=0.8, linewidth=1.5)
            plt.title(f'{component_names[i]}', fontsize=12, fontweight='bold')
            plt.xlabel('时间', fontsize=10)
            plt.ylabel('幅值', fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

        plt.suptitle('模态分解结果对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('modal_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("已保存模态分解图: modal_decomposition.png")

        # 4. 绘制误差分布
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        reconstruction_errors = np.abs(reconstructed - test_signals).flatten()
        plt.hist(reconstruction_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('重构误差', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.title('重构误差分布', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        component_errors = np.abs(decomposed - test_components).flatten()
        plt.hist(component_errors, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('模态分解误差', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.title('模态分解误差分布', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("已保存误差分布图: error_distribution.png")

        # 5. 绘制能量分布
        plt.figure(figsize=(8, 6))

        # 计算样本能量
        energies = []
        for i in range(4):
            energy = np.sum(np.abs(decomposed[sample_idx, i, :]) ** 2)
            energies.append(energy)

        total_energy = sum(energies)
        energy_ratios = [e / total_energy * 100 for e in energies]

        # 饼图
        plt.subplot(2, 1, 1)
        wedges, texts, autotexts = plt.pie(energy_ratios,
                                           labels=component_names,
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           startangle=90,
                                           explode=(0.05, 0.05, 0.05, 0.05))

        # 美化饼图文本
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            autotext.set_weight('bold')

        plt.title('模态能量占比', fontsize=14, fontweight='bold')

        # 条形图
        plt.subplot(2, 1, 2)
        bars = plt.bar(component_names, energies, color=colors, alpha=0.8)
        plt.xlabel('模态分量', fontsize=12)
        plt.ylabel('能量', fontsize=12)
        plt.title('模态能量对比', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, energy in zip(bars, energies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{energy:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('energy_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("已保存能量分布图: energy_distribution.png")

        print("\n" + "=" * 50)
        print("可视化完成!")
        print("=" * 50)
        print("\n生成的可视化文件:")
        print("1. training_results.png - 训练过程和重构对比")
        print("2. modal_decomposition.png - 模态分解结果")
        print("3. error_distribution.png - 误差分布")
        print("4. energy_distribution.png - 能量分布")

    except Exception as e:
        print(f"可视化过程中出现错误: {str(e)}")
        print("请确保已运行主程序并生成了结果文件")


if __name__ == "__main__":
    visualize_results()