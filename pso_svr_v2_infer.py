# 加载模型
metadata_path = 'saved_models/metadata_20231201_143022.pkl'
model, scaler_X, scaler_y, metadata = load_model(metadata_path)

# 准备新数据（需要与训练数据相同的特征）
new_data = [...]  # 新数据

# 归一化
new_data_scaled = scaler_X.transform(new_data)

# 预测
prediction_scaled = model.predict(new_data_scaled)

# 反归一化
prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))