import torch

torch.manual_seed(456)
# 定义输入矩阵的行和列
row_count, col_count = 4, 16
# 定义输入矩阵
long_input_vec: torch.Tensor = torch.rand((row_count, col_count))
# 定义在线softmax矩阵
online_softmax = torch.zeros_like(long_input_vec)

for row in range(row_count):
    # row_max是上述公式中的m，也就是一行中的最大值
    row_max = 0.0
    # normalizer_term是上述公式中的d，也就是一行中的归一化项
    normalizer_term = 0.0
    for col in range(col_count): 
        # 获取当前行、列的值
        val = long_input_vec[row, col]
        # 更新当前行的最大值
        old_row_max = row_max
        row_max = max(old_row_max, val)
        # 更新当前行的归一化项
        normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(val - row_max)
    # 计算当前行的softmax，其中long_input_vec[row, :] - row_max 是上述公式中的x_i - m_{new}，normalizer_term是上述公式中的d_{new}
    online_softmax[row, :] = torch.exp(long_input_vec[row, :] - row_max) / normalizer_term

expected_softmax = torch.softmax(long_input_vec, dim=1)
assert torch.allclose(online_softmax, expected_softmax, atol=1e-5)