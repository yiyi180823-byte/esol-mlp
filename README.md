# ESOL 分子性质预测

本项目使用 1024 位 ECFP 指纹和 PyTorch 全连接网络预测 Delaney ESOL 数据集中的实验 logS。数据按分子骨架划分为训练集、验证集和测试集；模型选择只使用验证集，测试集仅在加载验证集最佳权重后评估一次。

## 固定配置

- 数据集：Delaney ESOL，共 1128 个分子
- 特征：1024 位 ECFP，半径 2，不做特征标准化
- 划分：DeepChem scaffold split
- 模型：1024 → 256 → 64 → 1，ReLU，第一层后 Dropout 0.2
- 训练：MSELoss、Adam、学习率 0.001、Batch size 32
- 早停：最多 100 Epoch，Patience 10，依据验证集 RMSE
- 随机种子：42
- 运行设备：CPU

## 环境

已验证环境使用 Python 3.10.20。建议在项目目录创建独立环境，并始终通过 `python -m pip` 安装依赖。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## 训练与评价

```powershell
.venv\Scripts\python train.py
```

命令会依次完成数据下载与检查、32 条样本过拟合测试、正式训练、验证集早停、测试集最终评价以及结果保存。首次运行需要联网下载 Delaney 数据。

生成文件：

- `outputs/best_model.pt`：验证集最佳权重及模型、特征和训练配置
- `outputs/metrics.json`：测试集、训练集均值基线和环境信息
- `outputs/test_predictions.csv`：测试集 SMILES、真实 logS 和预测 logS
- `figures/loss_curve.png`：训练集与验证集 RMSE 曲线
- `figures/test_predictions.png`：真实值与预测值散点图，含 y=x 参考线

## 单分子推理

先运行训练命令生成模型，再输入一个合法 SMILES：

```powershell
.venv\Scripts\python predict.py "CCO"
```

输出为模型预测的 logS。无效 SMILES 会返回明确错误，不会生成预测。

## 当前结果

固定配置在当前 CPU 环境中的一次完整复现结果如下：

| 模型 | RMSE | MAE | R² |
|---|---:|---:|---:|
| PyTorch MLP | 1.630 | 1.257 | 0.409 |
| 训练集均值基线 | 2.315 | 1.776 | -0.193 |

验证集最佳轮次为第 3 轮，验证集 RMSE 为 1.713；Early stopping 在第 13 轮触发。32 条训练样本过拟合测试的 MSE 从 7.628 降至 0.00224，正式训练前的管线检查通过。

## 复现约束与局限性

- 该结果只针对一次固定 scaffold 划分，不代表跨划分或外部数据上的稳定性能。
- ECFP 丢失三维构象和部分连续分子信息；MLP 也没有显式建模图结构。
- ESOL 数据集较小，R² 等指标可能对少数难样本和划分方式敏感。
- 预测仅用于学习与基准复现，不应直接用于实验或药物开发决策。
