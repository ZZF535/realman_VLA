from datasets import load_dataset

try:
    dataset = load_dataset("samlcg/record-test")
    print(f"数据集版本: {dataset.version}")
    print(f"数据集特征: {dataset.features}")
except Exception as e:
    print(f"加载数据集时出错: {e}")