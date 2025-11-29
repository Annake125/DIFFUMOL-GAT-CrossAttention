"""
预计算分子指纹（ECFP/Morgan Fingerprints）for MOSES2 Dataset
用于训练时直接加载，避免重复计算

针对moses2.csv数据集优化，支持：
- 大小写不敏感的列名检测
- 自动识别SPLIT列（train/test/test_scaff）
- 详细的数据质量报告
- 按原始顺序保存指纹（与数据集索引对应）
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
from tqdm import tqdm
import argparse
import os
import sys

# 关闭RDKit警告
RDLogger.DisableLog('rdApp.*')


def compute_ecfp(smiles, radius=2, nBits=2048):
    """
    计算ECFP (Extended-Connectivity Fingerprints) / Morgan指纹

    Args:
        smiles: SMILES字符串
        radius: 指纹半径 (默认2, 对应ECFP4)
        nBits: 指纹向量维度

    Returns:
        numpy array: 二进制指纹向量
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        # 使用Morgan算法生成ECFP
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros(nBits, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

        return arr
    except Exception as e:
        return None


def analyze_moses_dataset(df):
    """分析MOSES数据集的基本信息"""
    print("\n" + "="*60)
    print("📊 MOSES数据集分析")
    print("="*60)

    # 列名标准化（转小写）
    df.columns = df.columns.str.lower()

    print(f"总分子数: {len(df):,}")
    print(f"列名: {', '.join(df.columns.tolist())}")

    # 检查SPLIT列
    if 'split' in df.columns:
        print(f"\n数据集划分 (SPLIT):")
        split_counts = df['split'].value_counts()
        for split_name, count in split_counts.items():
            print(f"  - {split_name:12s}: {count:7,} ({count/len(df)*100:5.2f}%)")
    else:
        print("\n⚠️  警告: 未找到'split'列，无法统计数据集划分")

    # 检查SMILES质量
    print(f"\n✅ SMILES列检测: {'smiles' if 'smiles' in df.columns else '❌ 未找到'}")

    return df


def precompute_moses_fingerprints(csv_path, output_path=None, radius=2, nBits=2048, validate=True):
    """
    为MOSES数据集预计算分子指纹

    Args:
        csv_path: moses2.csv文件路径
        output_path: 输出.npy文件路径（默认自动生成）
        radius: ECFP半径 (2=ECFP4, 3=ECFP6)
        nBits: 指纹维度
        validate: 是否进行数据验证
    """
    print(f"\n🧬 MOSES分子指纹预计算工具")
    print(f"="*60)
    print(f"输入文件: {csv_path}")
    print(f"ECFP参数: radius={radius} (ECFP{radius*2}), nBits={nBits}")

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    # 读取数据
    print(f"\n📂 加载数据...")
    df = pd.read_csv(csv_path)

    # 分析数据集
    df = analyze_moses_dataset(df)

    # 检查SMILES列（大小写不敏感）
    smiles_col = None
    for col in df.columns:
        if col.lower() == 'smiles':
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError(f"❌ 未找到SMILES列！可用列: {df.columns.tolist()}")

    smiles_list = df[smiles_col].tolist()
    num_mols = len(smiles_list)

    # 数据验证
    if validate:
        print(f"\n🔍 验证SMILES质量...")
        null_count = df[smiles_col].isnull().sum()
        if null_count > 0:
            print(f"  ⚠️  发现 {null_count} 个空值SMILES")
            smiles_list = [s if pd.notna(s) else '' for s in smiles_list]

    # 开始计算指纹
    print(f"\n⚙️  开始计算 {num_mols:,} 个分子的指纹...")
    print(f"  - 算法: Morgan/ECFP")
    print(f"  - 半径: {radius} (ECFP{radius*2})")
    print(f"  - 维度: {nBits}")

    fingerprints = np.zeros((num_mols, nBits), dtype=np.float32)
    invalid_indices = []
    invalid_smiles = []

    for idx, smiles in enumerate(tqdm(smiles_list, desc="计算指纹", ncols=80)):
        if pd.isna(smiles) or smiles == '':
            invalid_indices.append(idx)
            invalid_smiles.append("(empty)")
            continue

        fp = compute_ecfp(smiles, radius=radius, nBits=nBits)

        if fp is None:
            invalid_indices.append(idx)
            invalid_smiles.append(smiles[:50])  # 记录前50个字符
        else:
            fingerprints[idx] = fp

    # 统计结果
    valid_count = num_mols - len(invalid_indices)
    print(f"\n✅ 计算完成!")
    print(f"  有效分子: {valid_count:,} / {num_mols:,} ({valid_count/num_mols*100:.2f}%)")
    print(f"  无效分子: {len(invalid_indices):,}")

    if len(invalid_indices) > 0 and len(invalid_indices) <= 10:
        print(f"\n  ⚠️  无效SMILES列表:")
        for i, (idx, smi) in enumerate(zip(invalid_indices, invalid_smiles), 1):
            print(f"    {i}. [行{idx}] {smi}")
    elif len(invalid_indices) > 10:
        print(f"\n  ⚠️  无效SMILES过多 ({len(invalid_indices)}个)，仅显示前10个:")
        for i in range(10):
            idx = invalid_indices[i]
            smi = invalid_smiles[i]
            print(f"    {i+1}. [行{idx}] {smi}")
        print(f"    ...")

    # 指纹统计
    non_zero_fps = np.sum(fingerprints.sum(axis=1) > 0)
    avg_bits_set = np.mean(fingerprints.sum(axis=1)[fingerprints.sum(axis=1) > 0])
    print(f"\n📈 指纹统计:")
    print(f"  非零指纹: {non_zero_fps:,}")
    print(f"  平均激活位数: {avg_bits_set:.1f} / {nBits}")

    # 自动生成输出路径
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_ecfp{radius*2}_{nBits}.npy"

    # 保存指纹
    print(f"\n💾 保存指纹到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, fingerprints)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  文件大小: {file_size_mb:.2f} MB")

    # 验证保存
    print(f"\n🔍 验证保存的文件...")
    loaded_fp = np.load(output_path)
    assert loaded_fp.shape == (num_mols, nBits), "形状不匹配！"
    assert np.allclose(loaded_fp, fingerprints), "数据不匹配！"
    print(f"  ✅ 验证通过！")

    print(f"\n" + "="*60)
    print(f"✅ 指纹预计算完成!")
    print(f"="*60)
    print(f"📁 输出文件: {output_path}")
    print(f"📊 数据形状: {loaded_fp.shape}")
    print(f"💡 使用方法:")
    print(f"   1. 修改 diffumol/config.json:")
    print(f'      "use_fingerprint": true,')
    print(f'      "fingerprint_path": "{output_path}"')
    print(f"   2. 运行训练: python train.py")
    print(f"="*60)

    return fingerprints


def main():
    parser = argparse.ArgumentParser(
        description='为MOSES数据集预计算ECFP分子指纹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（默认ECFP4, 2048位）
  python precompute_fingerprints.py --data_path ./datasets/moses2.csv

  # 使用ECFP6
  python precompute_fingerprints.py --data_path ./datasets/moses2.csv --radius 3

  # 自定义维度（1024位，更快但精度略低）
  python precompute_fingerprints.py --data_path ./datasets/moses2.csv --nBits 1024

  # 指定输出路径
  python precompute_fingerprints.py --data_path ./datasets/moses2.csv \\
      --output_path ./my_fingerprints.npy
        """
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='moses2.csv文件路径'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='输出.npy文件路径（默认: <data_path>_ecfpX_XXXX.npy）'
    )

    parser.add_argument(
        '--radius',
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help='ECFP半径 (1=ECFP2, 2=ECFP4, 3=ECFP6, 默认=2)'
    )

    parser.add_argument(
        '--nBits',
        type=int,
        default=2048,
        choices=[512, 1024, 2048, 4096],
        help='指纹维度 (默认=2048)'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='跳过数据验证（加快速度）'
    )

    args = parser.parse_args()

    # 检查依赖
    try:
        from rdkit import Chem
    except ImportError:
        print("❌ 错误: 未安装RDKit!")
        print("请安装: conda install -c conda-forge rdkit")
        sys.exit(1)

    # 执行预计算
    try:
        precompute_moses_fingerprints(
            csv_path=args.data_path,
            output_path=args.output_path,
            radius=args.radius,
            nBits=args.nBits,
            validate=not args.no_validate
        )
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
