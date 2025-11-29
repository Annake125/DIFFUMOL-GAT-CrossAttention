"""
预计算分子指纹（ECFP/Morgan Fingerprints）
用于训练时直接加载，避免重复计算
"""

import numpy as np
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import argparse
import os


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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Invalid SMILES: {smiles}")
        return np.zeros(nBits, dtype=np.float32)

    # 使用Morgan算法生成ECFP
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros(nBits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def precompute_fingerprints_from_csv(csv_path, output_path, radius=2, nBits=2048):
    """
    从CSV文件预计算所有分子的指纹

    Args:
        csv_path: 输入CSV文件路径（包含'smiles'列）
        output_path: 输出.npy文件路径
        radius: ECFP半径
        nBits: 指纹维度
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'smiles' not in df.columns:
        raise ValueError("CSV file must contain 'smiles' column")

    smiles_list = df['smiles'].tolist()
    num_mols = len(smiles_list)

    print(f"Computing ECFP fingerprints for {num_mols} molecules...")
    print(f"Parameters: radius={radius}, nBits={nBits}")

    fingerprints = np.zeros((num_mols, nBits), dtype=np.float32)

    invalid_count = 0
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Computing fingerprints")):
        fp = compute_ecfp(smiles, radius=radius, nBits=nBits)
        fingerprints[idx] = fp
        if np.sum(fp) == 0:
            invalid_count += 1

    print(f"\nFingerprints computed successfully!")
    print(f"Valid molecules: {num_mols - invalid_count}/{num_mols}")
    print(f"Invalid/empty fingerprints: {invalid_count}")

    # 保存为.npy文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, fingerprints)
    print(f"Saved fingerprints to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return fingerprints


def precompute_fingerprints_from_jsonl(jsonl_path, output_path, radius=2, nBits=2048):
    """
    从JSONL文件预计算所有分子的指纹

    Args:
        jsonl_path: 输入JSONL文件路径（每行包含'smiles'或'trg'字段）
        output_path: 输出.npy文件路径
        radius: ECFP半径
        nBits: 指纹维度
    """
    print(f"Loading data from {jsonl_path}...")
    smiles_list = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 尝试获取SMILES（可能在'smiles'或'trg'字段）
            smiles = data.get('smiles') or data.get('trg')
            if smiles:
                smiles_list.append(smiles.strip())

    num_mols = len(smiles_list)
    print(f"Computing ECFP fingerprints for {num_mols} molecules...")
    print(f"Parameters: radius={radius}, nBits={nBits}")

    fingerprints = np.zeros((num_mols, nBits), dtype=np.float32)

    invalid_count = 0
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Computing fingerprints")):
        fp = compute_ecfp(smiles, radius=radius, nBits=nBits)
        fingerprints[idx] = fp
        if np.sum(fp) == 0:
            invalid_count += 1

    print(f"\nFingerprints computed successfully!")
    print(f"Valid molecules: {num_mols - invalid_count}/{num_mols}")
    print(f"Invalid/empty fingerprints: {invalid_count}")

    # 保存为.npy文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, fingerprints)
    print(f"Saved fingerprints to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return fingerprints


def main():
    parser = argparse.ArgumentParser(description='Precompute molecular fingerprints (ECFP)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input data file (CSV or JSONL)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to output .npy file (default: same as input with .npy extension)')
    parser.add_argument('--radius', type=int, default=2,
                        help='ECFP radius (default: 2 for ECFP4)')
    parser.add_argument('--nBits', type=int, default=2048,
                        help='Fingerprint dimension (default: 2048)')

    args = parser.parse_args()

    # 自动确定输出路径
    if args.output_path is None:
        base_name = os.path.splitext(args.data_path)[0]
        args.output_path = f"{base_name}_ecfp{args.radius*2}_{args.nBits}.npy"

    # 根据文件类型选择处理函数
    if args.data_path.endswith('.csv'):
        precompute_fingerprints_from_csv(
            args.data_path, args.output_path,
            radius=args.radius, nBits=args.nBits
        )
    elif args.data_path.endswith('.jsonl'):
        precompute_fingerprints_from_jsonl(
            args.data_path, args.output_path,
            radius=args.radius, nBits=args.nBits
        )
    else:
        raise ValueError("Unsupported file format. Use .csv or .jsonl")

    print("\n✅ Fingerprint precomputation completed!")
    print(f"📁 Output file: {args.output_path}")
    print(f"💡 Use this file in training with --fingerprint_path {args.output_path}")


if __name__ == '__main__':
    main()
