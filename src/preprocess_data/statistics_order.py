
"""
按 refreshed(JSONL) 的原始行序，打印“哪一段来自哪个切片”的连续分组结果
"""

import os
import glob
import json
from collections import defaultdict

# 配置
REFRESHED_FILE = "/Users/jasonh/Desktop/02807/FinalProject/DataPreprocess/arxiv-cs-data-with-citations_merged_20.json"
# 切片文件通配
SLICE_GLOB = "/Users/jasonh/Desktop/02807/FinalProject/DataPreprocess/20/arxiv-cs-data-with-citations_slice*"
# =====================================================

# 可选进度条（没有 tqdm 也能跑）
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x


def iter_jsonl(filepath):
    """逐行读取 JSONL；对混入前缀/后缀的行做“救援解析”"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 快速路径
            if line.startswith("{") and line.endswith("}"):
                try:
                    yield json.loads(line)
                    continue
                except Exception:
                    pass
            # 救援路径：从第一个 '{' 到最后一个 '}' 截取
            try:
                s = line.find("{")
                e = line.rfind("}")
                if s != -1 and e != -1 and e > s:
                    frag = line[s:e+1]
                    yield json.loads(frag)
            except Exception:
                # 跳过无法解析的行
                continue


def build_id_to_slice(slice_paths):
    """
    扫描所有 slice 文件，构建 id -> slice_basename 的映射。
    若同一 id 出现在多个切片，选择**字典序最小**的一个；并记录歧义。
    """
    id_locations = defaultdict(set)
    for spath in tqdm(slice_paths, desc="索引切片"):
        sname = os.path.basename(spath)
        for obj in iter_jsonl(spath):
            pid = obj.get("id")
            if pid is None:
                continue
            id_locations[str(pid)].add(sname)

    id2slice = {}
    ambiguous = 0
    for pid, files in id_locations.items():
        if not files:
            continue
        chosen = sorted(files)[0]
        id2slice[pid] = chosen
        if len(files) > 1:
            ambiguous += 1
    return id2slice, ambiguous


def rle_with_ranges(seq):
    """
    对切片名序列做 run-length 编码，并给出 refreshed 中的起止行号。
    返回列表：[(name, count, start_idx, end_idx), ...]
    """
    if not seq:
        return []
    runs = []
    cur = seq[0]
    cnt = 1
    start = 0
    for i in range(1, len(seq)):
        if seq[i] == cur:
            cnt += 1
        else:
            runs.append((cur, cnt, start, i - 1))
            cur = seq[i]
            cnt = 1
            start = i
    runs.append((cur, cnt, start, len(seq) - 1))
    return runs


def main():
    # 1) 找切片文件
    slice_paths = sorted(glob.glob(SLICE_GLOB))
    if not slice_paths:
        raise FileNotFoundError(f"未匹配到任何切片文件：{SLICE_GLOB}")
    print(f"[INFO] 发现切片文件：{len(slice_paths)} 个")

    # 2) 建索引：id -> slice
    id2slice, ambiguous_count = build_id_to_slice(slice_paths)
    print(f"[INFO] 已索引 ID 数：{len(id2slice)}（多切片歧义 ID：{ambiguous_count}）")

    # 3) 按 refreshed 行序映射切片名
    slice_sequence = []
    missing_ids = 0
    total_rows = 0

    for obj in tqdm(iter_jsonl(REFRESHED_FILE), desc="扫描 refreshed"):
        total_rows += 1
        pid = obj.get("id")
        if pid is None:
            # 没有 id 的行直接跳过，不算 missing
            continue
        sname = id2slice.get(str(pid))
        if sname is None:
            missing_ids += 1
            # 也可把 None 记入序列，但你数据保证顺序，这里直接跳过即可
            continue
        slice_sequence.append(sname)

    # 4) 连续分组 + 终端打印
    runs = rle_with_ranges(slice_sequence)
    print("\n==== refreshed 中的切片连续分组====")
    if not runs:
        print("(无数据)")
    else:
        width = max(len(name) for name, *_ in runs)
        for i, (name, cnt, a, b) in enumerate(runs, 1):
            # a、b 是“映射成功”的行序号（仅包含找到切片的那些行）
            print(f"#{i:04d}  {name:<{width}}  × {cnt:>6} 行   （refreshed 映射行号 {a}–{b}）")

    # 5) 简要统计
    print("\n==== 统计 ====")
    print(f"refreshed 总行数（含无法解析/无 id 行）：{total_rows}")
    print(f"被成功映射到某个切片的行数：{len(slice_sequence)}")
    print(f"在任何切片都找不到的 id 条数：{missing_ids}")
    if ambiguous_count > 0:
        print(f"注意：有 {ambiguous_count} 个 id 同时出现在多个切片里（本脚本取字典序最小的切片作为来源）。")


if __name__ == "__main__":
    main()
