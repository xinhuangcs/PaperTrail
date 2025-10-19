
"""
统计 refreshed 文件中 citation_count == -1 的 ID，
它们分别出现在 arxiv-cs-data-with-citations_slice* 各文件中的数量（按去重 ID 计）。

输出：
  - 控制台打印汇总表
  - 生成 CSV: -1_id_counts_per_slice.csv
  - 生成 JSON: -1_id_counts_per_slice.json
"""

import os
import json
import glob
from datetime import datetime
from collections import defaultdict, Counter

# 可选：进度条（若未安装 tqdm 也能正常运行）
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# 配置
REFRESHED_FILE = "/Users/jasonh/Desktop/02807/PaperTrail/data/preprocess/arxiv-cs-data-with-citations-refreshed.json"
# 切片文件通配
SLICE_GLOB = "/Users/jasonh/Desktop/02807/PaperTrail/data/preprocess/20/arxiv-cs-data-with-citations_slice*"


# 输出文件（默认写到当前工作目录）
timestamp = datetime.now().strftime("%Y%m%d_%H_%M")  # 例如 20250215_23_20
OUT_CSV = f"./-1_id_counts_per_slice_{timestamp}.csv"



def iter_jsonl(filepath):
    """
    逐行读取 JSONL。对行首/行尾可能混入的噪声做一次“救援”：
    截取第一个 '{' 到最后一个 '}' 的子串再做 json.loads。
    """
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
            # “救援”路径：尽量从第一个 '{' 到最后一个 '}' 解析
            try:
                start = line.find("{")
                end = line.rfind("}")
                if start != -1 and end != -1 and end > start:
                    frag = line[start:end+1]
                    yield json.loads(frag)
            except Exception:
                # 跳过无法解析的行
                continue


def load_minus_one_ids(refreshed_path):
    """
    从 refreshed JSONL 中加载所有 citation_count == -1 的 id（去重）。
    """
    minus_one_ids = set()
    total = 0
    for obj in tqdm(iter_jsonl(refreshed_path), desc="扫描 refreshed (-1) IDs"):
        total += 1
        cid = obj.get("citation_count", None)
        if cid == -1:
            pid = obj.get("id")
            if pid:
                minus_one_ids.add(str(pid))
    return minus_one_ids


def count_ids_in_slices(target_ids, slice_paths):
    """
    在每个切片文件中统计 target_ids 的出现数（按去重 ID 计）。
    返回：
      per_file_counts: dict[filename] = count
      seen_ids_global: set[ids]  # 在所有切片中至少出现过一次的 ID
    """
    per_file_counts = {}
    seen_ids_global = set()

    for spath in tqdm(slice_paths, desc="统计各切片文件"):
        seen_this_file = set()
        for obj in iter_jsonl(spath):
            pid = obj.get("id")
            if pid is None:
                continue
            pid = str(pid)
            if pid in target_ids:
                # 去重：同一文件同一 ID 只计一次
                if pid not in seen_this_file:
                    seen_this_file.add(pid)
        per_file_counts[os.path.basename(spath)] = len(seen_this_file)
        seen_ids_global.update(seen_this_file)

    return per_file_counts, seen_ids_global


def write_csv(per_file_counts, out_csv):
    # 按数量降序输出
    rows = sorted(per_file_counts.items(), key=lambda x: (-x[1], x[0]))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("filename,count\n")
        for name, cnt in rows:
            f.write(f"{name},{cnt}\n")


def write_json(per_file_counts, missing_ids_count, total_target_ids, total_seen_ids, out_json):
    data = {
        "per_file_counts": dict(sorted(per_file_counts.items(), key=lambda x: (-x[1], x[0]))),
        "summary": {
            "total_minus_one_ids": total_target_ids,
            "total_minus_one_ids_found_in_slices": total_seen_ids,
            "total_minus_one_ids_missing_in_all_slices": missing_ids_count,
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 1) 收集 -1 的 ID
    if not os.path.exists(REFRESHED_FILE):
        raise FileNotFoundError(f"找不到 refreshed 文件：{REFRESHED_FILE}")
    minus_one_ids = load_minus_one_ids(REFRESHED_FILE)
    print(f"\n[INFO] refreshed 中 citation_count == -1 的唯一 ID 数量：{len(minus_one_ids)}")

    # 2) 匹配切片文件
    slice_paths = sorted(glob.glob(SLICE_GLOB))
    if not slice_paths:
        raise FileNotFoundError(f"未匹配到任何切片文件（通配：{SLICE_GLOB}）")
    print(f"[INFO] 发现切片文件数量：{len(slice_paths)}")

    # 3) 统计每个切片文件里命中的 ID 数（去重）
    per_file_counts, seen_ids_global = count_ids_in_slices(minus_one_ids, slice_paths)

    # 4) 打印控制台汇总（按数量降序）
    print("\n==== 统计结果（按命中数量降序）====")
    for name, cnt in sorted(per_file_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{name:50s}  {cnt}")

    # 5) 汇总总计与缺失
    total_seen = len(seen_ids_global)
    missing = len(minus_one_ids - seen_ids_global)
    print("\n==== 汇总 ====")
    print(f"总计 -1 的唯一 ID 数量：{len(minus_one_ids)}")
    print(f"在切片中至少出现过一次的 -1 ID 数量：{total_seen}")
    print(f"在所有切片中完全没有出现的 -1 ID 数量：{missing}")

    # 6) 输出 CSV 报告
    write_csv(per_file_counts, OUT_CSV)
    print(f"\n已写出 CSV：{OUT_CSV}")


if __name__ == "__main__":
    main()
