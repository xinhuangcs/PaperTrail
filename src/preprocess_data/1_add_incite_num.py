"""
添加引用数（OpenAlex）：
- 有 DOI：OpenAlex（DOI）
- 无 DOI：OpenAlex 标题搜索（相似度匹配）
- 引用数拿不到则写 -1，并写明 citation_status
- 逐行流式处理
- 切片处理
- tqdm 显示该切片进度 / 速率 / 预计剩余时间

依赖：
  pip install requests tqdm
"""

import json
import os
import sys
import time
import shutil
import signal
from typing import Optional, Tuple, Dict

import requests
from difflib import SequenceMatcher
from tqdm import tqdm


# ========= 1) 配置=========
CONFIG = {
    # 路径
    "INPUT_FILE": "/Users/jasonh/Desktop/tools/FinalProject/DataPreprocess/arxiv-metadata-oai-snapshot.json",
    "OUTPUT_FILE_BASE": "/Users/jasonh/Desktop/tools/FinalProject/DataPreprocess/arxiv-with-citations",  # 加 _slice{idx}.jsonl
    "CACHE_FILE": "/Users/jasonh/Desktop/tools/FinalProject/DataPreprocess/citation_cache.json",

    # 切片（从 0 开始，举例：切成 10 片，跑第 2 片 → SLICE_COUNT=10, SLICE_INDEX=1）
    "SLICE_COUNT": 10,
    "SLICE_INDEX": 0,

    # 速率控制（避免限流）
    "SLEEP_SECS": 0.2,    # 每次请求后暂停秒数（建议 0.15–0.25）
    "SAVE_EVERY": 1000,   # 每处理多少条保存一次缓存
    "TITLE_SIM_RATIO": 0.90,  # 标题匹配的相似度阈值
}

# ========= 2) 工具函数 =========
def normalize_doi(raw: Optional[str]) -> Optional[str]:
    """规范化 DOI：去空白、去掉(https)://doi.org/ 前缀、转小写"""
    if not raw or not isinstance(raw, str):
        return None
    doi = raw.strip()
    if not doi:
        return None
    doi = doi.replace(" ", "")
    low = doi.lower()
    if low.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif low.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]
    return doi.lower()


def title_similarity(a: str, b: str) -> float:
    """标题相似度 [0,1]"""
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


# ========= 3) OpenAlex 调用 =========
def openalex_get_by_doi(doi: str, session: requests.Session,
                        sleep_secs: float = 0.2,
                        max_retries: int = 3,
                        timeout: float = 20.0) -> Tuple[int, str]:
    """用 DOI 调 OpenAlex，返回 (citation_count, status)"""
    base = "https://api.openalex.org/works"
    params = {"filter": f"doi:{doi}"}
    attempt = 0
    while True:
        try:
            resp = session.get(base, params=params, timeout=timeout)
            code = resp.status_code
            if code == 200:
                data = resp.json()
                results = data.get("results", [])
                nd = normalize_doi(doi)
                for item in results:
                    item_doi = normalize_doi(item.get("doi"))
                    if item_doi and item_doi == nd:
                        cbc = item.get("cited_by_count")
                        if isinstance(cbc, int):
                            time.sleep(sleep_secs)
                            return cbc, "ok_openalex_doi"
                        else:
                            return -1, "openalex_no_cited_by_count"
                return -1, "openalex_not_found_doi"
            elif code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    return -1, f"openalex_http_{code}_fail"
                backoff = sleep_secs * (2 ** attempt) + 0.05 * attempt
                time.sleep(backoff)
                continue
            else:
                return -1, f"openalex_http_{code}"
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                return -1, f"openalex_exception:{type(e).__name__}"
            backoff = sleep_secs * (2 ** attempt) + 0.05 * attempt
            time.sleep(backoff)


def openalex_get_by_title(title: str, session: requests.Session,
                          sleep_secs: float = 0.2,
                          timeout: float = 20.0,
                          min_ratio: float = 0.9) -> Tuple[int, str]:
    """用标题在 OpenAlex 搜索（返回最相近的一条的引用数）"""
    base = "https://api.openalex.org/works"
    params = {"filter": f"title.search:{title}", "per-page": 5}
    try:
        resp = session.get(base, params=params, timeout=timeout)
        if resp.status_code != 200:
            return -1, f"openalex_title_http_{resp.status_code}"
        data = resp.json()
        results = data.get("results", [])
        best_item, best_sim = None, 0.0
        for item in results:
            cand_title = item.get("title", "") or ""
            sim = title_similarity(title, cand_title)
            if sim > best_sim:
                best_sim = sim
                best_item = item
        if best_item and best_sim >= min_ratio:
            cbc = best_item.get("cited_by_count")
            if isinstance(cbc, int):
                time.sleep(sleep_secs)
                return cbc, "ok_openalex_title"
        return -1, "openalex_title_not_found"
    except requests.RequestException as e:
        return -1, f"openalex_title_exception:{type(e).__name__}"


# ========= 4) 缓存 =========
def load_cache(cache_path: Optional[str]) -> Dict[str, Dict[str, object]]:
    """加载缓存 JSON：{key: {"count": int, "status": str}}，key 可以是 DOI 或 'title::<title>'"""
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def atomic_save_json(obj, path: str):
    """原子写入 JSON，避免中断导致文件损坏"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    shutil.move(tmp, path)


# ========= 5) 主流程 =========
def main():
    # 读取配置
    INPUT_FILE      = CONFIG["INPUT_FILE"]
    OUTPUT_FILE_BASE= CONFIG["OUTPUT_FILE_BASE"]
    CACHE_FILE      = CONFIG["CACHE_FILE"]
    SLICE_COUNT     = int(CONFIG["SLICE_COUNT"])
    SLICE_INDEX     = int(CONFIG["SLICE_INDEX"])
    SLEEP_SECS      = float(CONFIG["SLEEP_SECS"])
    SAVE_EVERY      = int(CONFIG["SAVE_EVERY"])
    TITLE_SIM_RATIO = float(CONFIG["TITLE_SIM_RATIO"])

    # 输出文件按切片编号自动命名
    OUTPUT_FILE = f"{OUTPUT_FILE_BASE}_slice{SLICE_INDEX}.jsonl"

    print(f"[i] 参数：")
    print(f"    INPUT_FILE  = {INPUT_FILE}")
    print(f"    OUTPUT_FILE = {OUTPUT_FILE}")
    print(f"    CACHE_FILE  = {CACHE_FILE}")
    print(f"    SLICE       = {SLICE_INDEX}/{SLICE_COUNT - 1}")
    print(f"    SLEEP_SECS  = {SLEEP_SECS}")

    # 友好退出
    interrupted = {"flag": False}
    def handle_sigint(signum, frame):
        interrupted["flag"] = True
        print("\n 中断，保存缓存", file=sys.stderr)
    signal.signal(signal.SIGINT, handle_sigint)

    # 缓存
    cache = load_cache(CACHE_FILE)
    print(f"[i] 已加载缓存：{len(cache)} 条")

    # 统计总行数，计算切片范围
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    if SLICE_COUNT < 1:
        SLICE_COUNT = 1
    if SLICE_INDEX < 0 or SLICE_INDEX >= SLICE_COUNT:
        print(f"[x] SLICE_INDEX 超出范围 (0..{SLICE_COUNT-1})", file=sys.stderr)
        sys.exit(2)

    lines_per_slice = total_lines // SLICE_COUNT
    start_line = SLICE_INDEX * lines_per_slice + 1
    end_line = (SLICE_INDEX + 1) * lines_per_slice if SLICE_INDEX < SLICE_COUNT - 1 else total_lines
    total_this_slice = end_line - start_line + 1
    print(f"[i] 总行数={total_lines}, 本片范围={start_line} ~ {end_line}（共 {total_this_slice} 行）")

    # 输出准备
    out_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_f = open(OUTPUT_FILE, "w", encoding="utf-8")

    session = requests.Session()
    session.headers.update({"User-Agent": "dtuproject-citation-enricher/1.0"})

    processed, written, last_cache_dump = 0, 0, 0

    # tqdm 只覆盖当前切片
    with open(INPUT_FILE, "r", encoding="utf-8") as in_f, \
         tqdm(total=total_this_slice, unit="rec", desc=f"Slice {SLICE_INDEX}/{SLICE_COUNT-1}") as pbar:
        for lineno, line in enumerate(in_f, start=1):
            if lineno < start_line:
                continue
            if lineno > end_line:
                break
            if interrupted["flag"]:
                break

            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                pbar.update(1)
                continue

            doi = normalize_doi(rec.get("doi"))
            title = (rec.get("title") or "").strip()

            citation_count, citation_status = -1, "missing_id"

            if doi:
                # 以 DOI 为缓存 key
                if doi in cache:
                    citation_count = cache[doi]["count"]
                    citation_status = cache[doi]["status"]
                else:
                    # 仅用 OpenAlex（DOI）
                    citation_count, citation_status = openalex_get_by_doi(doi, session, sleep_secs=SLEEP_SECS)
                    cache[doi] = {"count": citation_count, "status": citation_status}
            else:
                # 无 DOI → 标题搜索（OpenAlex），cache key 用 title::
                if title:
                    cache_key = f"title::{title}"
                    if cache_key in cache:
                        citation_count = cache[cache_key]["count"]
                        citation_status = cache[cache_key]["status"]
                    else:
                        citation_count, citation_status = openalex_get_by_title(
                            title, session, sleep_secs=SLEEP_SECS, min_ratio=TITLE_SIM_RATIO
                        )
                        cache[cache_key] = {"count": citation_count, "status": citation_status}
                else:
                    citation_count, citation_status = -1, "no_doi_no_title"

            # 写回
            rec["citation_count"] = citation_count
            rec["citation_status"] = citation_status
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            written += 1
            processed += 1

            # 定期保存缓存
            if CACHE_FILE and (processed - last_cache_dump) >= CONFIG["SAVE_EVERY"]:
                atomic_save_json(cache, CACHE_FILE)
                last_cache_dump = processed

            pbar.update(1)

    # 最终缓存保存
    if CACHE_FILE:
        atomic_save_json(cache, CACHE_FILE)

    out_f.close()
    print(f"处理完成: processed={processed}, written={written}")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
