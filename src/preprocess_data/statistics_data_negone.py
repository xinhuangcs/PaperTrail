import json
from collections import defaultdict
#config
REFRESHED_FILE = "/Users/jasonh/Desktop/02807/PaperTrail/data/preprocess/arxiv-cs-data-with-citations-refreshed-all-even.json"  # 你的文件名



def main():
    counts_by_month = defaultdict(int)
    total_minus_one = 0

    with open(REFRESHED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("citation_count") == -1:
                update_date = record.get("update_date", "unknown")

                # Extract the year and month part, e.g. "2017-02" or "unknown"
                month = update_date[:7] if len(update_date) >= 7 else "unknown"
                counts_by_month[month] += 1
                total_minus_one += 1

    # result
    print("\nCount of records with citation_count = -1 by month\n")
    if not counts_by_month:
        print("No records found with citation_count = -1.")
        return

    for month in sorted(counts_by_month):
        print(f"{month} : {counts_by_month[month]}")

    print("\nTotal:", total_minus_one, "records")
    print("Counting complete!")


if __name__ == "__main__":
    main()
