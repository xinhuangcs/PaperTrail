import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()




# Automatically load API Key from .env or environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# config
INPUT_FILE = "/Users/jasonh/Desktop/02807/PaperTrail/data/recommend/recommend_top5_theoretical_20251025_110755.json"

#read file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    papers = [json.loads(line) for line in f if line.strip()]


#get key infor
summary = []
for p in papers:
    summary.append({
        "id": p.get("id"),
        "title": p.get("title"),
        "abstract": (p.get("abstract") or "")[:600],
        "citation_count": p.get("citation_count"),
    })

#3) Calling the OpenAI Model
prompt = f"""
You are a research advisor. Based on the following paper information, please provide:
1. A recommended reading sequence (from introductory to advanced)
2. The rationale for reading each paper
3. Study advice for master's students (including how to approach reading in phases)

Paper information is as follows:
{json.dumps(summary, ensure_ascii=False, indent=2)}
"""

response = client.responses.create(
    model="gpt-5",
    input=prompt,
    max_output_tokens=5000,
)

#result
result = response.output_text
print("\n===== advice =====\n")
print(result)

output_path = "plan_reading_result.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result)

print(f"\nResults saved to: {os.path.abspath(output_path)}")
