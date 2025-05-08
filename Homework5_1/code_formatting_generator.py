import json
import time
import random
from openai import OpenAI

client = OpenAI(api_key="")

BATCH_SIZE = 10
TOTAL_PAIRS = 2000
OUTPUT_FILE = "code_formatting_pairs.jsonl"
STYLE_GUIDE_FILE = "StyleGuide.md"

with open(STYLE_GUIDE_FILE, "r", encoding="utf-8") as f:
    full_style_guide = f.read()

# Chunk0 (사용 언어) + 나머지 Style Guide 분리
chunks = full_style_guide.split("## [Chunk")
chunk_map = {}
for chunk in chunks[1:]:
    header, *content = chunk.split("\n", 1)
    chunk_number = int(header.split("]")[0])
    header_text = header.split("]", 1)[1].strip()

    if chunk_number == 0:
        chunk_content = header_text
    else:
        chunk_content = content[0].strip() if content else ""

    chunk_map[chunk_number] = chunk_content


# 언어 정보는 Chunk0의 첫 번째 줄
language_info = chunk_map[0]

# 스타일 가이드는 Chunk1 이후를 합친 것
style_guide_body = "\n\n".join([chunk_map[i] for i in sorted(chunk_map.keys()) if i != 0])

# 코드 쌍 생성
def generate_code_pairs(batch_size, style_guide, language_info):
    prompt = f"""
You are a code style expert specialized in the following environment:

{language_info}

Below is the coding style guide you must strictly follow:

{style_guide}

Your task is to generate {batch_size} independent pairs of BAD_CODE and GOOD_CODE.

Rules:
- Each BAD_CODE and GOOD_CODE must be **pure code only**, without any extra text, explanations, or comments like "Pair N", "Example", or "Notes".
- Do NOT add any additional comments, pair numbers, headers, or separators outside of the BAD_CODE and GOOD_CODE blocks.
- Each BAD_CODE and GOOD_CODE should be separated exactly as shown below.

**BAD_CODE and GOOD_CODE formatting:**
BAD_CODE:
```{language_info}
(your bad code here)
```
- GOOD_CODE:
```{language_info}
(your corrected good code here)
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict code style reviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
    )

    text = response.choices[0].message.content.strip()

  
    pairs = []
    blocks = text.split("BAD_CODE:")
    for block in blocks[1:]:
        try:
            bad_part, good_part = block.split("GOOD_CODE:")
            bad_code = bad_part.replace(f"```{language_info}", "").replace("```", "").strip()
            good_code = good_part.replace(f"```{language_info}", "").replace("```", "").strip()
            pairs.append({"bad_code": bad_code, "good_code": good_code})
        except Exception as e:
            print(f"Skipping block due to parsing error: {e}")
            continue

    return pairs

def main():
    all_pairs = []
    while len(all_pairs) < TOTAL_PAIRS:
        try:
            new_pairs = generate_code_pairs(BATCH_SIZE, style_guide_body, language_info)
            print(new_pairs)
            all_pairs.extend(new_pairs)
            print(f"Generated code pairs: {len(all_pairs)}")
            time.sleep(1)
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(5)

    all_pairs = all_pairs[:TOTAL_PAIRS]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False))
            f.write("\n")

    print(f"Successfully saved {TOTAL_PAIRS} code pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
