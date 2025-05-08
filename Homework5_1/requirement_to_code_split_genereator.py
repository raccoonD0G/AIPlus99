import time
import json
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="")

def generate_requirements(n=100) -> list[str]:
    prompt = f"""
You are a creative Unreal Engine C++ designer.

Generate {n} **extremely super simple and beginner-friendly** Unreal Engine C++ requirements.

Each example should be very short, simple, and include creating a class, like "Create an actor with a visible mesh."

Each requirement should be a single sentence describing a concrete feature, component, or system to implement using Unreal Engine.

The requirements should:
- Only involve one class or one component
- Avoid complex logic or systems (no AI trees, no multiplayer logic, no advanced animation)
- Be suitable for Unreal Engine beginners
- Be implementable in under 30 lines of C++ code

Include a wide variety: basic actor behavior, input bindings, UPROPERTY fields, tick-based movement, simple blueprint integration, etc.

Return ONLY a raw JSON array of strings. No explanation, no markdown, no preamble.

"""

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()

            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            json_block = content[json_start:json_end]

            return json.loads(json_block)
        except Exception as e:
            print(f"[Requirement Generation Error] {e}, retrying...")
            time.sleep(5)

# 프롬프트 기반 코드 생성
def generate_dataset_entry(requirement: str) -> dict:
    prompt = f"""
You are an expert Unreal Engine C++ developer.

[Requirement]
{requirement}

Write Unreal Engine C++ code that satisfies the requirement.
Return ONLY a valid JSON object with:
- "requirement": The requirement
- "header_code": The .h C++ header code
- "cpp_code": The .cpp implementation

Do NOT include explanations or markdown. Just raw JSON.
"""

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500,
            )
            message = response.choices[0].message.content.strip()
            result = json.loads(message)
            return wrap_code_blocks(result)
        except Exception as e:
            print(f"[Code Generation Error] {e}, retrying in 5s...")
            time.sleep(5)

# 코드블록 감싸기
def wrap_code_blocks(data: dict) -> dict:
    for key in ["header_code", "cpp_code"]:
        code = data.get(key, "").strip()
        if not code.startswith("```cpp"):
            data[key] = f"```cpp\n{code}\n```"
    return data


if __name__ == "__main__":
    output_path = "unreal_code_dataset.jsonl"
    num_requirements = 100

    print(f"Generating {num_requirements} Unreal Engine requirements...")
    requirements = generate_requirements(num_requirements)

    with open(output_path, "w", encoding="utf-8") as f:
        for req in tqdm(requirements):
            data = generate_dataset_entry(req)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(requirements)} Unreal Engine C++ entries to {output_path}")
