import time
import json
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="")

def generate_requirements(n=300) -> list[str]:
    prompt = f"""
You are a professional Unreal Engine C++ instructor designing a practice dataset for beginner-level students.

Generate {n} varied and realistic Unreal Engine C++ implementation tasks.

Each task must follow these strict rules:

--- Implementation constraints ---
- Must be implementable in under 30 lines of code (header + cpp)
- Must involve exactly ONE class (e.g., an Actor or a Component)
- Must require common beginner-friendly features ONLY:
  - UCLASS, UPROPERTY, UFUNCTION, Tick, Input, Collision, BlueprintCallable, etc.
- The task must describe one concrete implementation objective, phrased clearly and naturally
- The output should be focused and non-ambiguous (no vague requirements)

--- Thematic variety (cover across the {n} samples) ---
Include many types of tasks such as:
- Creating a visible actor with a mesh
- Moving or rotating an actor every Tick
- Using UPROPERTY with EditAnywhere or metadata (e.g., ClampMin)
- Binding a key (e.g., "V" or "Space") to a function using SetupPlayerInputComponent
- Creating a component with a float property exposed to Blueprints
- Broadcasting a BlueprintAssignable delegate
- Loading assets using ConstructorHelpers
- Reacting to OnBeginOverlap with a printed message or property change

--- Forbidden topics ---
Avoid advanced Unreal topics including:
- Networking or Multiplayer
- AI Controllers or Behavior Trees
- UMG / Widgets
- Animation Graphs
- Skeletal mesh or montage usage

--- Output format ---
Return a raw JSON array of {n} strings.
Each string should be a single English sentence, describing a beginner Unreal Engine C++ implementation task.
Do NOT include explanations, formatting, numbering, markdown, or headings.
Only return the raw JSON array. Nothing else.
"""

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=10384,
            )
            content = response.choices[0].message.content.strip()

            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            json_block = content[json_start:json_end]

            return json.loads(json_block)
        except Exception as e:
            print(f"[Requirement Generation Error] {e}, retrying...")
            time.sleep(1)

# 프롬프트 기반 코드 생성
def generate_dataset_entry(requirement: str) -> dict:
    prompt = f"""
You are a senior Unreal Engine C++ engineer.

[Requirement]
{requirement}

Write Unreal Engine C++ code that satisfies the requirement.

Return ONLY a valid JSON object with the following keys:
- "requirement": (original requirement)
- "header_code": (full .h code including #pragma once and UCLASS/UPROPERTY macros)
- "cpp_code": (full .cpp code with correct includes, constructor, and methods)

The generated code MUST follow these rules:
- All classes must be prefixed with "A" (for Actor) or "U" (for UObject)
- Must include full UCLASS/USTRUCT boilerplate
- If a mesh is involved, use UStaticMeshComponent and ConstructorHelpers
- If using input, include SetupPlayerInputComponent override
- End with complete code fences (```cpp ... ```)

Do NOT include explanations, markdown, or additional formatting. Only return valid JSON.
"""


    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            message = response.choices[0].message.content.strip()
            result = json.loads(message)
            return wrap_code_blocks(result)
        except Exception as e:
            print(f"[Code Generation Error] {e}, retrying in 5s...")
            time.sleep(1)

# 코드블록 감싸기
def wrap_code_blocks(data: dict) -> dict:
    for key in ["header_code", "cpp_code"]:
        code = data.get(key, "").strip()
        if not code.startswith("```cpp"):
            data[key] = f"```cpp\n{code}\n```"
    return data


if __name__ == "__main__":
    output_path = "unreal_code_dataset.jsonl"
    num_requirements = 5

    print(f"Generating {num_requirements} Unreal Engine requirements...")
    requirements = generate_requirements(num_requirements)

    with open(output_path, "w", encoding="utf-8") as f:
        for req in tqdm(requirements):
            data = generate_dataset_entry(req)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(requirements)} Unreal Engine C++ entries to {output_path}")
