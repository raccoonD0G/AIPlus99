from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Any
import torch


class ChatCodeGenerator(BaseChatModel):
    def __init__(self, model: "CodeGenerator"):
        self._model = model

    def _llm_type(self) -> str:
        return "custom-code-generator"

    @property
    def header_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template("""You are an Unreal Engine C++ developer.

Write only the **Unreal Engine C++ header file (.h)** that satisfies the following requirement.
Start your C++ code inside a ```cpp code block and close it properly with ```.
Do not include any implementation or .cpp code.

Always start with a header file declaration like:
```cpp
// MovingActor.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MovingActor.generated.h"

Strict mode: Output only raw C++ code. Any non-code output is considered a mistake.

Here is an example.

Requirement:
Create an actor that moves forward constantly

Your Answer:
```cpp
// MovingActor.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MovingActor.generated.h"

UCLASS()
class MYPROJECT_API AMovingActor : public AActor
{{
    GENERATED_BODY()

    public:
    AMovingActor();

protected:
    virtual void BeginPlay() override;

public:
    virtual void Tick(float DeltaTime) override;

private:
    UPROPERTY(EditAnywhere)
    FVector MovementSpeed;
}};
```

The Requirement below is what you need to implement in Your Answer. **Answer Code Only**

Requirement:
{requirement}

Your Answer:
""")

    @property
    def cpp_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template("""You are an Unreal Engine C++ developer.

Write only the corresponding **Unreal Engine .cpp file** implementation.
Start your C++ code inside a ```cpp code block and close it properly with ```.

Always start with a header file declaration like:
```cpp
// MovingActor.cpp
#include "MovingActor.h"

AMovingActor::AMovingActor()
{{

Strict mode: Output only raw C++ code. Any non-code output is considered a mistake.

Here is an example.

Requirement:
Create an actor that moves forward constantly

Your Answer:
```cpp
// MovingActor.cpp
#include "MovingActor.h"

AMovingActor::AMovingActor()
{{
    PrimaryActorTick.bCanEverTick = true;
    MovementSpeed = FVector(100.f, 0.f, 0.f);
}}

void AMovingActor::BeginPlay()
{{
    Super::BeginPlay();
}}

void AMovingActor::Tick(float DeltaTime)
{{
    Super::Tick(DeltaTime);
    AddActorLocalOffset(MovementSpeed * DeltaTime);
}}
```

The Requirement below is what you need to implement in Your Answer. The Header File below is the header file (.h) that satisfies the requirement. Implement this Header File into Your Answer(.cpp).
**Answer Code Only**

Requirement:
{requirement}

Header File:
{header_code}

Your Answer:
""")

    def _generate(self, messages: List[HumanMessage], stop: List[str] = None) -> ChatResult:
        requirement = " ".join(m.content for m in messages if isinstance(m, HumanMessage))

        # header 생성
        header_prompt_str = self.header_prompt.format(requirement=requirement)
        with torch.no_grad():
            header_out = self._model.sample_code_batch_with_partial_grad([header_prompt_str])
        header_code = header_out["texts"][0]

        # cpp 생성
        cpp_prompt_str = self.cpp_prompt.format(requirement=requirement, header_code=header_code)
        with torch.no_grad():
            cpp_out = self._model.sample_code_batch_with_partial_grad([cpp_prompt_str])
        cpp_code = cpp_out["texts"][0]

        full_output = header_code + "\n\n" + cpp_code

        return ChatResult(
            generations=[
                ChatGeneration(
                    text=full_output,
                    message=AIMessage(content=full_output)
                )
            ]
        )

    def invoke(self, messages: List[Any]) -> AIMessage:
        return self._generate(messages).generations[0].message
