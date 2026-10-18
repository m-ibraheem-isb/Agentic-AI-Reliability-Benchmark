import os
import json
import time
import csv
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  base_url = os.getenv("DEEPSEEK_ENDPOINT"),
  api_key = os.getenv("DEEPSEEK_API_KEY")
)

model_name = "DeepSeek-V4-Pro"

def run_baseline(question):

  steps = 0
  tool_calls = 0
  input_tokens = 0
  output_tokens = 0

  response = client.chat.completions.create(
    model = model_name,
    messages = [
      {
        "role" : "user",
        "content" : question
      }
    ]
  )

  steps += 1
  result = response.choices[0].message
  if result.tool_calls:
    tool_calls += 1

  if response.usage is not None:
    input_tokens += response.usage.prompt_tokens
    output_tokens += response.usage.completion_tokens
    total_tokens = input_tokens + output_tokens
  else:
    print("Usage information missing")

  return result.content,steps,tool_calls,input_tokens,output_tokens,total_tokens

file = open("5-tasks.json", "r", encoding="utf-8")
tasks = json.load(file)
file.close()

passed = 0
results = []

for task in tasks:

    print("\nRunning:", task["id"])

    question = task["question"]
    expected = task["expected_answer"]

    start_time = time.perf_counter()
    answer,steps,tool_calls,input_tokens,output_tokens,total_tokens = run_baseline(question)
    end_time = time.perf_counter()

    print("Baseline answer: ",answer)
    print("Expected: ",expected)


    runtime = end_time - start_time

    if(answer == expected):
      print("Result: Pass")
      passed += 1
      success = True
    else:
      print("Result: Fail")
      success = False

    results.append(
      {
        "task_id": task["id"],
        "model": model_name,
        "configuration": "direct_llm",
        "expected_answer": expected,
        "agent_answer": answer,
        "success": success,
        "steps": steps,
        "tool_calls": tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "runtime_seconds": round(runtime, 2)
      }
    )

file = open("baseline_results.csv", "w", newline="", encoding="utf-8")

fieldnames = [
    "task_id",
    "model",
    "configuration",
    "expected_answer",
    "agent_answer",
    "success",
    "steps",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "runtime_seconds"
]

writer = csv.DictWriter(file,fieldnames=fieldnames)
writer.writeheader()
writer.writerows(results)

total = len(tasks)
success_rate = (passed / total) * 100

print("\nPassed:", passed, "/", total)
print("Success rate:", success_rate, "%")
print("Results saved to baseline_results.csv")
