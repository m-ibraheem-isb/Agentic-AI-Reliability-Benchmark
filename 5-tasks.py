import os
import json
import time
import csv
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()
MODEL_NAME = "DeepSeek-V4-Pro"

client = OpenAI(
    base_url=os.getenv("DEEPSEEK_ENDPOINT"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

def multiply(a,b):
  return a*b

def search_document(query):
  file = open("notes.txt")
  lines = file.readlines()
  file.close()

  matches = []

  for line in lines:

    if query.lower() in line.lower():
      matches.append(line.strip())

  if len(matches) > 0:
    return "\n".join(matches)

  return "No match found"

def run_agent(question):
  messages = [
      {
          "role": "user",
          "content": question
      }
  ]

  tools = [
    {
    "type": "function",
    "function":
      {
      "name": "multiply",
      "description": "multiply two numbers",
      "parameters":
        {
        "type": "object",
        "properties":
          {
          "a":
            {
            "type": "number"
            },
          "b":
            {
            "type": "number"
            }
          },
          "required": ["a","b"]
        }
      }
    },
    {
    "type": "function",
    "function":
      {
        "name": "search_document",
        "description": "Search the local notes file for information.",
        "parameters":
          {
            "type": "object",
            "properties":
              {
                "query":
                  {
                    "type": "string"
                  }
              },
              "required": ["query"]
          }
        }
      }
  ]

  max_steps = 5
  step = 0
  tool_calls_count = 0
  input_tokens = 0
  output_tokens = 0

  while step < max_steps:

      step += 1
      retries = 0

      while True:

          try:
              response = client.chat.completions.create(
                  model=MODEL_NAME,
                  messages=messages,
                  tools=tools,
                  tool_choice="auto",
              )

              if(response.usage is not None):
                input_tokens += response.usage.prompt_tokens
                output_tokens += response.usage.completion_tokens
              else:
                print("Usage information missing")

              break

          except RateLimitError:

              retries += 1

              if retries == 3:
                  return "API rate limit error", step, tool_calls_count, input_tokens, output_tokens

              print("Rate limit hit. Waiting 10 seconds...")
              time.sleep(10)

      result = response.choices[0].message

      if result.tool_calls:

          tool_calls_count += 1

          print("LLM requested a tool")

          tool_call = result.tool_calls[0]

          tool_name = tool_call.function.name

          arguments = json.loads(
              tool_call.function.arguments
          )

          print("Tool:", tool_name)
          print("Arguments:", arguments)

          if tool_name == "multiply":

              tool_result = multiply(
                  arguments["a"],
                  arguments["b"]
              )

          elif tool_name == "search_document":

              tool_result = search_document(
                  arguments["query"]
              )

          print("Tool result:", tool_result)

          messages.append(result)

          messages.append(
              {
                  "role": "tool",
                  "tool_call_id": tool_call.id,
                  "content": str(tool_result)
              }
          )

      else:

          return result.content, step, tool_calls_count,input_tokens, output_tokens


  return "Agent stopped: maximum steps reached", step, tool_calls_count, input_tokens, output_tokens



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
    answer, steps, tool_calls, input_tokens, output_tokens = run_agent(question)
    end_time = time.perf_counter()

    runtime = end_time - start_time
    total_tokens = input_tokens + output_tokens

    print("Agent answer:", answer)
    print("Expected:", expected)
    print("Steps:", steps)
    print("Tool calls:", tool_calls)
    print("Input tokens:", input_tokens)
    print("Output tokens:", output_tokens)
    print("Total tokens:", total_tokens)

    normalized_answer = answer.lower().replace(",", "")
    normalized_expected = expected.lower().replace(",", "")

    if normalized_expected in normalized_answer:
        print("Result: PASS")
        passed += 1
        success = True
    else:
        print("Result: FAIL")
        success = False

    results.append(
      {
        "task_id": task["id"],
        "model": MODEL_NAME,
        "configuration": "tool_agent",
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

file = open("results.csv", "w", newline="", encoding="utf-8")

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

file.close()


total = len(tasks)

success_rate = (passed / total) * 100

print("\nPassed:", passed, "/", total)
print("Success rate:", success_rate, "%")
print("Results saved to results.csv")