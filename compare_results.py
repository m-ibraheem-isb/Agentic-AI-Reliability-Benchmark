import csv

files = ["baseline_results.csv", "results.csv"]
names = ["Direct LLM", "Tool Agent"]

for i in range(2):

    with open(files[i], "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    total_tasks = len(rows)

    total_pass = 0
    total_tokens = 0
    total_steps = 0
    total_tool_calls = 0

    for task in rows:

        if task["success"] == "True":
            total_pass += 1

        total_tokens += int(task["total_tokens"])
        total_steps += int(task["steps"])
        total_tool_calls += int(task["tool_calls"])

    if total_tasks > 0:
        success_rate = (total_pass / total_tasks) * 100
        avg_tokens = total_tokens / total_tasks
        avg_steps = total_steps / total_tasks
        avg_tool_calls = total_tool_calls / total_tasks
    else:
        success_rate = 0
        avg_tokens = 0
        avg_steps = 0
        avg_tool_calls = 0

    print("\nConfiguration:", names[i])
    print("Total Tasks:", total_tasks)
    print("Passed Tasks:", total_pass)
    print("Success Rate:", round(success_rate, 2), "%")
    print("Average Tokens:", round(avg_tokens, 2))
    print("Average Steps:", round(avg_steps, 2))
    print("Average Tool Calls:", round(avg_tool_calls, 2))