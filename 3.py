import os
import subprocess
import time
import psutil
import matplotlib.pyplot as plt
from openai import OpenAI


def call_deepseek_api(prompt: str) -> str:
    client = OpenAI(api_key="sk-d86a3cacb9fb4007abe4a848625da226", base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是一个专业的算法工程师，请用Python编写准确高效的代码"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return ""


TEST_CASES = [
    {"input": "1 1 10\n5", "expected_output": 9},
    {"input": "8 2 10000\n23", "expected_output": 9001},
    {"input": "4 3 100\n111", "expected_output": 81},
    {"input": "10 3 5000\n987", "expected_output": 4960},
    {"input": "100 10 5000\n9876543210", "expected_output": 1}
]


CODING_PROMPT = """请严格按以下要求编写Python代码：

阿申准备报名参加 GT 考试，准考证号为 N 位数X1,X2…Xn (0≤Xi≤9)，他不希望准考证号上出现不吉利的数字。 他的不吉利数字A1,A2,⋯,Am (0≤Ai≤9) 有 M 位，
不出现是指 X1,X2⋯Xn 中没有一段恰好等于 A1,A2,⋯,Am，A1 和 X1 可以为 0。
输入格式
第一行输入 N,M,K 接下来一行输入 M 位的数。
输出格式
阿申想知道不出现不吉利数字的号码有多少种，输出模 K 取余的结果。
输入输出样例
输入 #1
4 3 100
111
输出 #1
81
请直接给出完整可运行的代码，不要任何解释！"""


def validate_code(code: str, test_case: dict) -> dict:
    result = {
        "compiled": False,
        "passed": False,
        "time_cost": None,
        "memory_usage": None,
        "error": None,
        "actual_output": None,
        "input_size": len(test_case["input"])  # 计算输入大小
    }

    try:
        start_time = time.time()
        process = psutil.Process(os.getpid())  # 获取当前进程
        mem_before = process.memory_info().rss  # 记录内存前

        proc = subprocess.run(
            ["python", "-c", code],
            input=test_case["input"],
            text=True,
            capture_output=True,
            timeout=15,
            check=True
        )

        mem_after = process.memory_info().rss  # 记录内存后
        result["memory_usage"] = (mem_after - mem_before) / 1024  # 计算占用的KB

        result["time_cost"] = time.time() - start_time

        output = proc.stdout.strip()
        numbers = [int(part) for part in output.split() if part.isdigit()]

        if not numbers:
            raise ValueError("输出中未找到有效数字")

        result_value = numbers[-1] % 1000000007
        result["actual_output"] = result_value
        result["passed"] = (result_value == test_case["expected_output"])

    except subprocess.TimeoutExpired:
        result["error"] = "运行超时"
    except subprocess.CalledProcessError as e:
        error_msg = f"运行错误({e.returncode}):\nStdout: {e.stdout}\nStderr: {e.stderr}"
        result["error"] = error_msg[:200]
    except Exception as e:
        result["error"] = f"验证错误: {str(e)[:200]}"

    return result


def test_deepseek():
    print("正在生成代码...")
    generated_code = call_deepseek_api(CODING_PROMPT)
    generated_code = generated_code.replace("```python", "").replace("```", "").strip()

    if not generated_code:
        print("代码生成失败")
        return

    print("\n生成的代码：")
    print("=" * 40)
    print(generated_code)
    print("=" * 40)

    print("\n正在验证代码...")
    input_sizes = []
    memory_usages = []
    validations = []

    for case in TEST_CASES:
        result = validate_code(generated_code, case)
        validations.append(result)
        if result["memory_usage"] is not None:
            input_sizes.append(result["input_size"])
            memory_usages.append(result["memory_usage"])

    print("\n测试结果：")
    for i, validation in enumerate(validations):
        print(f"测试用例 #{i+1}:")
        print(f"输入:\n{TEST_CASES[i]['input']}")
        print(f"预期输出: {TEST_CASES[i]['expected_output']}")
        print(f"实际输出: {validation.get('actual_output', '无')}")
        print(f"运行时间: {validation['time_cost']:.2f}s" if validation['time_cost'] else "运行时间: 未获取")
        print(f"内存使用: {validation['memory_usage']} KB" if validation['memory_usage'] else "内存使用: 未获取")
        print(f"结果: {'通过' if validation['passed'] else '失败'}")
        print(f"错误信息: {validation['error'] or '无'}")
        print("-" * 40)

    # 进行可视化
    plt.figure(figsize=(8, 5))
    plt.plot(input_sizes, memory_usages, marker="o", linestyle="-", color="b", label="Memory Usage vs. Input Size")
    plt.xlabel("Input Size (bytes)")
    plt.ylabel("Memory Usage (KB)")
    plt.title("Memory Usage vs. Input Size")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_deepseek()
