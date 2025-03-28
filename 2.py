import os
import subprocess
import time
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
    {
        "input": "4\n0 1 1 2",
        "expected_output": 6
    },
    {
        "input": "4\n0 1 2 0",
        "expected_output": 0
    },
    {
        "input": "3\n1 1 2",
        "expected_output": 0
    },
    {
        "input": "17\n0 1 1 2 2 4 3 2 4 5 3 3 2 1 5 4 2",
        "expected_output": 855391686
    }
]

CODING_PROMPT = """请严格按以下要求编写Python代码：

题目描述：
一个图有n个顶点，每条边长度为1，没有重边，满足从顶点1到顶点i的最短距离为A_i。计算符合条件的图总数，结果对1000000007取模。

输入格式：
第一行一个整数n。
第二行n个整数，分别为A1到An。

输出格式：
一行，一个整数，为符合条件的图总数 mod 1000000007。

示例输入1：
4
0 1 1 2

示例输出1：
6

要求：
1. 必须使用标准输入输出
2. 正确处理无法构造的情况（如A1≠0、存在比邻接点距离差超过1的情况）
3. 使用模运算1000000007处理大数
4. 确保时间复杂度合理，能够处理大n的情况

请直接给出完整可运行的代码，不要任何解释。"""


def validate_code(code: str, test_case: dict) -> dict:
    result = {
        "compiled": False,
        "passed": False,
        "time_cost": None,
        "memory_usage": None,
        "error": None,
        "actual_output": None
    }

    try:
        start_time = time.time()
        process = subprocess.run(
            ["python", "-c", code],
            input=test_case["input"],
            text=True,
            capture_output=True,
            timeout=15,
            check=True
        )
        result["time_cost"] = time.time() - start_time

        output = process.stdout.strip()
        numbers = []
        for part in output.split():
            if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                numbers.append(int(part))

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
    validations = []
    for case in TEST_CASES:
        validations.append(validate_code(generated_code, case))

    print("\n测试结果：")
    for i, validation in enumerate(validations):
        print(f"测试用例 #{i+1}:")
        print(f"输入:\n{TEST_CASES[i]['input']}")
        print(f"预期输出: {TEST_CASES[i]['expected_output']}")
        print(f"实际输出: {validation.get('actual_output', '无')}")
        print(f"运行时间: {validation['time_cost']:.2f}s" if validation['time_cost'] else "运行时间: 未获取")
        print(f"结果: {'通过' if validation['passed'] else '失败'}")
        print(f"错误信息: {validation['error'] or '无'}")
        print("-" * 40)


if __name__ == "__main__":
    test_deepseek()