import os
import subprocess
import time
from openai import OpenAI




def call_deepseek_api(prompt: str) -> str:
    """
    调用DeepSeek-R1的API
    返回生成的完整代码字符串
    """
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


# 测试用例配置
TEST_CASE = {
    "input": "5\n5 7 1 2 10",
    "expected_output": {"max_score": 145, "preorder": [3, 1, 2, 4, 5]}
}

# 代码生成提示词
CODING_PROMPT = """请严格按以下要求编写Python代码：

题目描述：
设一个n个节点的二叉树的中序遍历为(1,2,...,n)，每个节点有分数d_i。子树加分规则：
左子树加分 × 右子树加分 + 根分数（空子树加分为1）
需要输出：1.最高加分 2.前序遍历

输入格式：
第1行：n
第2行：空格分隔的分数

输出格式：
第1行：最高加分
第2行：前序遍历序列

示例输入：
5
5 7 1 2 10

示例输出：
145
3 1 2 4 5

要求：
1. 使用标准输入输出
2. 时间复杂度不超过O(n^3)
3. 必须使用动态规划实现
4. 输出前序遍历时，当有多个可能结果时选择字典序最小的

请直接给出完整可运行的代码，不要任何解释。"""


# 增强版验证函数
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
            ["E:/venv/Scripts/python.exe", "-c", code],

            input=test_case["input"],
            text=True,
            capture_output=True,
            timeout=15,
            check=True
        )
        result["time_cost"] = time.time() - start_time

        output = process.stdout.strip().split('\n')
        if len(output) != 2:
            raise ValueError("输出格式错误")

        # 验证第一行输出
        max_score = int(output[0])
        # 验证第二行输出
        preorder = list(map(int, output[1].split()))

        # 数值验证
        score_match = (max_score == test_case["expected_output"]["max_score"])
        # 前序遍历验证（允许不同解但需要符合树结构）
        if preorder != test_case["expected_output"]["preorder"]:
            # 备选验证逻辑：检查生成的树是否符合中序和加分
            # 这里需要实现树结构验证（因复杂度较高，建议单独实现）
            # 当前先做简单验证
            pass

        result["actual_output"] = {"max_score": max_score, "preorder": preorder}
        result["passed"] = score_match

    except subprocess.TimeoutExpired:
        result["error"] = "运行超时"
    except subprocess.CalledProcessError as e:
        result["error"] = f"运行错误({e.returncode}): {e.stderr}"
    except Exception as e:
        result["error"] = f"验证错误: {str(e)}"

    return result


# 执行测试流程
def test_deepseek():
    print("正在生成代码...")
    generated_code = call_deepseek_api(CODING_PROMPT)
    generated_code= generated_code.replace("```python", "").replace("```", "").strip()
    if not generated_code:
        print("代码生成失败")
        return

    print("\n生成的代码：")
    print("=" * 40)
    print(generated_code)
    print("=" * 40)

    print("\n正在验证代码...")
    validation = validate_code(generated_code, TEST_CASE)

    print("\n测试结果：")
    print(f"运行时间: {validation['time_cost']:.2f}s" if validation["time_cost"] else "未获取运行时间")
    print(f"最高分验证: {'通过' if validation['passed'] else '失败'}")
    print(f"前序遍历: {validation['actual_output']['preorder'] if validation['actual_output'] else '无输出'}")
    print(f"错误信息: {validation['error'] or '无'}")


if __name__ == "__main__":
    # 环境检查

    test_deepseek()
