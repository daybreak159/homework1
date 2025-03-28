import os
import subprocess
import time
import psutil
import matplotlib.pyplot as plt
from openai import OpenAI
import re
import ast


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


# 幻觉检测函数
def detect_hallucinations(code: str) -> dict:
    """
    检测代码中的幻觉现象，包括：
    1. 引用不存在的库或模块
    2. 使用未定义的函数或变量
    3. 错误的API调用方式
    4. 不符合语法的代码结构
    """
    hallucinations = {
        "nonexistent_imports": [],
        "undefined_functions": [],
        "api_misuse": [],
        "syntax_errors": [],
        "detected": False
    }
    
    # 标准Python库列表（部分）
    standard_libs = ["math", "re", "os", "sys", "time", "random", "collections", 
                     "itertools", "functools", "datetime", "json", "numpy", "pandas"]
    
    # 1. 检测引用不存在的库或模块
    import_pattern = re.compile(r'import\s+(\w+)|from\s+(\w+)\s+import')
    for match in import_pattern.finditer(code):
        lib_name = match.group(1) or match.group(2)
        if lib_name not in standard_libs and lib_name not in ["__future__"]:
            hallucinations["nonexistent_imports"].append(lib_name)
    
    # 2. 使用未定义的函数或变量
    try:
        tree = ast.parse(code)
        # 收集定义的函数和变量
        defined_names = set()
        used_names = set()
        
        # 收集函数定义和变量赋值
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_names.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
        
        # 排除内置函数和常见库函数
        builtin_funcs = ["print", "input", "len", "range", "int", "str", "list", "dict", "set", 
                        "sum", "min", "max", "abs", "all", "any", "enumerate", "map", "filter",
                        "zip", "sorted", "reversed"]
        for name in builtin_funcs:
            if name in used_names:
                used_names.remove(name)
        
        # 检测未定义就使用的变量
        for name in used_names:
            if name not in defined_names and not name.startswith(('__', 'math.', 're.', 'collections.')):
                hallucinations["undefined_functions"].append(name)
                
    except SyntaxError:
        hallucinations["syntax_errors"].append("代码存在语法错误")
    except Exception as e:
        hallucinations["syntax_errors"].append(f"AST分析错误: {str(e)}")
    
    # 3. 错误的API调用方式
    api_patterns = {
        r'\.read\(\s*\d+\s*,': "文件读取参数错误",
        r'numpy\.array\(\s*\[.*\]\s*,\s*dtype\s*=\s*[\'"]': "numpy dtype参数错误",
        r'open\([^,]+\)\s*\.[^.]+\(\).*\n(?!.*close)': "可能未关闭文件",
        r'pandas\.read_csv\(\s*[^,\)]*\s*,\s*encoding\s*=\s*[\'"]utf8[\'"]': "pandas编码错误(应为utf-8)",
        r'json\.loads\(\s*[^,\)]*\s*,\s*[^\)]+\)': "json.loads参数错误"
    }
    
    for pattern, message in api_patterns.items():
        if re.search(pattern, code):
            hallucinations["api_misuse"].append(message)
    
    # 4. 不符合语法的代码结构
    # 除了上面的SyntaxError检测外，这里添加一些特定模式
    syntax_patterns = {
        r'if.*:(?!\s*\n)': "if语句后缺少换行",
        r'else(?!\s*:)': "else后缺少冒号",
        r'for.*in.*(?!\s*:)': "for循环缺少冒号",
        r'def\s+\w+(?!\s*\()': "函数定义缺少括号",
        r'return\s+if': "return语句后不应直接跟if",
        r'\s+catch\s+': "Python使用except而非catch",
        r'switch\s*\(': "Python不支持switch语句"
    }
    
    for pattern, message in syntax_patterns.items():
        if re.search(pattern, code):
            hallucinations["syntax_errors"].append(message)
    
    # 汇总结果
    if (hallucinations["nonexistent_imports"] or 
        hallucinations["undefined_functions"] or 
        hallucinations["api_misuse"] or 
        hallucinations["syntax_errors"]):
        hallucinations["detected"] = True
    
    return hallucinations


def validate_code(code: str, test_case: dict) -> dict:
    result = {
        "compiled": False,
        "passed": False,
        "time_cost": None,
        "memory_usage": None,
        "error": None,
        "actual_output": None,
        "input_size": len(test_case["input"]),  # 计算输入大小
        "hallucination": False,
        "hallucination_details": None
    }

    # 幻觉检测
    hallucination_results = detect_hallucinations(code)
    result["hallucination"] = hallucination_results["detected"]
    result["hallucination_details"] = hallucination_results
    
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
        result["compiled"] = True

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

    # 首先进行幻觉检测
    print("\n进行幻觉检测...")
    hallucination_results = detect_hallucinations(generated_code)
    if hallucination_results["detected"]:
        print("⚠️ 检测到代码幻觉现象:")
        if hallucination_results["nonexistent_imports"]:
            print(f"- 引用不存在的库: {', '.join(hallucination_results['nonexistent_imports'])}")
        if hallucination_results["undefined_functions"]:
            print(f"- 使用未定义的函数或变量: {', '.join(hallucination_results['undefined_functions'])}")
        if hallucination_results["api_misuse"]:
            print(f"- API使用错误: {', '.join(hallucination_results['api_misuse'])}")
        if hallucination_results["syntax_errors"]:
            print(f"- 语法问题: {', '.join(hallucination_results['syntax_errors'])}")
    else:
        print("✓ 未检测到明显的代码幻觉现象")

    print("\n正在验证代码...")
    input_sizes = []
    memory_usages = []
    time_costs = []
    validations = []
    hallucination_count = 0

    for case in TEST_CASES:
        result = validate_code(generated_code, case)
        validations.append(result)
        if result["hallucination"]:
            hallucination_count += 1
        if result["memory_usage"] is not None:
            input_sizes.append(result["input_size"])
            memory_usages.append(result["memory_usage"])
        if result["time_cost"] is not None:
            time_costs.append(result["time_cost"])

    print("\n测试结果：")
    for i, validation in enumerate(validations):
        print(f"测试用例 #{i+1}:")
        print(f"输入:\n{TEST_CASES[i]['input']}")
        print(f"预期输出: {TEST_CASES[i]['expected_output']}")
        print(f"实际输出: {validation.get('actual_output', '无')}")
        print(f"运行时间: {validation['time_cost']:.4f}s" if validation['time_cost'] else "运行时间: 未获取")
        print(f"内存使用: {validation['memory_usage']:.2f} KB" if validation['memory_usage'] else "内存使用: 未获取")
        print(f"幻觉情况: {'有' if validation['hallucination'] else '无'}")
        print(f"结果: {'通过' if validation['passed'] else '失败'}")
        print(f"错误信息: {validation['error'] or '无'}")
        print("-" * 40)

    # 显示汇总结果
    passed_tests = len([v for v in validations if v['passed']])
    print(f"\n总结: {passed_tests}/{len(validations)}测试通过 ({passed_tests/len(validations)*100:.1f}%)")
    print(f"幻觉出现率: {hallucination_count/len(validations)*100:.1f}%")
    
    if time_costs:
        print(f"平均执行时间: {sum(time_costs)/len(time_costs):.4f}s")
    if memory_usages:
        print(f"平均内存使用: {sum(memory_usages)/len(memory_usages):.2f} KB")

    # 进行可视化
    if input_sizes and memory_usages:
        plt.figure(figsize=(15, 12))
        
        # 1. 绘制内存使用图
        plt.subplot(2, 2, 1)
        plt.plot(input_sizes, memory_usages, marker="o", linestyle="-", color="b")
        plt.xlabel("输入大小 (字节)")
        plt.ylabel("内存使用 (KB)")
        plt.title("内存使用与输入大小关系")
        plt.grid(True)
        
        # 2. 绘制幻觉情况饼图
        plt.subplot(2, 2, 2)
        labels = ['存在幻觉', '无幻觉']
        sizes = [hallucination_count, len(validations)-hallucination_count]
        colors = ['#ff9999','#66b3ff']
        explode = (0.1, 0)  # 突出显示幻觉部分
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # 保持饼图为圆形
        plt.title('代码幻觉检测结果')
        
        # 3. 通过/失败测试用例饼图
        plt.subplot(2, 2, 3)
        labels = ['通过', '失败']
        sizes = [passed_tests, len(validations)-passed_tests]
        colors = ['#99ff99','#ffcc99']
        explode = (0.1, 0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('测试用例通过情况')
        
        # 4. 如果有执行时间数据，绘制执行时间图表
        if time_costs and len(time_costs) == len(input_sizes):
            plt.subplot(2, 2, 4)
            plt.plot(input_sizes, time_costs, marker="s", linestyle="-", color="g")
            plt.xlabel("输入大小 (字节)")
            plt.ylabel("执行时间 (秒)")
            plt.title("执行时间与输入大小关系")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("kmp_test_results.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    test_deepseek()
