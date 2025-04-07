import random
from itertools import permutations
from collections import defaultdict, deque
from typing import Sequence
from itertools import product
from hashlib import sha256
import time
from statistics import mean

class MealyNFA:
    def __init__(self, states, inputs, outputs, transition_func, initial_state, accepting_states=None):
        self.states = states
        self.inputs = inputs
        self.outputs = outputs
        self.transition_func = transition_func
        self.initial_state = initial_state
        self.current_state = initial_state
        self.accepting_states = accepting_states if accepting_states is not None else set()

    def is_accepting(self, state=None):
        if state is None:
            state = self.current_state
        return state in self.accepting_states

    def step(self, input_symbol):
        if input_symbol not in self.inputs:
            raise ValueError(f"Invalid input symbol: {input_symbol}")
        
        transitions = self.transition_func[self.current_state].get(input_symbol, [])
        if not transitions:
            raise ValueError(f"No transition defined for state {self.current_state} with input {input_symbol}")
        
        output_symbol, next_state = random.choice(transitions)
        self.current_state = next_state
        return output_symbol
    
    def reset(self):
        self.current_state = self.initial_state
    
    def generate_accepting_path(self, max_steps=100):
        original_state = self.current_state
        self.reset()

        queue = deque([(self.current_state, [], [])])
        visited = set()

        while queue:
            current, path_details, input_sequence = queue.popleft()
            
            if self.is_accepting(current):
                self.current_state = original_state
                return (path_details, input_sequence)
            
            if current in visited:
                continue
            visited.add(current)
            
            for a in self.inputs:
                transitions = self.transition_func[current].get(a, [])
                for (output, next_state) in transitions:
                    if next_state not in visited:
                        new_details = path_details + [(current, a, output, next_state)]
                        new_inputs = input_sequence + [a]
                        queue.append((next_state, new_details, new_inputs))

        self.current_state = original_state
        return ([], [])
    
    def add_state(self, state, is_accepting=False):
        """安全添加新状态，并初始化转移函数"""
        if state not in self.states:
            self.states.add(state)
            self.transition_func[state] = defaultdict(list)  # 初始化空转移
            if is_accepting:
                self.accepting_states.add(state)
        else:
            raise ValueError(f"State {state} already exists")

    def generate_random_path(self, n):
        path_details = []
        input_sequence = []
        current_state = self.current_state
    
        for _ in range(n):
            available_inputs = list(self.transition_func[current_state].keys())
            if not available_inputs:
                break
        
            input_symbol = random.choice(available_inputs)
            output, next_state = random.choice(self.transition_func[current_state][input_symbol])
        
            path_details.append((current_state, input_symbol, output, next_state))
            input_sequence.append(input_symbol)
            current_state = next_state
        return (path_details, input_sequence)
    
    def print_automaton_info(self):
        print("States:", self.states)
        print("Inputs:", self.inputs)
        print("Outputs:", self.outputs)
        print("Initial state:", self.initial_state)
        print("Accepting state:", self.accepting_states)
        print("Transition Function:")
        for state in self.transition_func:
            for input_symbol in self.transition_func[state]:
                transitions = self.transition_func[state][input_symbol]
                print(f"  State: {state}, Input: {input_symbol} -> Transitions: {transitions}")
    
    def is_sequence_accepted(self, input_sequence):
        # 使用BFS来跟踪所有可能的路径
        current_states = {(self.initial_state, tuple())}  # 使用元组存储路径
    
        for input_symbol in input_sequence:
            if input_symbol not in self.inputs:
                return False, []  # 无效输入符号
        
            next_states = set()
            for state, path in current_states:
                transitions = self.transition_func[state].get(input_symbol, [])
                for output, next_state in transitions:
                    # 将路径转换为元组后再存储
                    new_path = path + ((state, input_symbol, output, next_state),)
                    next_states.add((next_state, new_path))
        
            if not next_states:
                return False, []  # 没有可能的转移
            current_states = next_states
    
        # 检查是否有任何路径到达接受状态
        for state, path in current_states:
            if self.is_accepting(state):
                return True, list(path)  # 返回时转换回列表
    
        return False, []

def build_subsequence_mealy_empty_output(sigma, sequence):
    num_states = len(sequence) + 1
    states = {f'q{i}' for i in range(num_states)}
    transition_func = defaultdict(lambda: defaultdict(list))
    
    for i in range(num_states):
        current_state = f'q{i}'
        if i == len(sequence):
            for symbol in sigma:
                transition_func[current_state][symbol].append(('', current_state))
            continue
        
        target = sequence[i]
        for symbol in sigma:
            transition_func[current_state][symbol].append(('', current_state))
            if symbol == target:
                next_state = f'q{i+1}'
                transition_func[current_state][symbol].append(('', next_state))
    
    return MealyNFA(
        states=states,
        inputs=sigma,
        outputs={''},
        transition_func=transition_func,
        initial_state='q0',
        accepting_states={f'q{len(sequence)}'}
    )

def multiply(mealy1, mealy2):
    initial_pair = (mealy1.initial_state, mealy2.initial_state)
    product_states = set()
    product_transition_func = defaultdict(lambda: defaultdict(list))
    visited = set()
    queue = deque([initial_pair])
    product_states.add(initial_pair)
    visited.add(initial_pair)

    product_inputs = mealy1.inputs.union(mealy2.inputs)

    while queue:
        s1, s2 = queue.popleft()

        for a in product_inputs:
            # 修改处：未定义输入时返回空列表，不添加默认转移
            trans1 = mealy1.transition_func[s1].get(a, [])  # 移除默认值
            trans2 = mealy2.transition_func[s2].get(a, [])  # 移除默认值
            
            # 仅处理实际存在的转移
            for (o1, ns1) in trans1:
                for (o2, ns2) in trans2:
                    next_pair = (ns1, ns2)
                    product_transition_func[(s1, s2)][a].append(((o1, o2), next_pair))
                    if next_pair not in visited:
                        visited.add(next_pair)
                        product_states.add(next_pair)
                        queue.append(next_pair)

    product_accepting = {(s1, s2) for s1, s2 in product_states 
                        if mealy1.is_accepting(s1) and mealy2.is_accepting(s2)}

    return MealyNFA(
        states=product_states,
        inputs=product_inputs,
        outputs={(o1, o2) for o1 in mealy1.outputs for o2 in mealy2.outputs},
        transition_func=product_transition_func,
        initial_state=initial_pair,
        accepting_states=product_accepting
    )

def have_intersection(mealy1, mealy2):
    start = (mealy1.initial_state, mealy2.initial_state)
    visited = set([start])
    queue = deque([start])

    while queue:
        s1, s2 = queue.popleft()
        if mealy1.is_accepting(s1) and mealy2.is_accepting(s2):
            return True

        inputs = mealy1.inputs.intersection(mealy2.inputs)  # 仅检查共同输入
        for a in inputs:
            trans1 = mealy1.transition_func[s1].get(a, [])  # 移除默认转移
            trans2 = mealy2.transition_func[s2].get(a, [])  # 移除默认转移
            
            for (_, ns1) in trans1:
                for (_, ns2) in trans2:
                    next_pair = (ns1, ns2)
                    if next_pair not in visited:
                        visited.add(next_pair)
                        queue.append(next_pair)
    return False

def is_subsequence(p, sequence):
    """检查排列p是否是输入序列sequence的子序列"""
    it = iter(sequence)
    return all(elem in it for elem in p)

class TestGenerator:
    def __init__(self, events, fsm, strength, max_tuples):
        self.I = events
        self.F = fsm
        self.t = strength
        self.N = max_tuples
        self.TS = []
    
    def generate_t_perms(self):
        return list(permutations(self.I, self.t))
    
    def generate_test_suite(self):
        T = self.generate_t_perms()
        TS = []
        
        while T:
            current_constraints = []
            active_fsm = self.F  # 使用临时变量保持原始FSM不变
            
            # 收集最多N个可行约束
            while len(current_constraints) < self.N and T:
                idx = random.randint(0, len(T)-1)
                p = T.pop(idx)
                a = build_subsequence_mealy_empty_output(self.I, p)
                
                if have_intersection(a, active_fsm):
                    current_constraints.append(a)
                else:
                    # 不可行约束直接丢弃
                    continue

            if current_constraints:
                # 构建组合约束
                combined = active_fsm
                for constraint in current_constraints:
                    combined = multiply(combined, constraint)
                
                # 生成测试用例
                try:
                    _, test_case = combined.generate_accepting_path()
                    TS.append(test_case)
                    
                    # 分析test_case覆盖的排列，并从T中移除
                    covered = []
                    # 遍历T的副本以避免修改问题
                    for p in list(T):
                        if is_subsequence(p, test_case):
                            covered.append(p)
                    # 从原T中移除所有覆盖的排列
                    for p in covered:
                        try:
                            T.remove(p)
                        except ValueError:
                            pass  # 如果p已经被移除，则忽略
                except:
                    pass

        return TS


def generate_vault_automaton(n):
    """
    生成一个vault自动机，密码长度为n，每个数字是0-9的自然数
    
    参数:
        n (int): 密码长度
        
    返回:
        MealyNFA: 生成的自动机
        str: 随机生成的密码（字符串形式）
    """
    if n < 1:
        raise ValueError("Password length must be at least 1")
    
    # 1. 随机生成一个n位密码（每个数字0-9）
    password = ''.join(str(random.randint(0, 9)) for _ in range(n))
    
    # 2. 构建自动机状态（S0, S1, ..., S{n}, S_accept, S_error）
    states = {f'S{i}' for i in range(n + 1)} | {'S_accept', 'S_error'}
    
    # 3. 输入是0-9的数字 + 'try_open'（尝试打开金库）
    inputs = {str(i) for i in range(10)} | {'try_open'}
    
    # 4. 输出是'wait'（等待更多输入）、'reject'（密码错误）、'accept'（密码正确）
    outputs = {'wait', 'reject', 'accept'}
    
    # 5. 构建转移函数
    transition_func = defaultdict(lambda: defaultdict(list))
    
    # 初始状态S0的转移
    for digit in range(10):
        if digit == int(password[0]):
            transition_func['S0'][str(digit)].append(('wait', 'S1'))
        else:
            transition_func['S0'][str(digit)].append(('wait', 'S_error'))
    
    # 中间状态S1到S{n-1}的转移
    for i in range(1, n):
        current_state = f'S{i}'
        correct_digit = int(password[i])
        
        for digit in range(10):
            if digit == correct_digit:
                next_state = f'S{i + 1}'
                transition_func[current_state][str(digit)].append(('wait', next_state))
            else:
                transition_func[current_state][str(digit)].append(('wait', 'S_error'))
    
    # 最后一个状态S{n}的转移（输入'try_open'才能决定是否接受）
    transition_func[f'S{n}']['try_open'] = [('accept', 'S_accept')]
    
    # 错误状态S_error的转移（任何输入都拒绝）
    for digit in range(10):
        transition_func['S_error'][str(digit)].append(('reject', 'S_error'))
    transition_func['S_error']['try_open'] = [('reject', 'S_error')]
    
    return MealyNFA(
        states=states,
        inputs=inputs,
        outputs=outputs,
        transition_func=transition_func,
        initial_state='S0',
        accepting_states={'S_accept'}
    ), password


def main():
    # 定义参数组合
    password_lengths = [5,10, 15, 20, 30]
    strengths = [2, 3, 4]
    max_tuples_list = [3, 5, 8]
    vault, password = generate_vault_automaton(5)
    events = vault.inputs
    generator = TestGenerator(events, vault, strength=2, max_tuples=3)
    test_suite = generator.generate_test_suite()
    print("Generated test suite:", test_suite)
    print("Generated password:", password)
    # with open('vault_result.txt', 'w') as f:
    #     f.write("密码长度(M)\t强度(t)\t最大元组数(N)\t平均时间(ms)\t最大长度\t最小长度\t测试用例数\n")
        
    #     for M in password_lengths:
    #         # 为每个密码长度生成一次自动机
    #         vault, password = generate_vault_automaton(M)
    #         events = vault.inputs
            
    #         for t in strengths:
    #             for N in max_tuples_list:
    #                 print(f"正在测试: M={M}, t={t}, N={N}")
                    
    #                 execution_times = []
    #                 all_test_lengths = []
                    
    #                 # 运行100次取平均值
    #                 for _ in range(1):
    #                     generator = TestGenerator(events, vault, strength=t, max_tuples=N)
                        
    #                     start_time = time.perf_counter()
    #                     test_suite = generator.generate_test_suite()
    #                     end_time = time.perf_counter()
                        
    #                     execution_times.append((end_time - start_time) * 1000)
                        
    #                     if test_suite:
    #                         lengths = [len(case) for case in test_suite]
    #                         all_test_lengths.extend(lengths)
                    
    #                 # 计算统计指标
    #                 avg_time = mean(execution_times) if execution_times else 0
    #                 max_length = max(all_test_lengths) if all_test_lengths else 0
    #                 min_length = min(all_test_lengths) if all_test_lengths else 0
    #                 test_count = len(all_test_lengths)
                    
    #                 # 写入结果文件
    #                 f.write(f"{M}\t{t}\t{N}\t{avg_time:.2f}\t{max_length}\t{min_length}\t{test_count}\n")
    #                 f.flush()  # 确保每次写入后立即保存

if __name__ == "__main__":
    main()