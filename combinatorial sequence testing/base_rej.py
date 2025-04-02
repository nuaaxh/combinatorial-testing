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

        inputs = mealy1.inputs.union(mealy2.inputs)
        for a in inputs:
            trans1 = mealy1.transition_func[s1].get(a, [('', s1)])
            trans2 = mealy2.transition_func[s2].get(a, [('', s2)])
            
            for (_, ns1) in trans1:
                for (_, ns2) in trans2:
                    next_pair = (ns1, ns2)
                    if next_pair not in visited:
                        visited.add(next_pair)
                        queue.append(next_pair)
                        if mealy1.is_accepting(ns1) and mealy2.is_accepting(ns2):
                            return True
    return False

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
        attempt_count = 0
        MAX_ATTEMPTS = 100  # 防止无限循环
        
        while T and attempt_count < MAX_ATTEMPTS:
            current_constraints = []
            temp_T = T.copy()
            
            # 阶段1：收集N个约束
            while len(current_constraints) < self.N and temp_T:
                idx = random.randint(0, len(temp_T)-1)
                p = temp_T.pop(idx)
                a = build_subsequence_mealy_empty_output(self.I, p)
                current_constraints.append( (p, a) )
            
            # 阶段2：生成测试用例
            try:
                if not current_constraints:
                    continue
                # 构建仅约束自动机的乘积
                combined = current_constraints[0][1]
                for _, constraint in current_constraints[1:]:
                    combined = multiply(combined, constraint)
                
                # 生成路径并检查是否非空
                path_info, test_case = combined.generate_accepting_path()
                if not test_case:  # 无有效路径时跳过
                    raise ValueError("No accepting path found")
                
                # 阶段3：独立验证是否被主自动机接受
                is_accepted, _ = self.F.is_sequence_accepted(test_case)
                if is_accepted:
                    TS.append(test_case)
                    # 仅移除当前处理成功的约束
                    for p, _ in current_constraints:
                        if p in T:
                            T.remove(p)
                    attempt_count = 0  # 重置尝试计数
                else:
                    attempt_count += 1  # 仅增加尝试次数
                
            except Exception as e:
                attempt_count += 1
        
        return TS




def create_constraintFSM():
    transition_func = {
        'S0': defaultdict(list, {
            # 'eb': [('stop', 'S0')],
            # 'sc_m': [('stop', 'S0')],
            'sc_a': [('low', 'S1')]
        }),
        'S1': defaultdict(list, {
            'eb': [('stop', 'S0')],
            'sc_d': [('stop', 'S0')],
            'sc_a': [('fast', 'S2')],
            'sc_m': [('low', 'S1')]
        }),
        'S2': defaultdict(list, {
            'eb': [('stop', 'S0')],
            'sc_d': [('low', 'S1')],
            'sc_m': [('fast', 'S2')]
        })
    }
    return MealyNFA(
        states={'S0', 'S1', 'S2'},
        inputs={'sc_d', 'sc_a', 'sc_m','eb'},
        outputs={'low', 'stop','fast'},
        transition_func=transition_func,
        initial_state='S0',
        accepting_states={'S0'}
    )

def create_modified_constraintFSM():
    transition_func = {
        'S0': defaultdict(list, {
            'eb': [('stop', 'S0')],
            'sc_m': [('stop', 'S0')],
            'sc_a': [('low', 'S1')]
        }),
        'S1': defaultdict(list, {
            'eb': [('stop', 'S0')],
            'sc_d': [('stop', 'S0')],
            'sc_a': [('fast', 'S2')],
            'sc_m': [('low', 'S1')],
            'A_dz': [('h2_I', 'S5')]
        }),
        'S2': defaultdict(list, {
            'eb': [('stop', 'S0')],
            'sc_d': [('low', 'S1')],
            'sc_m': [('fast', 'S2')],
            'sc_a': [('overspeed,h1_I', 'S3')],
            'A_dz': [('h2_I', 'S4')]
        }),
        'S3': defaultdict(list, {
            'sc_d': [('fast,h1_F', 'S2')],
            'eb': [('stop,h1_F', 'S0')]
        }),
        'S4': defaultdict(list, {
            'sc_d': [('h2_F,low', 'S1')]
        }),
        'S5': defaultdict(list, {
            'sc_m': [('h2_F,low', 'S1')]
        }),
    }
    return MealyNFA(
        states={'S0', 'S1', 'S2', 'S3', 'S4', 'S5'},
        inputs={'sc_d', 'sc_a', 'sc_m','eb', 'A_dz'},
        outputs={'low', 'stop','fast', 'overspeed,h1_I', 'fast,h1_F', 'stop,h1_F', 'h2_F,low','h2_I', 'h2_F'},
        transition_func=transition_func,
        initial_state='S0',
        accepting_states={'S0'}
    )

if __name__ == "__main__":
    events = {'sc_d', 'sc_a', 'sc_m', 'eb'}
    fsmA = create_constraintFSM()
    fsmB = create_modified_constraintFSM()
    # generator = TestGenerator(events, fsmB, strength=2, max_tuples=3)
    # test_suite = generator.generate_test_suite()
    # print("Generated Test Suite:")
    # for case in test_suite:
    #     print(case)
    # 性能统计变量
    execution_times = []
    all_test_lengths = []
    
    # 进行100次测试生成
    for _ in range(100):
        # 每次创建新的测试生成器
        generator = TestGenerator(events, fsmB, strength=2, max_tuples=10)
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 生成测试套件
        test_suite = generator.generate_test_suite()
        
        # 记录结束时间
        end_time = time.perf_counter()
        
        # 保存执行时间（毫秒）
        execution_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 收集测试序列长度
        if test_suite:
            lengths = [len(case) for case in test_suite]
            all_test_lengths.extend(lengths)
    
    # 计算统计指标
    avg_time = mean(execution_times) if execution_times else 0
    max_length = max(all_test_lengths) if all_test_lengths else 0
    min_length = min(all_test_lengths) if all_test_lengths else 0
    
    # 输出结果
    print("\n=== 性能统计结果 ===")
    print(f"平均运行时间: {avg_time:.2f} ms")
    print(f"最大测试序列长度: {max_length}")
    print(f"最小测试序列长度: {min_length}")
    print(f"总生成测试用例数: {len(all_test_lengths)}")
