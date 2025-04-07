import random
from itertools import permutations
from collections import defaultdict, deque
from typing import Sequence
from itertools import product
from hashlib import sha256
import time
from statistics import mean
from collections import OrderedDict
import copy

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
        self.TS_detail = []
    
    def generate_t_perms(self):
        return list(permutations(self.I, self.t))
    
    def extract_fsm_state(nested_state):
        """递归提取嵌套状态中的FSM状态"""
        if isinstance(nested_state, tuple):
            return extract_fsm_state(nested_state[0])  # 递归解包第一元素
        elif isinstance(nested_state, str) and nested_state.startswith('S'):  # 判断是否为FSM状态
            return nested_state
        else:
            return None  # 非FSM状态

    def generate_test_suite(self):
        T = self.generate_t_perms()
        self.TS = []  # 重置测试套件
        self.TS_detail = []  # 重置详细路径
        
        while T:
            current_constraints = []
            
            # 收集约束条件
            while len(current_constraints) < self.N and T:
                idx = random.randint(0, len(T)-1)
                p = T.pop(idx)
                a = build_subsequence_mealy_empty_output(self.I, p)
                
                if have_intersection(a, self.F):
                    current_constraints.append(a)
            
            if not current_constraints:
                continue
                
            # 构建组合自动机
            combined = self.F
            for constraint in current_constraints:
                combined = multiply(combined, constraint)
            
            try:
                path_details, test_case = combined.generate_accepting_path()
                if test_case:  # 确保生成了有效的测试用例
                    self.TS.append(test_case)
                    
                    # 处理详细序列
                    detail_sequence = []
                    for step in path_details:
                        # 提取FSM状态
                        state = step[0]
                        while isinstance(state, tuple):
                            state = state[0]
                        
                        if isinstance(state, str) and state.startswith('S'):
                            detail_sequence.extend([state, step[1]])
                    
                    # 添加最终状态
                    if path_details:
                        final_state = path_details[-1][3]
                        while isinstance(final_state, tuple):
                            final_state = final_state[0]
                        if isinstance(final_state, str) and final_state.startswith('S'):
                            detail_sequence.append(final_state)
                    
                    self.TS_detail.append(detail_sequence)
                    
                    # 移除已覆盖的排列
                    T = [p for p in T if not is_subsequence(p, test_case)]
                
            except Exception as e:
                print(f"生成路径时出错: {e}")
                continue

        return self.TS, self.TS_detail

class DeltaUpdate:
    def __init__(self, start_state, transitions, end_state):
        self.start_state = start_state
        self.transitions = transitions  # 每个transition是一个迁移路径的步骤列表，每个步骤包含输入、输出、下一状态
        self.end_state = end_state


class IncrementalTestGenerator(TestGenerator):
    def __init__(self, original_generator, delta, new_fsm, initial_fsm=None):
        
        super().__init__(original_generator.I, new_fsm, original_generator.t, original_generator.N)
        self.original_TS = original_generator.TS
        self.original_TS_detail = original_generator.TS_detail if hasattr(original_generator, 'TS_detail') else []
        self.delta = delta
        self.remaining_perms = set()  # 初始化为空集合
        self.TS_detail = []  # 保存详细的测试用例路径
        # 保存最初未更新的 FSM（如果未提供，默认使用 original_generator.F 的副本）
        self.initial_fsm = copy.deepcopy(initial_fsm) if initial_fsm else copy.deepcopy(original_generator.F)
    def generate_incremental_suite(self):
        new_TS = []
        new_TS_detail = []
        # 初始化为空集合（假设原始测试套件已100%覆盖）
        self.remaining_perms = set()  
        
        for tc_input, tc_detail in zip(self.original_TS, self.original_TS_detail):
            original_covered = self.get_covered_permutations(tc_input)
            replaced = False

            for start_idx, end_idx in self.find_replacements(tc_detail):
                new_input, detail_fragment = self.replace_subsequence(tc_input, start_idx, end_idx)
                accepted, _ = self.F.is_sequence_accepted(new_input)
                
                if accepted:
                    new_covered = self.get_covered_permutations(new_input)
                    lost_coverage = original_covered - new_covered
                    
                    # 将丢失的组合添加到remaining_perms
                    self.remaining_perms.update(lost_coverage)
                    
                    # 构建新测试用例（保持原逻辑）
                    new_detail = (tc_detail[:start_idx*2] + 
                                detail_fragment + 
                                tc_detail[end_idx*2+1:])
                    
                    new_TS.append(new_input)
                    new_TS_detail.append(new_detail)
                    replaced = True
                    break

            if not replaced:
                new_TS.append(tc_input)
                new_TS_detail.append(tc_detail)

        # 生成补充测试用例（只针对新增的未覆盖组合）
        if self.remaining_perms:
            supplementary_TS, supplementary_detail = self.generate_remaining_suite(list(self.remaining_perms))
            new_TS.extend(supplementary_TS)
            new_TS_detail.extend(supplementary_detail)

        self.TS = new_TS
        self.TS_detail = new_TS_detail
        return new_TS

    def find_replacements(self, detail_sequence):
        replacements = []
        states = []
        inputs = []

        # Parse detail sequence into states and inputs
        for i, elem in enumerate(detail_sequence):
            if i % 2 == 0 or i == len(detail_sequence) - 1:
                states.append(elem)
            else:
                inputs.append(elem)

        # Find all valid replacement windows
        for i in range(len(states)):
            if states[i] != self.delta.start_state:
                continue
            for j in range(i+1, len(states)):
                if states[j] == self.delta.end_state:
                    replacements.append((i, j))

        return replacements

    def replace_subsequence(self, original_input, start_idx, end_idx):
        # 随机选择一个迁移路径
        selected_transition = random.choice(self.delta.transitions)
        # 提取输入序列并生成详细路径片段
        input_sequence = [step[0] for step in selected_transition]
        detail_fragment = []
        current_state = self.delta.start_state
        for step in selected_transition:
            input_symbol, output, next_state = step
            detail_fragment.append(current_state)
            detail_fragment.append(input_symbol)
            current_state = next_state
        # 添加最终状态
        detail_fragment.append(current_state)
        # 替换原输入中的部分
        new_input = original_input[:start_idx] + input_sequence + original_input[end_idx:]
        return new_input, detail_fragment

    def get_covered_permutations(self, input_sequence):
        from itertools import combinations
        covered = set()
        n = len(input_sequence)
        for indices in combinations(range(n), self.t):
            perm = tuple(input_sequence[i] for i in indices)
            covered.add(perm)
        return covered

    def generate_remaining_suite(self, remaining_perms):
        """基于最初未更新的 FSM 生成剩余测试用例"""
        T = remaining_perms.copy()
        TS = []
        TS_detail = []
        
        while T:
            current_constraints = []
            while len(current_constraints) < self.N and T:
                idx = random.randint(0, len(T)-1)
                p = T.pop(idx)
                a = build_subsequence_mealy_empty_output(self.I, p)
                # 使用最初未更新的 FSM 检查交集
                if have_intersection(a, self.initial_fsm):  # 关键修改：使用 self.initial_fsm
                    current_constraints.append(a)

            if not current_constraints:
                continue

            # 使用最初未更新的 FSM 进行组合
            combined = copy.deepcopy(self.initial_fsm)  # 关键修改：使用 self.initial_fsm
            for constraint in current_constraints:
                combined = multiply(combined, constraint)

            path_details, test_case = combined.generate_accepting_path()
            if test_case:
                # 生成详细的路径序列
                detail_sequence = []
                for step in path_details:
                    state = step[0]
                    while isinstance(state, tuple):
                        state = state[0]
                    detail_sequence.extend([state, step[1]])
                if path_details:
                    final_state = path_details[-1][3]
                    while isinstance(final_state, tuple):
                        final_state = final_state[0]
                    detail_sequence.append(final_state)
                TS.append(test_case)
                TS_detail.append(detail_sequence)
                # 移除已覆盖的排列
                T = [p for p in T if not is_subsequence(p, test_case)]

        return TS, TS_detail

def apply_delta_updates(original_fsm, delta):
    new_fsm = copy.deepcopy(original_fsm)
   
    start_state = delta.start_state
    for path in delta.transitions:
        current_state = start_state
        for step in path:
            input_symbol, output, next_state = step
            # 确保输入符号被添加到输入集合
            new_fsm.inputs.add(input_symbol)
            # 确保当前状态存在
            new_fsm.add_state(current_state)
            # 确保下一状态存在
            new_fsm.add_state(next_state)
            # 添加迁移到transition_func
            new_fsm.transition_func[current_state][input_symbol].append((output, next_state))
            # 添加输出符号到outputs集合
            new_fsm.outputs.add(output)
            current_state = next_state  # 更新当前状态为下一步的起点
    return new_fsm

def create_constraintFSM():
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

# def run_single_experiment(N_value):
#     original_fsm = create_constraintFSM()
#     test_gen = TestGenerator(
#         events={'sc_d', 'sc_a', 'sc_m','eb'}, 
#         fsm=original_fsm,
#         strength=2,
#         max_tuples=N_value
#     )
#     # 生成原始测试套件
#     test_gen.generate_test_suite()
#     # print(test_gen.TS)
#     # print(test_gen.TS_detail)
#     # 定义四个增量更新
#     delta1 = DeltaUpdate(
#         start_state='S1',
#         transitions=[
#             [('A_dz', 'h2_I', 'S5'), ('sc_m', 'h2_F,low', 'S1')]
#         ],
#         end_state='S1'
#     )
#     delta2 = DeltaUpdate(
#         start_state='S2',
#         transitions=[
#             [('sc_a', 'overspeed,h1_I', 'S3'), ('sc_d', 'low', 'S2')]
#         ],
#         end_state='S2'
#     )
#     delta3 = DeltaUpdate(
#         start_state='S2',
#         transitions=[
#             [('sc_a', 'overspeed,h1_I', 'S3'), ('eb','stop,h1_F', 'S0')]
#         ],
#         end_state='S0'
#     )
#     delta4 = DeltaUpdate(
#         start_state='S2',
#         transitions=[
#             [('A_dz', 'h2_I', 'S4'), ('sc_d','h2_F,low', 'S1')]
#         ],
#         end_state='S1'
#     )

#     deltas = [delta1, delta2, delta3, delta4]
#     current_fsm = original_fsm
#     current_test_gen = test_gen
#     incremental_times = []

#     for delta in deltas:
#         updated_fsm = apply_delta_updates(current_fsm, delta)
#         inc_test_gen = IncrementalTestGenerator(current_test_gen, delta, updated_fsm)
#         start_time = time.time()
#         inc_test_gen.generate_incremental_suite()
#         end_time = time.time()
#         incremental_times.append(end_time - start_time)
#         current_fsm = updated_fsm
#         current_test_gen = inc_test_gen
#         # print(inc_test_gen.TS)
#         # print(inc_test_gen.TS_detail)

#     final_ts_size = len(current_test_gen.TS)
#     test_case_lengths = [len(tc) for tc in current_test_gen.TS]

#     return incremental_times, final_ts_size, test_case_lengths

# def main():
#     N_values = [3,5,8,10]
#     results = {}

#     for N in N_values:
#         total_times = []
#         suite_sizes = []
#         all_lengths = []
        
#         for _ in range(10):
#             incremental_times, size, lengths = run_single_experiment(N)
#             total_time = sum(incremental_times)
#             total_times.append(total_time)
#             suite_sizes.append(size)
#             all_lengths.extend(lengths)
        
#         avg_time = mean(total_times)
#         avg_size = mean(suite_sizes)
#         max_len = max(all_lengths) if all_lengths else 0
#         min_len = min(all_lengths) if all_lengths else 0
        
#         results[N] = {
#             'avg_time': avg_time,
#             'avg_size': avg_size,
#             'max_len': max_len,
#             'min_len': min_len
#         }

#     # 输出结果
#     for N in N_values:
#         res = results[N]
#         print(f"N = {N}:")
#         print(f"  Average time for four updates: {res['avg_time']:.4f} sec")
#         print(f"  Average test suite size: {res['avg_size']:.2f}")
#         print(f"  Max test length: {res['max_len']}")
#         print(f"  Min test length: {res['min_len']}")
#         print()

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    original_fsm = create_constraintFSM()
    test_gen = TestGenerator(events={'sc_d', 'sc_a', 'sc_m','eb'}, 
                        fsm=original_fsm,
                        strength=2,
                        max_tuples=10)
    start  = time.time()
    original_TS, original_TS_detail = test_gen.generate_test_suite()
    end = time.time()
    print(f"Original Test Generation Time: {end - start:.4f} seconds")
    
    # 定义四个增量更新
    delta1 = DeltaUpdate(
        start_state='S1',
        transitions=[
            [('A_dz', 'h2_I', 'S5'), ('sc_m', 'h2_F,low', 'S1')]
        ],
        end_state='S1'
    )
    delta2 = DeltaUpdate(
        start_state='S2',
        transitions=[
            [('sc_a', 'overspeed,h1_I', 'S3'), ('sc_d', 'low', 'S2')]
        ],
        end_state='S2'
    )
    delta3 = DeltaUpdate(
        start_state='S2',
        transitions=[
            [('sc_a', 'overspeed,h1_I', 'S3'), ('eb','stop,h1_F', 'S0')]
        ],
        end_state='S0'
    )
    delta4 = DeltaUpdate(
        start_state='S2',
        transitions=[
            [('A_dz', 'h2_I', 'S4'), ('sc_d','h2_F,low', 'S1')]
        ],
        end_state='S1'
    )

    # 应用四个增量更新并记录时间
    deltas = [delta1, delta2, delta3, delta4]
    current_fsm = original_fsm
    current_test_gen = test_gen
    incremental_times = []
    
    for delta in deltas:
        # 应用当前delta更新到FSM
        updated_fsm = apply_delta_updates(current_fsm, delta)
        # 创建增量测试生成器
        inc_test_gen = IncrementalTestGenerator(current_test_gen, delta, updated_fsm, initial_fsm=original_fsm)
        # 生成增量测试套件并计时
        start_time = time.time()
        new_TS = inc_test_gen.generate_incremental_suite()
        end_time = time.time()
        delta_time = end_time - start_time
        incremental_times.append(delta_time)
        print(f"Delta update applied in {delta_time:.4f} seconds")
        # 更新当前FSM和测试生成器
        current_fsm = updated_fsm
        current_test_gen = inc_test_gen
    
    # 计算平均执行时间
    average_time = sum(incremental_times) / len(incremental_times)
    print(f"Average execution time for 4 incremental updates: {average_time:.4f} seconds")
    
    print("Original Test Suite:")
    print(original_TS)
    print("New Test Suite after all Incremental Updates:")
    print(current_test_gen.TS)
    print("New Test Suite Details:") 
    print(current_test_gen.TS_detail)