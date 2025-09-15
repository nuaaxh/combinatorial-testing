import random
from itertools import permutations
from collections import defaultdict, deque
import time
from statistics import mean
import copy
from typing import List, Tuple, Set
import matplotlib.pyplot as plt
import random
from itertools import permutations
from collections import defaultdict, deque
import time
import multiprocessing
import gc
import random
import time
from collections import defaultdict
from statistics import mean
import copy

class MealyNFA:
    def __init__(self, states, inputs, outputs, transition_func, initial_state, accepting_states=None):
        self.states = states
        self.inputs = set(inputs)  # <- 保证是集合
        self.outputs = set(outputs)
        self.transition_func = transition_func
        self.initial_state = initial_state
        self.current_state = initial_state
        self.accepting_states = set(accepting_states) if accepting_states else set()

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
                if not transitions:
                    continue  # 避免空列表
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
            available_inputs = [i for i in self.transition_func[current_state]
                                if self.transition_func[current_state][i]]
            if not available_inputs:
                break  # 当前状态没有可用输入，提前退出

            input_symbol = random.choice(available_inputs)
            transitions = self.transition_func[current_state][input_symbol]
            if not transitions:
                break  # 防止空列表
            output, next_state = random.choice(transitions)

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

    @staticmethod
    def trim_to_accepting(path_details, accepting_states):
        """
        裁剪随机路径，使其最终状态为可接受状态
        保留到最后一个可接受状态的位置，其后的输入全部删除
        """
        last_accept_idx = -1
        for idx, (_, _, _, next_state) in enumerate(path_details):
            if next_state in accepting_states:
                last_accept_idx = idx

        if last_accept_idx == -1:
            return [], []  # 整条路径未到达接受状态

        trimmed_path = path_details[:last_accept_idx + 1]
        trimmed_input = [step[1] for step in trimmed_path]

        return trimmed_path, trimmed_input

    def generate_random_tests(self, num_tests=100):
        """在FSM上随机生成若干条测试用例，裁剪至最后可接受状态"""
        tests = []
        for _ in range(num_tests):
            path_details, input_sequence = self.F.generate_random_path(self.N)
            path_details, input_sequence = self.trim_to_accepting(path_details, self.F.accepting_states)
            if input_sequence:  # 保留有效路径
                tests.append((path_details, input_sequence))
        return tests

    def get_covered_permutations(self, input_sequence):
        """计算一个测试用例能覆盖的t-perms"""
        from itertools import combinations
        covered = set()
        n = len(input_sequence)
        for indices in combinations(range(n), self.t):
            perm = tuple(input_sequence[i] for i in indices)
            covered.add(perm)
        return covered

    def generate_test_suite(self):
        T = set(self.generate_t_perms())  # 所有 t-perms
        covered = set()
        self.TS = []
        self.TS_detail = []

        # 1️⃣ 随机生成候选测试用例（数量可以大一些保证覆盖性）
        candidate_tests = self.generate_random_tests(num_tests=10 * len(T))

        # 将候选按增量覆盖数量贪心排序
        while T - covered and candidate_tests:
            # 对每个候选测试，计算它能覆盖的新排列数
            best_test = max(candidate_tests,
                            key=lambda tc: len(self.get_covered_permutations(tc[1]) - covered))
            candidate_tests.remove(best_test)

            new_cov = self.get_covered_permutations(best_test[1]) - covered
            if not new_cov:
                continue  # 这个候选没有增量覆盖，跳过

            # 添加到最终测试套件
            path_details, test_case = best_test
            self.TS.append(test_case)

            # 构建详细路径
            detail_sequence = []
            for step in path_details:
                state = step[0]
                while isinstance(state, tuple):
                    state = state[0]
                if isinstance(state, str) and state.startswith('S'):
                    detail_sequence.extend([state, step[1]])
            if path_details:
                final_state = path_details[-1][3]
                while isinstance(final_state, tuple):
                    final_state = final_state[0]
                if isinstance(final_state, str) and final_state.startswith('S'):
                    detail_sequence.append(final_state)
            self.TS_detail.append(detail_sequence)

            # 更新已覆盖集合
            covered |= new_cov

        # 2️⃣ 兜底生成剩余未覆盖排列（如随机生成未覆盖时）
        remaining_perms = T - covered
        while remaining_perms:
            # 这里可以使用原来的组合 FSM 方法兜底
            current_constraints = []
            while len(current_constraints) < self.N and remaining_perms:
                idx = random.randint(0, len(remaining_perms) - 1)
                p = list(remaining_perms)[idx]
                remaining_perms.remove(p)
                a = build_subsequence_mealy_empty_output(self.I, p)
                if have_intersection(a, self.F):
                    current_constraints.append(a)
            if not current_constraints:
                continue

            combined = self.F
            for constraint in current_constraints:
                combined = multiply(combined, constraint)

            try:
                path_details, test_case = combined.generate_accepting_path()
                if test_case:
                    self.TS.append(test_case)
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
                    self.TS_detail.append(detail_sequence)

                    covered |= self.get_covered_permutations(test_case)
                    remaining_perms = T - covered
            except Exception as e:
                print(f"兜底生成路径出错: {e}")
                continue

        return self.TS, self.TS_detail


class DeltaUpdate:
    def __init__(self, start_state, transitions, end_state):
        self.start_state = start_state
        self.transitions = transitions  # 每个transition是一个迁移路径的步骤列表，每个步骤包含输入、输出、下一状态
        self.end_state = end_state


class IncrementalTestGenerator(TestGenerator):
    def __init__(self, original_generator, delta, new_fsm, initial_fsm=None, max_replacements=1):
        # 使用完全深拷贝的事件集和 FSM
        merged_events = set(original_generator.I) | set(new_fsm.inputs)
        super().__init__(merged_events, copy.deepcopy(new_fsm), original_generator.t, original_generator.N)

        # 深拷贝原始测试套件，避免引用累积
        self.original_TS = copy.deepcopy(original_generator.TS)
        self.original_TS_detail = copy.deepcopy(getattr(original_generator, 'TS_detail', []))

        self.delta = delta
        self.F = copy.deepcopy(new_fsm)  # 使用深拷贝保证不共享状态
        self.initial_fsm = copy.deepcopy(initial_fsm) if initial_fsm else copy.deepcopy(original_generator.F)
        self.TS_detail = []

        self.original_events = set(original_generator.I)
        self.max_replacements = max_replacements

    # ------------------------
    # 启发式生成方法
    # ------------------------
    def heuristic_completion(self, remaining_perms, num_tests=200):
        """
        启发式补全未覆盖排列，随机生成序列并裁剪到接受状态
        仅保留能带来新增覆盖的测试用例
        """
        completed_TS = []
        completed_detail = []
        covered = set()

        # 随机生成候选测试
        candidates = []
        for _ in range(num_tests):
            path_details, test_case = self.F.generate_random_path(self.N)
            path_details, test_case = self.trim_to_accepting(path_details, self.F.accepting_states)
            if test_case:
                candidates.append((path_details, test_case))

        # 贪心选择覆盖最多 t-perms 的测试用例
        while candidates and remaining_perms - covered:
            # 按新增覆盖数量排序
            best_test = max(candidates, key=lambda x: len(self.get_covered_permutations(x[1]) - covered))
            candidates.remove(best_test)

            new_cov = self.get_covered_permutations(best_test[1]) - covered
            if not new_cov:
                continue  # 跳过没有新增覆盖的测试用例

            # 添加测试用例
            path_details, test_case = best_test
            completed_TS.append(test_case)
            covered |= new_cov

            # 构建详细序列
            detail_sequence = []
            for step in path_details:
                state = step[0]
                while isinstance(state, tuple):
                    state = state[0]
                if isinstance(state, str) and state.startswith('S'):
                    detail_sequence.extend([state, step[1]])
            if path_details:
                final_state = path_details[-1][3]
                while isinstance(final_state, tuple):
                    final_state = final_state[0]
                if isinstance(final_state, str) and final_state.startswith('S'):
                    detail_sequence.append(final_state)
            completed_detail.append(detail_sequence)

            # 一旦覆盖了所有剩余排列，立即返回
            if remaining_perms <= covered:
                break

        return completed_TS, completed_detail, covered

    # ------------------------
    # 增量更新生成测试套件
    # ------------------------
    def generate_incremental_suite(self):
        """
        增量测试生成：
          1) 保留原测试套件
          2) 在整个套件中只替换一次受 delta 影响的片段（不再做 BFS 可接受性验证）
          3) 启发式补全覆盖剩余未覆盖的 t-perm（基于更新后的 FSM self.F）
          4) 若仍有未覆盖 t-perm，则用乘积自动机兜底
        最终更新 self.TS / self.TS_detail，并返回 self.TS
        """
        # 1) 保留原测试套件
        new_TS = list(self.original_TS)
        new_TS_detail = list(self.original_TS_detail)

        # 2) 全局只替换一次受 delta 影响的片段
        replacements_done = 0
        max_replacements = getattr(self, "max_replacements", 1)  # 若未设置，默认只替换一次
        for idx, (old_case, old_detail) in enumerate(zip(self.original_TS, self.original_TS_detail)):
            if replacements_done >= max_replacements:
                break

            reps = self.find_replacements(old_detail)
            if not reps:
                continue

            # 只挑一个窗口（这里取第一个；也可 random.choice(reps)）
            start_idx, end_idx = reps[0]

            # 直接替换：不再做 is_sequence_accepted 验证（起止状态一致即可衔接）
            new_input, detail_fragment = self.replace_subsequence(old_case, start_idx, end_idx)
            new_TS[idx] = new_input

            # 重建 detail：old_detail 是 [state,input,state,input,...,state]
            new_detail = old_detail[:start_idx * 2] + detail_fragment + old_detail[end_idx * 2 + 1:]
            new_TS_detail[idx] = new_detail

            replacements_done += 1
            if replacements_done >= max_replacements:
                break

        # 3) 计算覆盖与剩余 t-perm
        from itertools import permutations as _perms
        all_t_perms = set(_perms(self.I, self.t))
        covered = set()
        for test_case in new_TS:
            covered |= self.get_covered_permutations(test_case)
        remaining_perms = all_t_perms - covered

        # 4) 启发式补全（在更新后的 FSM self.F 上）
        if remaining_perms:
            add_TS, add_detail, newly_covered = self.heuristic_completion(remaining_perms)
            if add_TS:
                new_TS.extend(add_TS)
                new_TS_detail.extend(add_detail)
                covered |= newly_covered
                remaining_perms = all_t_perms - covered

        # 5) 兜底：用乘积自动机覆盖剩余 t-perm
        while remaining_perms:
            current_constraints = []
            remaining_list = list(remaining_perms)
            # 一次打包若干个未覆盖排列（数量不超过 self.N）
            while len(current_constraints) < self.N and remaining_list:
                ridx = random.randint(0, len(remaining_list) - 1)
                p = remaining_list.pop(ridx)
                constraint_fsm = build_subsequence_mealy_empty_output(self.I, p)
                # 只保留与更新后的 FSM 有交集的约束
                if have_intersection(constraint_fsm, self.F):
                    current_constraints.append(constraint_fsm)

            if not current_constraints:
                # 没有可交集的约束，跳出避免死循环
                break

            # 组合：从更新后的 FSM 开始依次相乘
            combined = copy.deepcopy(self.F)
            for c in current_constraints:
                combined = multiply(combined, c)

            try:
                path_details, test_case = combined.generate_accepting_path()
            except Exception:
                # 出错则跳过本轮继续
                continue

            if test_case:
                new_TS.append(test_case)

                # 从 path_details 构建 detail 序列
                detail_sequence = []
                for step in path_details:
                    state = step[0]
                    while isinstance(state, tuple):
                        state = state[0]
                    if isinstance(state, str) and state.startswith('S'):
                        detail_sequence.extend([state, step[1]])
                if path_details:
                    final_state = path_details[-1][3]
                    while isinstance(final_state, tuple):
                        final_state = final_state[0]
                    if isinstance(final_state, str) and final_state.startswith('S'):
                        detail_sequence.append(final_state)
                new_TS_detail.append(detail_sequence)

                # 更新覆盖与剩余
                covered |= self.get_covered_permutations(test_case)
                remaining_perms = all_t_perms - covered
            # 若未生成 test_case，则继续 while remaining_perms 重试

        # 6) 写回并返回
        self.TS = new_TS
        self.TS_detail = new_TS_detail
        return self.TS

    def find_replacements(self, detail_sequence):
        """
        返回 replacement windows，但基于 inputs 的索引区间：
        states = [s0, s1, s2, ...]
        inputs = [a0, a1, ...]  # len(inputs) = len(states)-1
        若要替换从 states[i] 到 states[j]（j>i），对应 input 的切片是 inputs[i:j]
        本函数返回 (i, j)，表示 inputs 的切片范围 [i, j)
        """
        states = []
        inputs = []
        for i, elem in enumerate(detail_sequence):
            if i % 2 == 0 or i == len(detail_sequence) - 1:
                states.append(elem)
            else:
                inputs.append(elem)

        replacements = []
        for si in range(len(states)):
            if states[si] != self.delta.start_state:
                continue
            for sj in range(si + 1, len(states)):
                if states[sj] == self.delta.end_state:
                    # 对应的 inputs 切片是 inputs[si:sj]
                    replacements.append((si, sj))
        return replacements

    def replace_subsequence(self, original_input, start_idx, end_idx):
        if not self.delta.transitions:
            return original_input, []  # 避免 delta 为空时报错

        selected_transition = random.choice(self.delta.transitions)
        if not selected_transition:
            return original_input, []  # 避免空列表

        input_sequence = [step[0] for step in selected_transition]
        detail_fragment = []
        current_state = self.delta.start_state

        for step in selected_transition:
            input_symbol, output, next_state = step
            detail_fragment.append(current_state)
            detail_fragment.append(input_symbol)
            current_state = next_state
        detail_fragment.append(current_state)

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
            if input_symbol not in new_fsm.inputs:
                new_fsm.inputs.add(input_symbol)
                # 🔑 初始化所有状态的转移字典，避免 KeyError
                for s in new_fsm.states:
                    if input_symbol not in new_fsm.transition_func[s]:
                        new_fsm.transition_func[s][input_symbol] = []

            # 确保当前状态存在
            new_fsm.add_state(current_state)
            # 确保下一状态存在
            new_fsm.add_state(next_state)

            # 添加迁移到 transition_func
            new_fsm.transition_func[current_state][input_symbol].append((output, next_state))
            # 添加输出符号到 outputs 集合
            new_fsm.outputs.add(output)

            current_state = next_state  # 更新当前状态为下一步的起点
    return new_fsm




import random
import time
import gc
from itertools import permutations
from multiprocessing import Process, Queue
import copy



def build_large_fsm(num_states=20, num_inputs=5, seed=42):
    """
    改进版 FSM 生成：
      - 每个状态至少有一条路径到接受状态
      - 输出和下一状态部分随机，但结构稳定
      - 可选随机种子固定结果
    """
    import random
    random.seed(seed)

    states = [f"S{i}" for i in range(num_states)]
    inputs = [f"a{i}" for i in range(num_inputs)]
    outputs = [f"o{i}" for i in range(num_inputs)]
    transition_func = {s: {i: [] for i in inputs} for s in states}

    initial_state = "S0"
    accepting_state = f"S{num_states-1}"
    accepting_states = {accepting_state}

    # 1️⃣ 先生成一条必达路径，保证每个状态至少可达接受状态
    path_states = list(range(num_states))
    for idx in range(num_states - 1):
        s = states[path_states[idx]]
        next_s = states[path_states[idx + 1]]
        i = random.choice(inputs)
        o = random.choice(outputs)
        transition_func[s][i].append((o, next_s))

    # 2️⃣ 对每个状态补充额外随机转移，增加随机性
    for s_index, s in enumerate(states):
        for i in inputs:
            # 已有转移则跳过一部分，保证每个输入至少有一条转移
            if not transition_func[s][i]:
                next_state = random.choice(states[s_index:])  # 优先向后连
                output = random.choice(outputs)
                transition_func[s][i].append((output, next_state))
            # 随机再增加一个转移，提高非确定性
            if random.random() < 0.3:
                next_state = random.choice(states)
                output = random.choice(outputs)
                transition_func[s][i].append((output, next_state))

    return MealyNFA(set(states), set(inputs), set(outputs), transition_func, initial_state, accepting_states)


def demo_example(num_runs=5, max_print=3):
    print("=== Demo Example: Small FSM with Delta Update ===")

    noninc_times, noninc_sizes, noninc_lengths = [], [], []
    inc_times, inc_sizes, inc_lengths = [], [], []

    for run in range(num_runs):
        print(f"\n--- Run {run+1} ---")
        # 1️⃣ 构建原始 FSM
        original_fsm = build_large_fsm(num_states=10, num_inputs=5)

        # 2️⃣ 构建 delta
        delta_transitions = [
            [("new_a0", "new_o0", "S5")]
        ]
        delta = DeltaUpdate("S0", delta_transitions, "S5")

        # 3️⃣ 应用 delta 更新 FSM
        new_fsm = apply_delta_updates(original_fsm, delta)

        # 4️⃣ 非增量生成
        tg_full = TestGenerator(list(new_fsm.inputs), new_fsm, strength=2, max_tuples=8)
        start_time = time.time()
        TS_full, TS_full_detail = tg_full.generate_test_suite()
        time_no_inc = time.time() - start_time

        noninc_times.append(time_no_inc)
        noninc_sizes.append(len(TS_full))
        if TS_full:
            noninc_lengths.append(sum(len(tc) for tc in TS_full) / len(TS_full))
        else:
            noninc_lengths.append(0)

        print(f"[非增量] 时间: {time_no_inc:.4f}s, 测试数: {len(TS_full)}, 平均长度: {noninc_lengths[-1]:.2f}")
        for i, (tc, detail) in enumerate(zip(TS_full[:max_print], TS_full_detail[:max_print])):
            print(f"  Test {i+1}: Input: {tc}, Detail: {detail}")

        # 5️⃣ 增量生成
        tg_old = TestGenerator(list(original_fsm.inputs), original_fsm, strength=2, max_tuples=8)
        TS_old, TS_old_detail = tg_old.generate_test_suite()
        inc_tg = IncrementalTestGenerator(tg_old, delta, new_fsm)
        start_time = time.time()
        TS_inc = inc_tg.generate_incremental_suite()
        time_inc = time.time() - start_time

        inc_times.append(time_inc)
        inc_sizes.append(len(TS_inc))
        if TS_inc:
            inc_lengths.append(sum(len(tc) for tc in TS_inc) / len(TS_inc))
        else:
            inc_lengths.append(0)

        print(f"[增量]   时间: {time_inc:.4f}s, 测试数: {len(TS_inc)}, 平均长度: {inc_lengths[-1]:.2f}")
        for i, tc in enumerate(TS_inc[:max_print]):
            print(f"  Incremental Test {i+1}: Input: {tc}")

    # 6️⃣ 输出平均结果
    print("\n=== 平均结果 (运行 {} 次) ===".format(num_runs))
    print(f"[非增量] 平均时间: {sum(noninc_times)/num_runs:.4f}s, "
          f"平均测试数: {sum(noninc_sizes)/num_runs:.2f}, "
          f"平均长度: {sum(noninc_lengths)/num_runs:.2f}")
    print(f"[增量]   平均时间: {sum(inc_times)/num_runs:.4f}s, "
          f"平均测试数: {sum(inc_sizes)/num_runs:.2f}, "
          f"平均长度: {sum(inc_lengths)/num_runs:.2f}")



if __name__ == "__main__":
    demo_example(num_runs=10)

