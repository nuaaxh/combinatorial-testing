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
                except:
                    pass

        return TS


class IncrementalMultiProduct:
    def __init__(self, base_machines, existing_product):
        self.machines = base_machines
        self.special_idx = 0  # Index of the machine we're modifying
        self.product = existing_product
        self.modifications = set()

    def add_transition(self, state, input_symbol, output, next_state):
        """Add transition to the special machine and record modification"""
        special_machine = self.machines[self.special_idx]

        # Ensure states exist
        if state not in special_machine.states:
            special_machine.states.add(state)
            special_machine.transition_func[state] = defaultdict(list)

        if next_state not in special_machine.states:
            special_machine.states.add(next_state)
            special_machine.transition_func[next_state] = defaultdict(list)

        # Add transition
        special_machine.transition_func[state][input_symbol].append((output, next_state))

        # Record modification
        self.modifications.add((state, input_symbol))

    def _get_component(self, state, index):
        """Flatten state tuple and get component at index"""
        if not isinstance(state, tuple):
            return state if index == 0 else None

        components = []
        stack = [state]
        while stack:
            current = stack.pop()
            if isinstance(current, tuple):
                stack.extend(reversed(current))
            else:
                components.append(current)

        return components[index] if index < len(components) else None

    def _set_component(self, state, index, new_value):
        """Set component at index to new_value, preserving structure"""
        if not isinstance(state, tuple):
            return new_value if index == 0 else state

        components = []
        structure = []
        stack = [state]
        while stack:
            current = stack.pop()
            if isinstance(current, tuple):
                structure.append(len(current))
                stack.extend(reversed(current))
            else:
                components.append(current)

        if index >= len(components):
            return state

        components[index] = new_value

        # Rebuild the original structure
        def build(comps, struct):
            if not struct:
                return comps.pop(0)
            size = struct.pop(0)
            return tuple(build(comps, struct) for _ in range(size))

        return build(components, structure[::-1])

    def incremental_update(self):
        """Update product automaton incrementally"""
        special = self.machines[self.special_idx]

        # 1. Collect new states from modifications
        new_states = set()
        for state, input_sym in self.modifications:
            for _, next_state in special.transition_func[state].get(input_sym, []):
                if next_state not in special.states:
                    new_states.add(next_state)

        # 2. Add new states to special machine
        for new_state in new_states:
            special.states.add(new_state)
            special.transition_func[new_state] = defaultdict(list)
            for sym in special.inputs:
                special.transition_func[new_state][sym].append(('', new_state))

        # 3. Generate new product states
        for new_state in new_states:
            for prod_state in list(self.product.states):
                new_prod_state = self._set_component(prod_state, self.special_idx, new_state)

                if new_prod_state not in self.product.states:
                    self.product.states.add(new_prod_state)
                    # Check if accepting
                    if all(m.is_accepting(self._get_component(new_prod_state, i))
                           for i, m in enumerate(self.machines)):
                        self.product.accepting_states.add(new_prod_state)

        # 4. Update transition relations
        for mod_state, mod_input in self.modifications:
            for prod_state in [s for s in self.product.states
                               if self._get_component(s, self.special_idx) == mod_state]:

                # Clear existing transitions for this input
                if mod_input in self.product.transition_func[prod_state]:
                    del self.product.transition_func[prod_state][mod_input]

                # Collect transitions from all machines
                transitions = []
                for i, machine in enumerate(self.machines):
                    component = self._get_component(prod_state, i)
                    trans = machine.transition_func[component].get(mod_input, [])
                    # Filter out empty/default transitions
                    filtered_trans = [t for t in trans if t[0] != ('' if i != self.special_idx else 'DEFAULT_LOOP')]
                    transitions.append(filtered_trans if filtered_trans else trans)

                # Generate all possible transition combinations
                for trans_combo in product(*transitions):
                    outputs, next_states = zip(*trans_combo)
                    # Build new product state
                    next_prod_state = tuple(next_states)

                    # Add to product if new
                    if next_prod_state not in self.product.states:
                        self.product.states.add(next_prod_state)
                        # Check if accepting
                        if all(m.is_accepting(self._get_component(next_prod_state, i))
                               for i, m in enumerate(self.machines)):
                            self.product.accepting_states.add(next_prod_state)

                    # Add transition
                    self.product.transition_func[prod_state][mod_input].append((outputs, next_prod_state))

        self.modifications = set()
        return self.product


class IncrementalTestGenerator(TestGenerator):
    def __init__(self, events, fsm, strength, max_tuples):
        super().__init__(events, fsm, strength, max_tuples)
        self.product_cache = {}
        self.current_product = fsm
        self.updater = IncrementalMultiProduct([fsm], fsm)

    def _get_fsm_hash(self, fsm):
        """生成自动机内容哈希"""
        # 处理状态集合
        states_str = [str(s) for s in fsm.states]  # 强制转换为字符串
        states_hash = sha256(','.join(sorted(states_str)).encode()).hexdigest()
        
        # 处理转移关系
        transitions = []
        for state in fsm.transition_func:
            for symbol in fsm.transition_func[state]:
                for output, next_state in fsm.transition_func[state][symbol]:
                    # 统一输出值为字符串
                    output_str = ','.join(map(str, output)) if isinstance(output, tuple) else str(output)
                    transitions.append(
                        f"{str(state)}-{str(symbol)}-{output_str}-{str(next_state)}"
                    )
        
        # 生成转移哈希
        trans_hash = sha256('|'.join(sorted(transitions)).encode()).hexdigest()
        
        # 综合哈希
        return sha256(f"{states_hash}:{trans_hash}".encode()).hexdigest()[:16]

    def _incremental_multiply(self, base, constraint):
        cache_key = (self._get_fsm_hash(base), self._get_fsm_hash(constraint))
        if cache_key in self.product_cache:
            # print(f"[cache got] Key: {cache_key}")
            return self.product_cache[cache_key]
        else:
            # print(f"[cache not got] Key: {cache_key}")
            product = multiply(base, constraint)
            self.product_cache[cache_key] = product
            return product

    def clear_cache(self):
        self.product_cache.clear()

    def generate_test_suite(self, modifications=None):
        """支持增量更新的测试套件生成方法"""
        if modifications:
            self._apply_modifications(modifications)

        T = self.generate_t_perms()
        TS = []

        while T:
            current_constraints = []

            # 阶段1：收集可行约束
            while len(current_constraints) < self.N and T:
                p = T.pop(random.randint(0, len(T) - 1))
                constraint_fsm = build_subsequence_mealy_empty_output(self.I, p)

                # 使用增量检查优化
                if self._incremental_intersection_check(constraint_fsm):
                    current_constraints.append(constraint_fsm)

            # 阶段2：增量组合约束
            if current_constraints:
                temp_product = self.current_product
                for c in current_constraints:
                    temp_product = self._incremental_multiply(temp_product, c)

                # 阶段3：路径生成
                path = self._optimized_path_generation(temp_product)
                if path:
                    TS.append(path)

        return TS

    def _apply_modifications(self, modifications):
        """应用模型变更并增量更新"""
        for state, input_symbol, output, next_state in modifications:
            self.updater.add_transition(state, input_symbol, output, next_state)

        # 增量更新乘积自动机
        self.current_product = self.updater.incremental_update()
        self._update_product_cache()

    def _incremental_intersection_check(self, constraint_fsm):
        """优化的交集检查"""
        # 检查缓存中是否存在已验证的约束
        cache_key = tuple(sorted(constraint_fsm.states))
        if cache_key in self.product_cache:
            return self.product_cache[cache_key]

        # 执行实际检查
        result = have_intersection(self.current_product, constraint_fsm)
        self.product_cache[cache_key] = result
        return result

    def _optimized_path_generation(self, product):
        """优化的接受路径生成"""
        # 优先探索新增转移的路径
        queue = deque([(product.initial_state, [])])
        visited = set()

        while queue:
            current, path = queue.popleft()

            if product.is_accepting(current):
                return path

            # 优先处理新增转移
            transitions = []
            for input_symbol in product.inputs:
                for (output, next_state) in product.transition_func[current].get(input_symbol, []):
                    is_new = any((current, input_symbol) in self.updater.modifications
                                 for _ in self.updater.machines)
                    transitions.append((is_new, input_symbol, next_state))

            # 按是否新增排序
            transitions.sort(reverse=True, key=lambda x: x[0])

            for is_new, input_symbol, next_state in transitions:
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [input_symbol]))

        return None

    def _update_product_cache(self):
        """更新乘积自动机缓存"""
        new_cache = {}
        for key, product in self.product_cache.items():
            # 重新计算受影响的乘积
            if any(m in key for m in self.updater.modifications):
                continue
            new_cache[key] = product
        self.product_cache = new_cache

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
            # 'eb': [('stop', 'S0')],
            # 'sc_m': [('stop', 'S0')],
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
    execution_times = []
    all_test_lengths = []
    for _ in range(100):
        # 每次创建新的测试生成器
        generator = TestGenerator(events, fsmB, strength=2, max_tuples=3)

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
