import random
import numpy as np
from itertools import permutations
from collections import defaultdict, deque
from typing import Sequence
from itertools import product
from hashlib import sha256
import time
from statistics import mean, stdev
from Incremental import MealyNFA, build_subsequence_mealy_empty_output, have_intersection, multiply, TestGenerator

class SequenceProcessor:
    def __init__(self, original_inputs, new_inputs, t_strength):
        self.original_inputs = original_inputs
        self.new_inputs = new_inputs
        self.t_strength = t_strength
        self.coverage_records = set()
        self.bug_reports = []

    def process_test_suite(self, test_suite, fsm):
        filtered_suite = self._filter_sequences(test_suite)
        self._record_coverage(filtered_suite)
        bug_count = self._insert_and_test(filtered_suite, fsm)
        return {
            'filtered_suite': filtered_suite,
            'coverage': self.coverage_records.copy(),
            'bug_count': bug_count
        }

    def _record_coverage(self, test_suite):
        for seq in test_suite:
            for i in range(len(seq) - self.t_strength + 1):
                perm = tuple(seq[i:i+self.t_strength])  # 确保 perm 是元组
                if all(x in self.original_inputs for x in perm):
                    self.coverage_records.add(perm)

    def _filter_sequences(self, test_suite):
        return [
            [x for x in seq if x in self.original_inputs]
            for seq in test_suite
            if any(x in self.original_inputs for x in seq)
        ]

    def _insert_and_test(self, test_suite, fsm):
        bug_count = 0
        for seq in test_suite:
            if len(seq) < 2:
                continue

            if random.random() < 0.1:  # 10% 概率插入新输入
                insert_pos = random.randint(1, len(seq)-1)
                new_input = random.choice(list(self.new_inputs))
                modified_seq = seq[:insert_pos] + [new_input] + seq[insert_pos:]

                accepted, _ = fsm.is_sequence_accepted(modified_seq)
                if not accepted:
                    bug_count += 1
                    self.bug_reports.append({
                        'original': seq,
                        'modified': modified_seq,
                        'position': insert_pos,
                        'new_input': new_input
                    })
        return bug_count

def save_results_to_txt(filename, test_suite, coverage, bug_reports):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== 测试序列总数量 ===\n")
        f.write(f"{len(test_suite)}\n\n")

        f.write("=== 所有测试序列及其覆盖的排列和Bug状态 ===\n")
        # 提取所有触发Bug的原始序列（用于判断当前序列是否有Bug）
        bug_original_sequences = {tuple(bug['original']) for bug in bug_reports}

        for i, seq in enumerate(test_suite):
            covered_perms = []
            for j in range(len(seq) - 1):
                perm = tuple(seq[j:j+2])
                if perm in coverage:
                    covered_perms.append(perm)
            
            # 判断当前序列是否有Bug
            has_bug = tuple(seq) in bug_original_sequences
            
            f.write(f"序列 {i+1}: {seq}\n")
            f.write(f"覆盖的排列: {covered_perms}\n")
            f.write(f"是否有Bug: {'是' if has_bug else '否'}\n")
            f.write("---\n")

        f.write("\n=== 所有发现的Bug详情 ===\n")
        for i, bug in enumerate(bug_reports):
            f.write(f"Bug {i+1}:\n")
            f.write(f"原始序列: {bug['original']}\n")
            f.write(f"修改后序列: {bug['modified']}\n")
            f.write(f"插入位置: {bug['position']}\n")
            f.write(f"插入输入: {bug['new_input']}\n")
            f.write("---\n")

class OptimizedTestGenerator:
    def __init__(self, events, fsm, strength, max_tuples):
        self.I = events
        self.F = fsm
        self.t = strength
        self.N = max_tuples
        self.TS = []
        
        # 贝叶斯概率统计
        self.perm_stats = defaultdict(lambda: {'total': 0, 'bugs': 0})
        self.bug_prob = defaultdict(float)

    def generate_t_perms(self):
        """生成所有t-way排列"""
        return list(permutations(self.I, self.t))

    def update_bug_probabilities(self, test_suite, bug_reports):
        bug_sequences = {tuple(bug['original']) for bug in bug_reports}

        for seq in test_suite:
            seq_tuple = tuple(seq)  # 确保 seq 是元组
            for i in range(len(seq) - self.t + 1):
                perm = tuple(seq[i:i+self.t])  # 确保 perm 是元组
                self.perm_stats[perm]['total'] += 1
                if seq_tuple in bug_sequences:
                    self.perm_stats[perm]['bugs'] += 1

        alpha_prior, beta_prior = 1, 1
        for perm in self.perm_stats:
            alpha_post = alpha_prior + self.perm_stats[perm]['bugs']
            beta_post = beta_prior + (self.perm_stats[perm]['total'] - self.perm_stats[perm]['bugs'])
            self.bug_prob[perm] = alpha_post / (alpha_post + beta_post)

    def get_low_risk_perms(self, perms):
        """获取Bug概率最低的排列"""
        if not self.bug_prob:  # 如果没有历史数据，随机选择
            return random.sample(perms, min(len(perms), self.N))
        
        # 按Bug概率升序排序
        sorted_perms = sorted(perms, key=lambda p: self.bug_prob.get(p, 0.0))
        return sorted_perms[:self.N]
    
    def generate_test_suite(self, num_sequences=None):
        """生成优化的测试套件"""
        T = self.generate_t_perms()
        TS = []
        
        while T and (num_sequences is None or len(TS) < num_sequences):
            current_constraints = []
            active_fsm = self.F
            
            # 优先选择低Bug概率的可行约束
            candidate_perms = self.get_low_risk_perms(T)
            
            for p in candidate_perms:
                if len(current_constraints) >= self.N:
                    break
                    
                a = build_subsequence_mealy_empty_output(self.I, p)
                
                if have_intersection(a, active_fsm):
                    current_constraints.append(a)
                    try:
                        T.remove(p)  # 确保每个排列只使用一次
                    except ValueError:
                        continue
            
            if current_constraints:
                # 构建组合约束
                combined = active_fsm
                for constraint in current_constraints:
                    combined = multiply(combined, constraint)
                
                # 生成测试用例
                try:
                    _, test_case = combined.generate_accepting_path()
                    TS.append(list(test_case))  # 转换为列表存储
                except:
                    continue

        return TS

# 示例使用
def create_constraintFSM():
    transition_func = {
        'S0': defaultdict(list, {
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
    original_inputs = {'sc_d', 'sc_a', 'sc_m', 'eb'}
    new_inputs = {'A_dz'}
    t_strength = 2
    max_tuples = 5
    events = original_inputs
    fsmB = create_modified_constraintFSM()

    # 生成初始测试套件
    generator = TestGenerator(events, fsmB, strength=2, max_tuples=5)
    test_suite = []
    for _ in range(100):
        test_suite.extend(generator.generate_test_suite())

    processor = SequenceProcessor(original_inputs, new_inputs, t_strength)
    result = processor.process_test_suite(test_suite, fsmB)

    # 初始化优化生成器并更新Bug概率
    optimized_generator = OptimizedTestGenerator(events, fsmB, t_strength, max_tuples)
    optimized_generator.update_bug_probabilities(result['filtered_suite'], processor.bug_reports)

    # 生成优化后的测试套件（Bug概率更低）
    optimized_suite = optimized_generator.generate_test_suite(100)

    # 验证优化效果
    optimized_result = processor.process_test_suite(optimized_suite, fsmB)
    print(f"原始Bug数量: {result['bug_count']}")
    print(f"优化后Bug数量: {optimized_result['bug_count']}")