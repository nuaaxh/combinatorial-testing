from Incremental import MealyNFA, TestGenerator
import random
from collections import defaultdict

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

    def _filter_sequences(self, test_suite):
        return [
            [x for x in seq if x in self.original_inputs]
            for seq in test_suite
            if any(x in self.original_inputs for x in seq)
        ]

    def _record_coverage(self, test_suite):
        for seq in test_suite:
            for i in range(len(seq) - self.t_strength + 1):
                perm = tuple(seq[i:i+self.t_strength])
                if all(x in self.original_inputs for x in perm):
                    self.coverage_records.add(perm)

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
            f.write(f"是否有Bug: {'是' if has_bug else '否'}\n")  # 新增字段
            f.write("---\n")

        f.write("\n=== 所有发现的Bug详情 ===\n")
        for i, bug in enumerate(bug_reports):
            f.write(f"Bug {i+1}:\n")
            f.write(f"原始序列: {bug['original']}\n")
            f.write(f"修改后序列: {bug['modified']}\n")
            f.write(f"插入位置: {bug['position']}\n")
            f.write(f"插入输入: {bug['new_input']}\n")
            f.write("---\n")
            
if __name__ == "__main__":
    original_inputs = {'sc_d', 'sc_a', 'sc_m', 'eb'}
    new_inputs = {'A_dz'}
    t_strength = 2
    events = {'sc_d', 'sc_a', 'sc_m', 'eb'}
    fsmB = create_modified_constraintFSM()
    
    generator = TestGenerator(events, fsmB, strength=2, max_tuples=5)
    test_suiteA = []
    for _ in range(100):
        test_suite = generator.generate_test_suite()
        test_suiteA.extend([tuple(seq) for seq in test_suite])

    processor = SequenceProcessor(original_inputs, new_inputs, t_strength)
    processed_suite = [list(seq) for seq in test_suiteA]
    result = processor.process_test_suite(processed_suite, fsmB)

    # 保存所有结果到txt文件
    save_results_to_txt(
        "test_results.txt",
        result['filtered_suite'],
        result['coverage'],
        processor.bug_reports
    )

    print("所有测试结果已保存到 test_results.txt")