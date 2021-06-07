from ManualProgram import operators
from inspect import getmembers, isfunction
import itertools
import math


constant = [30, 60, 90, 180, 360, math.pi, 0.618]
op_dict = {0: 'g_equal', 1: 'g_double', 2: 'g_half', 3: 'g_add', 4: 'g_minus',
          5: 'g_sin', 6: 'g_cos', 7: 'g_tan', 8: 'g_asin', 9: 'g_acos',
          10: 'gougu_add', 11: 'gougu_minus', 12: 'g_bili',
          13: 'g_mul', 14: 'g_divide', 15: 'cal_circle_area', 16: 'cal_circle_perimeter', 17: 'cal_cone'}
op_list = [op_dict[key] for key in sorted(op_dict.keys())]


class Equations:
    def __init__(self):

        self.op_list = op_list
        self.op_num = {}
        self.call_op = {}
        self.exp_info = None
        self.results = []
        self.max_step = 3
        self.max_len = 7
        for op in self.op_list:
            self.call_op[op] = eval('operators.{}'.format(op))
            # self.call_op[op] = eval(op)
            self.op_num[op] = self.call_op[op].__code__.co_argcount

    def str2exp(self, inputs):
        inputs = inputs.split(',')
        exp = inputs.copy()
        for i, s in enumerate(inputs):
            if 'n' in s or 'v' in s or 'c' in s:
                exp[i] = s.replace('n', 'N_').replace('v', 'V_').replace('c', 'C_')
            else:
                exp[i] = op_dict[int(s[2:])]
            exp[i] = exp[i].strip()

        self.exp = exp
        return exp

    def excuate_equation(self, exp, source_nums=None):

        if source_nums is None:
            source_nums = self.exp_info['nums']
        vars = []
        idx = 0
        while idx < len(exp):
            op = exp[idx]
            if op not in self.op_list:
                return None
            op_nums = self.op_num[op]
            if idx + op_nums >= len(exp):
                return None
            excuate_nums = []
            for tmp in exp[idx + 1: idx + 1 + op_nums]:
                if tmp[0] == 'N' and int(tmp[-1]) < len(source_nums):
                    excuate_nums.append(source_nums[int(tmp[-1])])
                elif tmp[0] == 'V' and int(tmp[-1]) < len(vars):
                    excuate_nums.append(vars[int(tmp[-1])])
                elif tmp[0] == 'C' and int(tmp[-1]) < len(constant):
                    excuate_nums.append(constant[int(tmp[-1])])
                else:
                    return None
            idx += op_nums + 1
            v = self.call_op[op](*excuate_nums)
            if v is None:
                return None
            vars.append(v)
        return vars


if __name__ == '__main__':
    eq = Equations()

