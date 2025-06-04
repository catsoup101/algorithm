# %%
from typing import Any, Callable, List, Dict, Literal, Set, Tuple, Optional
from networkx import union # type: ignore
import numpy, sympy, scipy, gc
from enum import auto
from sympy import Symbol, false, true
from numpy.typing import NDArray


class ListNode:
    # 属性声明: public
    val: int
    next: Optional["ListNode"]

    # 方法声明
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"{hex(id(self))}"


class Solution:
    # 1.删除有序数组重复项
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)
        slow: int = 2
        fast: int = 2

        while fast < len(nums):
            if nums[fast] != nums[slow - 2]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow

    # 2.分隔链表
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        if head == None:
            return None
        small: list[ListNode] = list[ListNode]()
        big: list[ListNode] = list[ListNode]()

        while head != None:
            if head.val < x:
                small.append(ListNode(head.val))
            elif head.val >= x:
                big.append(ListNode(head.val))
            head = head.next
        small.extend(big)

        for i in range(0, len(small) - 1):
            small[i].next = small[i + 1]
        return small[0]

    # 3.搜索旋转排序数组Ⅰ
    def _search(self, nums: List[int], left: int, right: int):
        if left >= right:
            return left + 1
        elif nums[left] < nums[right]:
            return 0
        return self._search(nums, left, (left + right) // 2) or self._search(
            nums, (left + right) // 2, right
        )

    def search(self, nums: List[int], target: int) -> int:
        if (0 < len(nums) <= 1) and (target == nums[0]):
            return 0
        if (0 < len(nums) <= 1) and (target != nums[0]):
            return -1

        num_min: int = self._search(nums, 0, len(nums) - 1)
        left: int = 0
        right: int = num_min - 1
        mid: int = 0

        if nums[num_min] <= target <= nums[len(nums) - 1]:
            left = num_min
            right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if target < nums[mid]:
                right = mid - 1
            elif target > nums[mid]:
                left = mid + 1
            else:
                return mid
        return -1

    # 4.搜索旋转排序数组II
    def searchTwo(self, nums: List[int], target: int) -> bool:
        return True


class Numeric:
    # 1.集邮员问题(调和级数)
    def StampCollector(self):
        pass

    # 2.套装集齐问题(容斥原理)
    def completeSet(self):
        pass

    # 3.洗牌算法
    def shufflingAlgorithm(self):
        pass

    # 4.拉格朗日乘数(最优化问题)
    def planningIissues(
        self, Lagrange_func: Symbol, a: Symbol, b: Symbol, c: Symbol, lambda_: Symbol
    ) -> auto:

        grad_L: list[sympy.Expr] = []
        grad_L_func: List[Callable[..., float]] = []

        for var in (a, b, c, lambda_):
            grad_L.append(sympy.diff(Lagrange_func, var))
        for eq in grad_L:
            grad_L_func.append(sympy.lambdify((a, b, c, lambda_), eq))

        equations: Callable[[List[float]], List[float]] = lambda vars: [
            func(*vars) for func in grad_L_func
        ]
        initial_guess: list[int] = [1, 1, 1, 1]
        solution = scipy.optimize.fsolve(equations, initial_guess)
        return solution

    # 5.解线性方程组
    def solve_least_squares(self, A, b) -> List[int]:
        x, residuals, rank, s = numpy.linalg.lstsq(A, b)
        return x.tolist()

    # 6.概率递增问题(求马尔科夫链稳态概率)
    def steady_state(self, P: numpy.ndarray):
        dim = P.shape[0]  # 行数
        Q = P.T - numpy.eye(dim)  # 创建单位矩阵
        ones_row = numpy.ones(dim)  # 创建状态概率x1+x2x+x3
        Q = numpy.vstack([Q, ones_row])  # 将这个概率放在Q马尔科夫链下

        b = numpy.zeros(dim + 1)  # 构建态概率x1+x2x+x3的系数
        b[-1] = 1
        # 使用最小二乘法求解线性方程组
        pi, residuals, rank, s = numpy.linalg.lstsq(Q, b, rcond=None)

        return pi

    # 直接保底模型的无限次单卡期望价值(求马尔科夫链平均吸收时间)
    def calculate_expectations(self) -> tuple[float, float, float]:
        T: List[float] = [0.0] * 11  # 11个状态，0-9是中间状态，10是吸收状态
        T[9] = 1.0  # 第9个状态直接转移到吸收状态10

        # 根据递归公式计算其他T[i]的值.T[i]表示的是平均尝试次数,即逃逸到吸收状态的时间
        for i in range(8, -1, -1):
            T[i] = 1.0 + 0.9 * T[i + 1] + 0.1 * T[10]

        # 计算综合保底概率，表示抽到SR卡的平均概率
        SP: float = 1.0 / T[0]

        # 卡片的价值
        SR: float = 20.0  # SR卡的价值
        R: float = 1.0  # R卡的价值

        # 计算单次抽卡的期望价值
        E: float = SP * SR + (1 - SP) * R

        return T[0], SP, E

    # 计算prd模型的保底单卡期望
    def calculateExpectedValue(self, p: float, M: int, N: int, T) -> float:
        # 计算几何级数部分的期望值
        geometric_sum: float = sum(n * p * pow((1 - p), (n - 1)) for n in range(1, M))

        # 计算增长概率部分的期望值
        growth_sum: float = 0
        for n in range(M, N + 1):
            prod: numpy.generic = numpy.prod([(1 - T(m)) for m in range(M, n)])
            growth_sum += n * T(n) * prod

        # 获取总期望值
        return geometric_sum + growth_sum

    @classmethod
    def calculateValues(cls) -> None:
        guarantee: Numeric = Numeric()
        # 马尔科夫链
        # P: numpy.ndarray = numpy.array(
        #     [
        #         [0, 1, 0, 0, 0, 0],  # S0 -> S1
        #         [0.16, 0, 0.84, 0, 0, 0],  # S1 -> S0, S2
        #         [0.32, 0, 0, 0.68, 0, 0],  # S2 -> S0, S3
        #         [0.48, 0, 0, 0, 0.52, 0],  # S3 -> S0, S4
        #         [0.64, 0, 0, 0, 0, 0.36],  # S4 -> S0, S5
        #         [0.8, 0, 0, 0, 0, 0.2],  # S5 -> S0, S5
        #     ]
        # )
        # crit_rates: numpy.ndarray = numpy.array([0, 0.16, 0.32, 0.48, 0.64, 0.80])
        # pi: numpy.ndarray = steady_state(P)
        # average_crit_rate: float = numpy.dot(pi, crit_rates)
        # print(f"Steady State:{pi}, Average Crit:{average_crit_rate}")

        # 解线性方程组
        # coefficients_matrix: NDArray[Any] = numpy.array(
        #     [
        #         [1.00, 0.00, 0.00, 0.00, 0.00],
        #         [0.80, 0.20, 0.00, 0.00, 0.00],
        #         [0.69, 0.30, 0.01, 0.00, 0.00],
        #         [0.53, 0.42, 0.05, 0.00, 0.00],
        #         [0.42, 0.45, 0.12, 0.01, 0.00],
        #         [0.30, 0.42, 0.25, 0.02, 0.01],
        #         [0.26, 0.40, 0.28, 0.04, 0.02],
        #         [0.20, 0.39, 0.32, 0.06, 0.03],
        #         [0.18, 0.25, 0.40, 0.12, 0.05],
        #         [0.12, 0.15, 0.28, 0.35, 0.10],
        #     ]
        # )
        # result_vector: NDArray[Any] = numpy.array(
        #     [399.60, 324.36, 256.32, 196.28, 144.24, 100.2, 64.16, 36.12, 16.08, 4.04]
        # )
        # solution: List[int] = solve_least_squares(coefficients_matrix, result_vector)
        # print(f"{solution}")

        # 拉格朗日乘数
        # parma: tuple[Symbol, Symbol, Symbol, Symbol] = sympy.symbols("a b c lambda")
        # a, b, c, lambda_ = parma

        # target_fun: numpy.multiply = 5 * a * (1 + 0.1 * b) * (1 + 0.2 * c)  # type: ignore
        # constraint: numpy.add = a + b + c - 45  # type: ignore
        # lagrange_fun: Symbol = target_fun + lambda_ * constraint

        # a_val, b_val, c_val, lambda_val = planningIissues(lagrange_fun, a, b, c, lambda_)
        # max_f = 5 * a_val * (1 + 0.1 * b_val) * (1 + 0.2 * c_val)
        # print(f"Optimal a:{a_val},Optimal b:{b_val},Optimal c:{c_val},Maximum:{max_f}")

        #  连续补偿的抽卡概率模型
        # basic_p: float = 0.02
        # max_count: int = 50
        # basic_count: int = 80
        # tFunctoin: Callable[[int], float] = lambda n: 0.02 + 0.01 * (n - max_count)

        # e_x: float = guarantee.calculateExpectedValue(
        #     basic_p, max_count, basic_count, tFunctoin
        # )
        # print(e_x)


def main() -> None:
    solution: Solution = Solution()
    print(solution.search([3, 1], 0))

    # Numeric().calculateValues()


if __name__ == "__main__":
    main()
