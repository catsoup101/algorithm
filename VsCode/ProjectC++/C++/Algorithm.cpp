#include "my_libraries.h"
struct ListNode
{
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

// 递增矩阵搜索
bool binarySearch(const vector<int> &array, int target)
{
    int mid, left = 0, right = array.size() - 1;

    while (left <= right)
    {
        mid = (left + right) / 2;
        if (array[mid] == target)
        {
            return true;
        }
        else if (array[mid] > target)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return false;
}
bool searchMatrix(vector<vector<int>> &matrix, int target)
{
    if (matrix.empty() || matrix[0].empty())
        return false;

    int row_size = matrix.size();
    int column_size = matrix.front().size();
    int mid, left = 0, right = matrix.size() - 1;

    while (left <= right)
    {
        mid = (left + right) / 2;

        if (matrix[mid][column_size - 1] == target)
        {
            return true;
        }
        else if (matrix[mid][column_size - 1] > target)
        {
            right = mid - 1;
        }
        else if (matrix[mid][column_size - 1] < target)
        {
            left = mid + 1;
        }
    }
    if (left >= 0 && left < row_size)
    {
        return binarySearch(matrix[left], target);
    }
    return false;
}

// 组合
void backtraceCombine(vector<vector<int>> &result, const pair<int, int> &pair, vector<int> &combine, int start)
{
    if (pair.second <= combine.size())
    {
        result.push_back(combine);
        return;
    }
    for (int i = start; i <= pair.first; i++)
    {
        combine.push_back(i);
        backtraceCombine(result, pair, combine, i + 1);

        combine.pop_back();
    }
}
vector<vector<int>> combine(int n, int k)
{
    std::pair<int, int> pair = {n, k};
    vector<int> array;
    vector<vector<int>> result;

    backtraceCombine(result, pair, array, 1);
    return result;
}

// 子集
void backtraceSubsets(vector<vector<int>> &result, vector<int> &combine, vector<int> &nums, int start)
{

    result.push_back(combine);
    if (start > nums.size())
    {
        return;
    }
    for (int i = start; i <= nums.size(); i++)
    {
        combine.push_back(nums[i - 1]);
        backtraceSubsets(result, combine, nums, i + 1);
        combine.pop_back();
    }
}
vector<vector<int>> subsets(vector<int> &nums)
{
    std::sort(nums.begin(), nums.end());
    vector<vector<int>> result;
    vector<int> combine;
    backtraceSubsets(result, combine, nums, 1);
    return result;
}

// 单词搜索
bool dfs(vector<vector<char>> &board, const string &word, int i, int j, int index)
{
    if (index == word.length())
        return true;
    if ((i < 0 || i >= board.size()) || (j < 0 || j >= board[0].size()) || (board[i][j] != word[index]))
        return false;

    char temp = board[i][j];
    board[i][j] = '#'; // 标记这个位置已经访问

    vector<pair<int, int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // 定义四个方向的移动
    for (auto &dir : dirs)
    {
        int new_i = i + dir.first, new_j = j + dir.second;
        bool is_proper = dfs(board, word, new_i, new_j, index + 1); // 符合的话则会继续递归，不符合则会直接然回上一层递归

        if (is_proper)
        {
            return true;
        }
    }
    board[i][j] = temp; // 还原这个位置的字符
    return false;
}
bool exist(vector<vector<char>> &board, string word)
{
    int rows = board.size(), cols = board[0].size();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (dfs(board, word, i, j, 0))
            {
                return true;
            }
        }
    }
    return false;
}

// 只出现一次的数字
int findUnique(int a, int b, int c)
{
    return a ^ b ^ c;
}
int findSingleCow(vector<int> &nums)
{
    std::sort(nums.begin(), nums.end());

    for (int i = 0; i < nums.size() - 2; i += 3)
    {
        if (nums[i] != nums[i + 2])
        {
            return findUnique(nums[i], nums[i + 1], nums[i + 2]);
        }
    }
    return nums[nums.size() - 1];
}

// 只出现一次的数字Ⅱ
vector<int> findSingleCowsII(vector<int> &nums)
{
    std::unordered_map<int, int> count;
    vector<int> result;

    for (auto &num : nums)
    {
        count[num]++;
    }

    for (auto &single_val : count)
    {
        if (single_val.second == 1)
        {
            result.push_back(single_val.first);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// 实现单例模式
class Singleton
{
private:
    static Singleton *instance; // 获取类全局单例
    static std::mutex mutex;    // 获取类全局互斥锁对象

    Singleton() {}
    Singleton(const Singleton &) = delete;            // 禁止拷贝构造函数
    Singleton &operator=(const Singleton &) = delete; // 禁止赋值操作
public:
    static Singleton *getInstance() // 若想用函数访问其中的成员，如 Singleton::getInstance->instance，必须声明初始地址
    {
        if (instance == nullptr)
        {
            std::lock_guard<std::mutex> lock(mutex); // 使用lock_guard自动管理锁
            if (instance == nullptr)
            {
                instance = new Singleton();
            }
        }
        return instance;
    }
};

// 约瑟夫问题
bool validateCowCircle(vector<int> &enter, vector<int> &leave)
{
    std::list<int> cow;
    int direction;

    for (int i = 0; i < enter.size(); i++)
    {
        cow.push_back(enter[i]);
    }

    auto it = std::find(cow.begin(), cow.end(), leave[0]);

    if (it == cow.end())
    {
        return false;
    }
    if (leave[1] == *prev(it))
    {
        direction = 0; // 前一个元素匹配，方向设置为 0。
    }
    else if (leave[1] == *next(it))
    {
        direction = 1; // 后一个元素匹配，方向设置为 1
    }

    for (int j = 0; j < leave.size(); j++)
    {
        if (*it != leave[j])
        {
            return false;
        }
        while (*it == leave[j])
        {
            it = (direction == 0 ? prev(it) : next(it));
        }
    }
    return true;
}

// 合法的括号字符串
bool isValidString(string s)
{
    // 使用平衡计数方法
    int low = 0;
    int high = 0;

    for (char c : s)
    {
        if (c == '(')
        {
            low++;
            high++;
        }
        else if (c == ')')
        {
            if (low > 0)
                low--;
            high--;
        }
        else
        {
            if (low > 0)
                low--;
            high++;
        }

        if (high < 0)
            return false;
    }

    return low == 0;
}

// 数字编码成字符串
int numDecodings(std::string s)
{
    int n = s.size();
    if (s[0] == '0')
    {
        return 0;
    }

    std::vector<int> dp(n + 1, 0);
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= n; ++i)
    {
        if (s[i - 1] != '0')
        {
            dp[i] += dp[i - 1];
        }
        int twoDigits = (s[i - 2] - '0') * 10 + (s[i - 1] - '0');
        if (twoDigits >= 10 && twoDigits <= 26)
        {
            dp[i] += dp[i - 2];
        }
    }

    return dp[n];
}

// 非递减数列
bool checkPossibility(vector<int> &nums)
{
    if (nums.size() <= 2)
        return true;

    int count = 0; // 用于记录需要修改的次数

    for (int i = 1; i < nums.size() && count <= 1; i++)
    {
        if (nums[i - 1] > nums[i])
        {
            if (i - 2 >= 0 && nums[i - 2] > nums[i]) // 如果通过降低nums[i-1]解决，尝试提高nums[i]
            {
                if (i + 1 < nums.size() && nums[i - 1] > nums[i + 1]) // 如果提高nums[i]无法解决，则返回false
                {
                    return false;
                }
                nums[i] = nums[i - 1];
            }
            else if (i - 2 < 0 || nums[i - 2] <= nums[i]) // 如果能通过降低nums[i-1]解决
            {
                nums[i - 1] = nums[i];
            }

            count++; // 修改过一次
        }
    }

    return count <= 1;
}

// 最小的K个数(Top-k问题)
void heapify(int index, int size, std::vector<int> &heap)
{
    // 建堆过程
    int largest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    if (left < size && heap[left] > heap[largest])
    {
        largest = left; // 如果大于或者小于，那么交换坐标
    }

    if (right < size && heap[right] > heap[largest])
    {
        largest = right;
    }

    if (largest != index) // 如果坐标发生改变，则交换堆中数据保持堆结构
    {
        std::swap(heap[index], heap[largest]);
        heapify(largest, size, heap);
    }
}
std::vector<int> GetLeastNumbers_Solution(std::vector<int> &input, int k)
{
    // 如果是找出最小k个数值，则需要大根堆。反之，则需要小根堆
    if (k >= input.size())
        return input;

    std::vector<int> result(input.begin(), input.begin() + k);
    for (int i = k / 2 - 1; i >= 0; i--)
    {
        heapify(i, k, result);
    }

    for (size_t i = k; i < input.size(); i++)
    {
        if (input[i] < result[0])
        {
            result[0] = input[i];
            heapify(0, k, result);
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

// 删除有序数组中的重复项Ⅱ
int removeDuplicates(vector<int> &nums)
{
}

int maxSumIncreasingSubsequence(vector<int> &nums)
{
    int N = nums.size();
    if (N == 0)
        return 0;

    vector<int> dp = nums;

    for (int i = 1; i < N; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (nums[i] > nums[j])
            {
                dp[i] = max(dp[i], dp[j] + nums[i]);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}
int main()
{
    vector<int> nums = {3, 2, 6, 4, 5, 1};
    maxSumIncreasingSubsequence(nums);
    system("puase");
}
