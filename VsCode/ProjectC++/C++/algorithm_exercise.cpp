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

// 判断矩阵是否是 X 矩阵
bool CheckXMatrix(vector<vector<int>> &grid)
{
    int diagonal_e, sum;
    int width_length = grid.size();
    for (int i = 0; i < width_length; i++)
    {
        sum = 0;
        diagonal_e = grid[i][i] + grid[i][width_length - 1 - i];
        if (i == width_length - 1 - i)
            diagonal_e = grid[i][i];
        for (int j = 0; j < width_length; j++)
            sum += grid[i][j];
        if (sum != diagonal_e || grid[i][i] == 0 || grid[i][width_length - 1 - i] == 0)
            return false;
    }
    return true;
}

// 等差数列排序
void SequenceSort(int x)
{
    vector<int> &array = *new vector<int>();
    for (int i = 5; i >= 0; i--)
    {
        array.push_back((54 + i * x));
    }
    for (vector<int>::iterator it = array.begin(); it != array.end(); it++)
    {
        cout << *it << endl;
    }
}

// 计算日期到天数变换
void DateConversionDays()
{
    int mothdays[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int year, days, month;
    int n = 0;
    while (cin >> year >> month >> days)
    {
        for (int i = 0; i < month; i++)
        {
            n += mothdays[i];
        }
        n += days;
        if (month > 2 && ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0))
        {
            n += 1;
        }
    }
    cout << n << endl;
}

// 环元素的去除
void ElementJudgement()
{
    vector<int> array;
    int temp = 0;
    while (temp != -1)
    {
        cin >> temp;
        if (temp != -1)
        {
            array.push_back(temp);
        }
    }
    int flag = 0;
    for (int i = 3; array.size() > 1; i %= array.size())
    {
        flag = ((flag + 3) - 1) % array.size();
        array.erase(array.begin() + flag);
    }

    for (vector<int>::iterator it = array.begin(); it != array.end(); ++it)
    {
        cout << *it << " ";
    }
}

// 求约分因子个数
void DivisorMount(int x)
{
    int abs_x = abs(x);
    int prime_number = 0;
    double result = sqrt(abs_x);

    for (int i = 1; i <= result; i++)
    {
        if (abs_x % i == 0 && (double)(abs_x / i) != i)
            prime_number += 2;
        else if ((double)(abs_x / i) == i)
            prime_number += 1;
    }
    cout << prime_number;
}

// 约分个数的排序
void DivisorSort(int x)
{
    int abs_x = abs(x);
    int prime_number = 0;
    double result = sqrt(abs_x);
    vector<int> divisor;
    for (int i = 1; i <= result; i++)
    {
        if (abs_x % i == 0)
        {
            if ((double)(abs_x / i) != i)
            {
                prime_number += 2;
                divisor.push_back(x / i);
            }
            else
            {
                prime_number += 1;
            }
            divisor.push_back(i);
        }
    }
    if (prime_number >= 8)
    {
        sort(divisor.begin(), divisor.end(), greater<int>());
        divisor.erase(divisor.begin() + 8, divisor.end());
    }
    else
    {
        sort(divisor.begin(), divisor.end());
    }
    cout << "number:" << prime_number << endl;
    for (vector<int>::iterator it = divisor.begin(); it != divisor.end(); ++it)
    {
        cout << *it << " ";
    }
}

// 十进制转换九进制
void Conversion(int x)
{
    int ramainder = x;
    vector<int> v;
    while (ramainder >= 9)
    {
        v.insert(v.begin(), ramainder % 9);
        ramainder /= 9;
    }
    v.insert(v.begin(), ramainder);
    for (vector<int>::iterator it = v.begin(); it != v.end(); ++it)
    {
        cout << *it << " ";
    }
}

// 正整数n阶乘的约数
int factorial(int n)
{
    if (n == 0 || n == 1)
        return 1;
    else
        return n * factorial(n - 1);
}
vector<int> findDivisors(int n)
{
    vector<int> divisors;
    for (int i = 1; i <= sqrt(n); i++)
    {
        if (n % i == 0)
        {
            if (n / i == i)
            {
                divisors.push_back(i);
            }
            else
            {
                divisors.push_back(i);
                divisors.push_back(n / i);
            }
        }
    }
    sort(divisors.begin(), divisors.end());
    return divisors;
}

// 求数组中的两数和
void TwoSum()
{
    int target, complement;
    cin >> target;
    vector<int> array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    unordered_map<int, int> array_copy;

    for (size_t i = 0; i < array.size(); i++)
    {
        array_copy[array[i]] = i;
    }
    for (size_t j = 0; j < array.size(); j++)
    {
        complement = target - array[j];
        if (static_cast<size_t>(array_copy.count(complement)) && j <= static_cast<size_t>(array_copy[complement]))
        {
            cout << j << " " << array_copy[complement] << endl;
        }
    }
}

// 求一个数组的除下标外的所有元素和的一半
void ElementSum()
{
    vector<int> array_A = {1, 4, 56, 1, 6, 2};

    int sum_A, size;
    size = array_A.size();
    sum_A = 0;
    while (size > 0)
    {
        sum_A += array_A[--size];
    }
    vector<int> array_B;
    int pre_multiple = 0;
    for (size_t i = 0; i < array_A.size(); i++)
    {
        pre_multiple = 2 * (sum_A - array_A[i]);
        array_B.push_back(abs(pre_multiple));
    }
    for (vector<int>::iterator it = array_B.begin(); it != array_B.end(); ++it)
    {
        cout << *it << " ";
    }
}

// 有效的括号
bool isValid(string s)
{
    stack<char> st;
    int tail = s.length();

    if (tail <= 1)
        return false;
    while (tail--)
    {
        if (s[tail] == '}' || s[tail] == ')' || s[tail] == ']')
        {
            st.push(s[tail]);
        }
        else
        {
            if (st.empty())
                return false;
            char top = st.top();
            st.pop();
            if ((top == ')' && s[tail] != '(') || (top == '}' && s[tail] != '{') || (top == ']' && s[tail] != '['))
                return false;
        }
        //--tail;
    }
    if (!st.empty())
        return false;
    else
        return true;
}

// 用队列实现栈
class MyStack
{
public:
    queue<int> *queue_1;
    queue<int> *queue_2;
    MyStack()
    {
        queue_1 = new queue<int>();
        queue_2 = new queue<int>();
    }

    void push(int x)
    {
        queue<int> *empty = queue_1;
        queue<int> *not_empty = queue_2;
        if (queue_1->empty() && queue_2->empty())
        {
            not_empty->push(x);
            return;
        }
        if (!empty->empty())
        {
            empty = queue_2;
            not_empty = queue_1;
        }
        not_empty->push(x);
    }
    int pop()
    {
        queue<int> *empty = queue_1;
        queue<int> *not_empty = queue_2;
        if (!empty->empty())
        {
            empty = queue_2;
            not_empty = queue_1;
        }
        while (not_empty->size() > 1)
        {
            empty->push(not_empty->front());
            not_empty->pop();
        }
        int front = not_empty->front();
        not_empty->pop();
        return front;
    }

    int top()
    {
        queue<int> *empty = queue_1;
        queue<int> *not_empty = queue_2;
        if (!empty->empty())
        {
            empty = queue_2;
            not_empty = queue_1;
        }
        return not_empty->back();
    }

    bool empty()
    {
        if (queue_1->empty() && queue_2->empty())
            return true;
        else
            return false;
    }
};

// 用栈实现队
class MyQueue
{
public:
    stack<int> *pop_stack;
    stack<int> *push_stack;
    MyQueue()
    {
        pop_stack = new stack<int>();
        push_stack = new stack<int>();
    }

    void push(int x)
    {
        push_stack->push(x);
    }

    int pop()
    {
        if (pop_stack->empty() && !push_stack->empty())
        {
            while (!push_stack->empty())
            {
                pop_stack->push(push_stack->top());
                push_stack->pop();
            }
        }
        int front = pop_stack->top();
        pop_stack->pop();
        return front;
    }

    int peek()
    {
        if (pop_stack->empty() && !push_stack->empty())
        {
            while (!push_stack->empty())
            {
                pop_stack->push(push_stack->top());
                push_stack->pop();
            }
        }
        return pop_stack->top();
    }

    bool empty()
    {
        return (pop_stack->empty() && push_stack->empty());
    }
};

// 设计循环队列
struct Node
{
    int data;
    Node *next;
};
class MyCircularQueue
{
public:
    Node *circular_front;
    Node *circular_tail;
    int capacity;
    int size;
    MyCircularQueue(int k)
    {
        circular_front = nullptr;
        circular_tail = nullptr;
        capacity = k;
        size = 0;
    }
    bool enQueue(int value)
    {
        if (isFull())
            return false;
        Node *new_node = new Node();
        if (isEmpty())
        {
            circular_front = new_node;
        }
        else
        {
            circular_tail->next = new_node;
        }
        circular_tail = new_node;
        circular_tail->next = circular_front;
        circular_tail->data = value;
        ++size;
        return true;
    }
    bool deQueue()
    {
        if (isEmpty())
            return false;
        Node *temp = circular_front;
        if (circular_front == circular_tail)
        {
            circular_front = circular_front->next;
        }
        else
        {
            circular_front = circular_front->next;
            circular_tail->next = circular_front;
        }
        delete temp;
        temp = nullptr;
        --size;
        return true;
    }
    int Front()
    {
        if (isEmpty())
            return -1;
        else
            return circular_front->data;
    }
    int Rear()
    {
        if (isEmpty())
            return -1;
        else
            return circular_tail->data;
    }
    bool isEmpty()
    {
        return size == 0;
    }
    bool isFull()
    {
        return size == capacity;
    }
};

// 数组入栈
void arrayFlip()
{
    vector<int> vec;
    stack<int> stk;
    int x;
    while (cin >> x)
    {
        vec.push_back(x);
    }
    int count = vec.size();
    if (count > 13)
    {
        count = 13;
    }
    if (count <= 4)
    {
        for (int i = 0; i < count; i++)
        {
            stk.push(vec[i]);
        }
        vec.clear();
    }
    else
    {
        for (int i = 0; i < count; i++)
        {
            stk.push(vec[i]);
        }

        for (int j = 0; j < 4; j++)
        {
            stk.pop();
        }
        vec.clear();
        for (int k = 4; k < count; k++)
        {
            vec.push_back(stk.top());
            stk.pop();
        }
    }
}

// 砝码称重
void weight_weighing()
{
    const int N = 20;         // 砝码的最大数量
    const int M = 10000;      // 所有砝码重量之和的最大值
    int n;                    // 砝码的数量
    int w[N];                 // 砝码的重量
    int dp[N + 1][M * 2 + 1]; // 动态规划数组

    cin >> n;    // 输入砝码数量
    int sum = 0; // 计算所有砝码重量之和
    for (int i = 0; i < n; i++)
    {
        cin >> w[i]; // 输入每个砝码的重量
        sum += w[i];
    }
    // j是重量，i是某一个
    dp[0][sum] = 1; // 初始化状态
    for (int i = 1; i <= n; i++)
    { // 遍历每一个砝码
        for (int j = 0; j <= sum * 2; j++)
        { // 遍历每一个可能的重量
            if (dp[i - 1][j])
            {                 // 如果前i-1个砝码可以称出该重量
                dp[i][j] = 1; // 不使用第i个砝码
                if (j - w[i - 1] >= 0)
                {                            // 如果不越界
                    dp[i][j - w[i - 1]] = 1; // 使用第i个砝码放在左边
                }
                if (j + w[i - 1] <= sum * 2)
                {                            // 如果不越界
                    dp[i][j + w[i - 1]] = 1; // 使用第i个砝码放在右边
                }
            }
        }
    }
    int ans = -1; // 统计答案（减去了dp[n][sum])
    for (int j = sum + 1; j <= sum * 2; j++)
    {
        if (dp[n][j])
            ans++;
    }

    cout << ans << endl;
}

// 杨辉三角
vector<vector<int>> generate(int numRows)
{
    vector<vector<int>> dp(numRows);
    dp[0].push_back(1); // dp[0][0]=1;
    for (int i = 1; i < numRows; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            dp[i].push_back(0);
            if (j == 0)
            {
                dp[i][j] = dp[i - 1][j];
            }
            else if (i == j)
            {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }
    }
    return dp;
}

// 杨辉三角(滚动数组)
vector<int> getRow(int rowIndex)
{
    vector<int> pre, cur;
    for (int i = 0; i < rowIndex + 1; i++)
    {
        pre.resize(i + 1);
        pre[0] = 1;
        pre[i] = 1;
        for (int j = 1; j < i; j++)
        {
            pre[j] = cur[j - 1] + cur[j];
        }
        cur = pre;
    }
    return pre;
}

// 判断子序列
bool isSubsequence(string s, string t)
{
    int s_size = s.length() - 1, t_size = t.length() - 1;
    if (s_size > t_size)
        return false;
    while (s_size >= 0 && t_size >= 0)
    {
        if (s[s_size] == t[t_size])
        {
            s_size--;
        }
        t_size--;
    }
    return s_size < 0;
}

// 三步问题
int waysToStep(int n)
{
    if (n <= 2)
        return n;
    if (n == 3)
        return 4;
    int dp[n + 1];
    dp[0] = 0;
    dp[1] = 1;
    dp[2] = 2;
    dp[3] = 4;
    for (int i = 4; i < n; i++)
    {
        dp[i] = (((dp[i - 1] + dp[i - 2]) % 1000000007) + dp[i - 3]) % 1000000007;
    }
    return dp[n];
}

// 最大连续数组和
int maxSubArray(vector<int> &nums)
{
    if (nums.empty())
    {
        return 0;
    }
    int size = nums.size(), dp[size];
    dp[0] = nums[0];
    int maxSum = dp[0];

    for (int i = 1; i < size; i++)
    {
        dp[i] = max(nums[i], dp[i - 1] + nums[i]);
        maxSum = max(maxSum, dp[i]);
    }

    return maxSum;
    // Kadane解法
    //     if (nums.empty())
    //     {
    //         return 0;
    //     }
    //     int max_subarraySum = nums[0], max_wholeSum = nums[0], nums_size = nums.size();
    //     for (int i = 0; i < nums_size; i++)
    //     {
    //         max_subarraySum = max(max_subarraySum + nums[i], nums[i]);
    //         max_wholeSum = max(max_wholeSum, max_subarraySum);
    //     }
    //     return max_wholeSum;
    //
}

// 按摩师
int massage(vector<int> &nums)
{
    int n = nums.size();
    vector<int> dp(n);
    dp[0] = nums[0];
    dp[1] = nums[1];
    for (int i = 2; i < n; i++)
    {
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
    }
    return dp[n - 1];
}

// 翻转数位
int reverseBits(int num)
{
    unsigned int realNum = (unsigned int)num;
    int preBit = 0;
    int preTransed = 0;
    int maxTransed = 0;

    for (int i = 0; i < 32; i++)
    {
        int maxCur = 0;
        if ((realNum & ((unsigned int)1 << i)) != 0)
        {
            maxCur = preTransed + 1;
            preTransed = maxCur;
            preBit++;
        }
        else
        {
            maxCur = preBit + 1;
            preTransed = maxCur;
            preBit = 0;
        }

        maxTransed = max(maxTransed, maxCur);
    }
    return maxTransed;
}

// 比特计数
vector<int> countBits(int n)
{
    vector<int> dp(n + 1, 0);
    for (int i = 1; i < n + 1; i++)
    {
        dp[i] = dp[i >> 1] + (i & 1);
    }
    return dp;
}

// 下载插件
int leastMinutes(int n)
{
    vector<int> dp(n + 1);
    dp[1] = 1;
    for (int i = 2; i <= n; ++i)
        dp[i] = dp[(i + 1) / 2] + 1;
    return dp[n];
}

vector<int> order_preorder;
unordered_map<int, int> dic;
TreeNode *recur(int root, int left, int right)
{
    if (left > right)
        return nullptr;                                       // 递归终止
    TreeNode *node = new TreeNode(order_preorder[root]);      // 建立根节点
    int i = dic[order_preorder[root]];                        // 划分根节点、左子树、右子树
    node->left = recur(root + 1, left, i - 1);                // 开启左子树递归
    node->right = recur(root + 1 + (i - left), i + 1, right); // 开启右子树递归
    return node;                                              // 回溯返回根节点
}
// 通过中序前序重建二叉树
TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder)
{
    order_preorder = preorder;
    for (size_t i = 0; i < inorder.size(); i++)
        dic[inorder[i]] = i;
    return recur(0, 0, inorder.size() - 1);
}

// 利用队输入树的层序遍历节点
void sequenceTraversal(TreeNode *root)
{
    if (root == nullptr)
        return;

    queue<TreeNode *> BinaryTree;
    BinaryTree.push(root);

    while (!BinaryTree.empty())
    {
        TreeNode *front = BinaryTree.front();
        cout << front->val << " ";
        if (front->left != nullptr)
        {
            BinaryTree.push(front->left);
        }
        if (front->right != nullptr)
        {
            BinaryTree.push(front->left);
        }
        BinaryTree.pop();
    }
}

// 判断一个树是否是完全二叉树
bool isComplateBinaty(TreeNode *root)
{
    if (root == nullptr)
        return false;
    queue<TreeNode *> BinaryTree;
    BinaryTree.push(root);
    bool hasNull = false;

    while (!BinaryTree.empty())
    {
        TreeNode *front = BinaryTree.front();

        if (front == nullptr)
        {
            bool hasNull = true;
        }
        else
        {
            if (hasNull == true)
            {
                return false;
            }
            BinaryTree.push(front->left);
            BinaryTree.push(front->right);
        }
        BinaryTree.pop();
    }
    return hasNull;
}

// 二叉树的层序遍历
vector<vector<int>> levelOrder(TreeNode *root)
{
    queue<TreeNode *> tree;
    // queue<int> leve;
    vector<vector<int>> vv;

    if (root != nullptr)
        tree.push(root);
    while (!tree.empty())
    {
        vector<int> v;
        int depth_size = tree.size();
        for (int i = 0; i < depth_size; i++)
        {
            TreeNode *front = tree.front();
            if (front->left != nullptr)
            {
                tree.push(front->left);
            }
            if (front->right != nullptr)
            {
                tree.push(front->right);
            }
            v.push_back(front->val);
            tree.pop();
        }
        vv.push_back(v);
    }
    return vv;
}

// 查找树的公共祖先
bool Find(TreeNode *root, TreeNode *pq)
{
    if (root == nullptr)
        return false;
    return pq == root || Find(root->left, pq) || Find(root->right, pq);
}
TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q)
{
    if (root == nullptr)
        return nullptr;
    if (root == p || root == q)
        return root;

    bool qInleft, qInright, pInleft, pInright;
    qInleft = Find(root->left, q);
    qInright = !qInleft;
    pInleft = Find(root->left, p);
    pInright = !pInleft;

    if ((qInleft && pInright) || (pInleft && qInright))
        return root;
    if (qInleft && pInleft)
        return lowestCommonAncestor(root->left, p, q);
    if (qInright && pInright)
        return lowestCommonAncestor(root->right, p, q);
    return nullptr;
}

// 搜索树转换成循环双向链表
void DFS(TreeNode *cur, TreeNode *&prev)
{
    if (cur == nullptr)
        return;
    DFS(cur->left, prev);
    if (prev != nullptr)
        cur->left = prev;
    prev->right = cur;
    prev = cur;
    DFS(cur->right, prev);
}
TreeNode *treeToDoublyList(TreeNode *root)
{
    TreeNode *prev = nullptr;
    TreeNode *head = root;
    DFS(root, prev);
    while (head)
    {
        head = head->left;
    }
    return head;
}

// 迭代先序遍历二叉树
// 思路一
vector<int> preorderTraversal1(TreeNode *root)
{
    vector<int> ret;
    stack<TreeNode *> stk;
    TreeNode *cur = root;
    while (cur || !stk.empty())
    {
        while (cur)
        {
            ret.push_back(cur->val);
            stk.push(cur);
            cur = cur->left;
        }
        TreeNode *top = stk.top();
        stk.pop();
        cur = top->right;
    }
    return ret;
}
// 思路二
vector<int> preorderTraversal2(TreeNode *root)
{
    vector<int> ret;
    if (!root)
        return ret; // 空树直接返回
    stack<TreeNode *> s;
    s.push(root);
    while (!s.empty())
    {
        TreeNode *node = s.top();
        s.pop();
        ret.push_back(node->val);
        if (node->right)
            s.push(node->right);
        if (node->left)
            s.push(node->left);
    }
    return ret;
}

// 升序数组建立avl树
TreeNode *recur(vector<int> &nums, int left, int right)
{
    if (left > right)
        return nullptr;
    int mid = (left + right) / 2;
    TreeNode *node = new TreeNode(nums[mid]);
    node->left = recur(nums, left, mid - 1);
    node->right = recur(nums, mid + 1, right);
    return node;
}
TreeNode *sortedArrayToBST(vector<int> &nums)
{
    return recur(nums, 0, nums.size() - 1);
}

// 判断二叉树层级大小
int binaryTreeLevesize(TreeNode *root, int k)
{
    if (root == nullptr)
        return 0;
    if (k == 1)
        return 1;
    return binaryTreeLevesize(root->left, k - 1) + binaryTreeLevesize(root->right, k - 1);
}

// 获取树中的数字
int getNumber(TreeNode *root, vector<vector<int>> &ops)
{
    set<int> s;
    function<void(TreeNode *)> dfs = [&dfs, &s](TreeNode *node)
    {
        if (node != nullptr)
        {
            dfs(node->left);
            s.insert(node->val);
            dfs(node->right);
        }
    };
    dfs(root);
    int ret = 0;
    for (int i = ops.size() - 1; i >= 0; --i)
    {
        int type = ops[i][0];
        auto l = s.lower_bound(ops[i][1]), r = s.upper_bound(ops[i][2]);
        if (type == 1)
        {
            ret += distance(l, r);
        }
        s.erase(l, r);
    }
    return ret;
}

// 验证一棵树是否是二叉搜索树
bool _isValidBST(TreeNode *root, long long lower, long long upper)
{
    if (root == nullptr)
    {
        return true;
    }
    if (root->val <= lower || root->val >= upper)
    {
        return false;
    }
    return _isValidBST(root->left, lower, root->val) && _isValidBST(root->right, root->val, upper);
}

// 最长回文串
string longestPalindrome(string s)
{
    int n = s.size();
    if (n <= 1)
    {
        return s;
    }
    int maxarr = 1, start = 0;
    vector<vector<bool>> dp(n, vector<bool>(n, false));

    for (int i = 0; i < n; i++)
    {
        dp[i][i] = true;
    }
    for (int i = 0; i < n - 1; i++)
    {
        if (s[i] == s[i + 1])
        {
            dp[i][i + 1] = true;
            maxarr = 2;
            start = i;
        }
    }
    for (int len = 3; len <= n; len++)
    {
        for (int i = 0; i + len - 1 < n; i++)
        {
            int j = i + len - 1;
            if (s[i] == s[j] && dp[i + 1][j - 1])
            {
                dp[i][j] = true;
                maxarr = len;
                start = i;
            }
        }
    }
    return s.substr(start, maxarr);
}

// N 字形变换
string convert(string s, int numRows)
{
    if (numRows <= 1)
        return s;
    int after_corner = (numRows - 1) * 2, before_corner = 0;
    bool iscorner;
    string ret;
    for (int i = 0; i < numRows; i++)
    {
        size_t j = i;
        iscorner = true;
        while (j < s.size())
        {
            ret.push_back(s[j]);
            if (i == 0 || i == (numRows - 1))
            {
                j += (numRows - 1) * 2;
                continue;
            }
            iscorner = !iscorner;
            iscorner ? j += before_corner : j += after_corner;
        }
        after_corner -= 2;
        before_corner += 2;
    }
    return ret;
}

// 整数反转
int reverse(int x)
{
    int result = 0;
    int p = x;

    while (p != 0) // 正负
    {
        if (result > INT_MAX / 10 || result < INT_MIN / 10) // 事先判断
            return 0;

        result = result * 10 + p % 10;
        p /= 10;
    }
    return result;
}

// 字符串转换整数
int myAtoi(string s)
{
    int result = 0, sign = 1, count = 0;
    for (size_t i = 0; i < s.size(); i++)
    {

        if (count < 1)
        {
            if (s[i] == ' ')
                continue;
            s[i] == '-' ? sign = -1 : sign = 1;
            count++;
            if (s[i] == '+' || (s[i] == '-'))
                continue;
        }
        if (s[i] >= '0' && s[i] <= '9')
        {
            int digit = s[i] - '0';
            if (result > (INT_MAX - digit) / 10)
            {
                return sign == 1 ? INT_MAX : INT_MIN;
            }
            result = result * 10 + digit;
        }
        else
        {
            break;
        }
    }
    return result * sign;
}

// 盛最多水的容器
int maxArea(vector<int> &height)
{
    int n = height.size();
    int from = 0, after = n - 1, area = 0;
    for (size_t i = 0; from < after; i++)
    {
        int prev_area = area;

        if (height[from] < height[after])
        {
            area = (after - from) * height[from];
            from++;
        }
        else
        {
            area = (after - from) * height[after];
            after--;
        }
        area = max(area, prev_area);
    }
    return area;
}

// 整数转罗马数字
string intToRoman(int num)
{
    string result;
    map<int, string> Roman_number{{1, "I"}, {4, "IV"}, {5, "V"}, {9, "IX"}, {10, "X"}, {40, "XL"}, {50, "L"}, {90, "XC"}, {100, "C"}, {400, "CD"}, {500, "D"}, {900, "CM"}, {1000, "M"}};

    for (auto it = Roman_number.rbegin(); it != Roman_number.rend(); ++it)
    {
        while (num >= it->first)
        {
            num -= it->first;
            result += it->second;
        }
    }
    return result;
}

// 两数之和
vector<int> twoSum(vector<int> &nums, int target)
{
    unordered_map<int, int> numIndexMap;

    for (int i = 0; i < nums.size(); ++i)
    {
        int complement = target - nums[i];
        if (numIndexMap.find(complement) != numIndexMap.end())
        {
            return {numIndexMap[complement], i};
        }
        numIndexMap[nums[i]] = i;
    }
    return {};
}

// 三数之和
vector<vector<int>> threeSum(vector<int> &nums)
{
    vector<vector<int>> ret;
    unordered_map<int, int> numIndexMap;
    sort(nums.begin(), nums.end());

    for (int i = 0; i < nums.size(); ++i)
    {
        if (i > 0 && nums[i] == nums[i - 1])
            continue;
        int target = -nums[i];
        for (int j = i + 1; j < nums.size(); ++j)
        {
            int complement = target - nums[j];
            if (numIndexMap.find(complement) != numIndexMap.end())
            {
                vector<int> legal = {nums[i], complement, nums[j]};
                ret.push_back(legal);
                while (j + 1 < nums.size() && nums[j] == nums[j + 1])
                {
                    ++j;
                }
            }
            numIndexMap[nums[j]] = j;
        }
        numIndexMap.clear();
    }

    return ret;
}

// 最接近的三数之和
int threeSumClosest(vector<int> &nums, int target)
{
    std::sort(nums.begin(), nums.end());
    int n = nums.size(), pre_limit = nums[0] + nums[1] + nums[2];

    for (int i = 0; i < n - 2; i++)
    {
        if (i > 0 && nums[i] == nums[i - 1])
            continue;
        int left = i + 1, right = n - 1, limit = 0;
        while (left < right)
        {
            limit = (nums[i] + left + right);
            if (abs(target - pre_limit) - abs(target - limit) > 0)
            {
                pre_limit = limit;
            }
            if (target > limit)
            {
                left++;
            }
            else
            {
                right--;
            }
        }
    }
    return pre_limit;
}

// 电话号码的符号组合
vector<string> _letterCombinations(int depth, const string &digits, const unordered_map<int, vector<char>> &mapping)
{
    vector<string> ret;

    if (depth >= digits.size())
    {
        ret.push_back("");
        return ret;
    }

    auto it = mapping.find(digits[depth] - '0');
    for (size_t i = 0; i < it->second.size(); i++)
    {
        vector<string> combinations = _letterCombinations(depth + 1, digits, mapping);
        for (string j : combinations)
        {
            ret.push_back(it->second[i] + j);
        }
    }

    return ret;
}
vector<string> letterCombinations(string digits)
{
    if (digits.empty())
    {
        return {};
    }
    vector<string> result;
    unordered_map<int, vector<char>> mapping = {
        {2, {'a', 'b', 'c'}},
        {3, {'d', 'e', 'f'}},
        {4, {'g', 'h', 'i'}},
        {5, {'j', 'k', 'l'}},
        {6, {'m', 'n', 'o'}},
        {7, {'p', 'q', 'r', 's'}},
        {8, {'t', 'u', 'v'}},
        {9, {'w', 'x', 'y', 'z'}}};
    result = _letterCombinations(0, digits, mapping);
    return result;
}

// 四数之和
vector<vector<int>> fourSum(vector<int> &nums, int target)
{
    std::sort(nums.begin(), nums.end());
    int n = nums.size();
    vector<vector<int>> ret;

    if (n < 4)
        return ret;

    for (int i = 0; i < n - 3; i++)
    {
        if (i > 0 && nums[i] == nums[i - 1])
            continue;

        for (int j = n - 1; j > i; j--)
        {
            if (j < n - 1 && nums[j] == nums[j + 1])
                continue;

            int left = i + 1, right = j - 1;
            while (left < right)
            {
                if (left > i + 1 && nums[left] == nums[left - 1])
                {
                    left++;
                    continue;
                }
                if (right < j - 1 && nums[right] == nums[right + 1])
                {
                    right--;
                    continue;
                }
                long sum = (long)nums[i] + nums[j] + nums[left] + nums[right];
                if (sum == target)
                {
                    ret.push_back({nums[i], nums[j], nums[left], nums[right]});
                    left++;
                    right--;
                }
                else if (sum < target)
                {
                    left++;
                }
                else
                {
                    right--;
                }
            }
        }
    }
    return ret;
}

// 复杂链表的复制
class LinkedList
{
public:
    struct Node
    {
        int val;
        Node *next;
        Node *random;
        Node(int _val)
        {
            val = _val;
            next = NULL;
            random = NULL;
        }
    };
    Node *copyRandomList(Node *head)
    {
        if (head == NULL)
        {
            return NULL;
        }
        Node *cur = head;
        unordered_map<Node *, Node *> map;
        while (cur != NULL)
        {
            Node *newHead = new Node(1);
            newHead->val = cur->val;
            map[cur] = newHead;
            cur = cur->next;
        }
        cur = head;

        while (cur != NULL)
        {
            map[cur]->next = map[cur->next];
            map[cur]->random = map[cur->random];
            cur = cur->next;
        }
        return map[head];
    }
};

// 删除链表的的倒数第N个节点
ListNode *removeNthFromEnd(ListNode *head, int n)
{
    if (n == 0 || head == nullptr)
        return head;

    int first = 1;
    unordered_map<int, ListNode *> map;
    ListNode *cur = head;
    while (cur != nullptr)
    {
        map[first++] = cur;
        cur = cur->next;
    }
    int location = abs(first - n);
    ListNode *node = map.find(location)->second;
    if (location <= 1)
    {
        ListNode *temp_node = head->next;
        head->next = nullptr;
        delete head;
        head = temp_node;
    }
    else
    {
        ListNode *prev_node = map.find(location - 1)->second;
        prev_node->next = node->next;
        node->next = nullptr;
        delete node;
    }
    return head;
}

// 括号生成
vector<string> generateParenthesis(int n)
{
    vector<vector<string>> dp(n + 1);
    dp[0] = {""};

    for (int i = 1; i <= n; i++)
    {
        for (int j = 0; j <= i - 1; j++)
        {
            for (string &left : dp[j])
            {
                for (string &right : dp[i - j - 1])
                    dp[i].push_back('(' + left + ')' + right);
            }
        }
    }
    return dp[n];
}

// 两数相除
int divide(int dividend, int divisor)
{
    if (dividend == INT_MIN && divisor == -1)
        return INT_MAX;

    int sign = ((dividend > 0 && divisor > 0) || (dividend < 0 && divisor < 0)) ? 1 : -1;
    long long absDividend = abs((long long)dividend);
    long long absDivisor = abs((long long)divisor);
    long long quotient = 0;

    while (absDividend >= absDivisor)
    {
        long long temp = absDivisor, multiple = 1;
        while (absDividend >= (temp << 1))
        {
            temp <<= 1;
            multiple <<= 1;
        }

        absDividend -= temp;
        quotient += multiple;
    }
    return (quotient * -1 <= INT_MIN) ? INT_MIN : ((quotient > INT_MAX) ? INT_MAX : sign * quotient);
}

// 两两交换链表中的节点
ListNode *_swapPairs(ListNode *head, int count)
{
    if (head == nullptr)
        return nullptr;
    ListNode *swap_node = _swapPairs(head->next, count + 1);

    if (count % 2 == 1 && head->next != nullptr)
    {
        ListNode *next_node = head->next;
        head->next = next_node->next;
        next_node->next = head;
        return next_node;
    }
    head->next = swap_node;
    return head;
}
ListNode *swapPairs(ListNode *head)
{
    if (head == nullptr)
        return nullptr;
    if (head->next == nullptr)
        return head;
    ListNode *guard = new ListNode();
    guard->next = head;
    return _swapPairs(guard, 0)->next;
}

// 搜索旋转排序数组
int search(vector<int> &nums, int target)
{
    int mid, left = 0, right = nums.size() - 1;
    while (left <= right)
    {
        mid = (left + right) / 2;
        if (nums[mid] == target)
            return nums[mid];
        if (nums[left] <= nums[mid])
        {
            (target < nums[mid]) && (nums[left] <= target) ? right = mid - 1 : left = mid + 1;
        }
        else
        {
            (target > nums[mid]) && (target <= nums[right]) ? left = mid + 1 : right = mid - 1;
        }
    }
    return -1;
}

// 在排序数组中查找元素的第一个和最后一个位置
vector<int> searchRange(vector<int> &nums, int target)
{
    int mid, front = -1, left = 0, right = nums.size();

    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (nums[mid] == target)
        {
            front = mid;
            right = mid - 1;
        }
        else if (target < nums[mid] && target >= nums[left])
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }

    int back = -1;
    mid = 0, left = 0, right = nums.size();

    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (nums[mid] == target)
        {
            back = mid;
            left = mid + 1;
        }
        else if (target < nums[mid] && target >= nums[left])
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return {front, back};
}

// 有效的数独
bool isValidSudoku(vector<vector<char>> &board)
{
    unordered_set<char> rowSet;
    unordered_set<char> colSet;
    unordered_set<char> subgridSet;

    for (int i = 0; i < 9; i++)
    {

        for (int j = 0; j < 9; j++)
        {
            if (board[i][j] != '.' && rowSet.count(board[i][j]))
            {
                return false;
            }
            rowSet.insert(board[i][j]);

            if (board[j][i] != '.' && colSet.count(board[j][i]))
            {
                return false;
            }
            colSet.insert(board[j][i]);

            int subgridRow = (i / 3) * 3;
            int subgridCol = (i % 3) * 3;
            if (board[subgridRow + j / 3][subgridCol + j % 3] != '.' && subgridSet.count(board[subgridRow + j / 3][subgridCol + j % 3]))
            {
                return false;
            }
            subgridSet.insert(board[subgridRow + j / 3][subgridCol + j % 3]);
        }
        rowSet.clear();
        colSet.clear();
        subgridSet.clear();
    }
    return true;
}

// 外观数列
string countAndSay(int n)
{
    vector<string> ret(n + 1);
    ret[0] = {"0"};
    ret[1] = {"1"};

    for (int i = 2; i <= n; i++)
    {
        int string_count = 0;
        for (size_t j = 1; j <= ret[i - 1].size(); j++)
        {
            string_count++;

            if (ret[i - 1][j] != ret[i - 1][j - 1] || (j == ret[i - 1].size()))
            {
                ret[i] += std::to_string(string_count) + ret[i - 1][j - 1];
                string_count = 0;
            }
        }
    }
    return ret[n];
}

// 组合总和
void backtrack(vector<int> &candidates, int target, int start, vector<int> combina, vector<vector<int>> &result)
{
    // 无重复元素的整数数组 c 和目标整数 t，找出可以使数字和为目标数的所有不同组合
    if (target == 0) // 如果找到相应的组合，则入栈然后退出
    {
        result.push_back(combina);
        return;
    }
    for (size_t i = start; i < candidates.size(); i++) // 否则继续挑选
    {
        if (candidates[i] > target) // 剪枝
        {
            break;
        }
        combina.push_back(candidates[i]); // 挑选可能的组合
        backtrack(candidates, target - candidates[i], start, combina, result);

        combina.pop_back(); // 弹出一位数字继续循环挑选
    }
}
vector<vector<int>> combinationSum(vector<int> &candidates, int target)
{
    vector<vector<int>> result;
    vector<int> current;
    std::sort(candidates.begin(), candidates.end());

    backtrack(candidates, target, 0, current, result);
    return result;
}
// 组合总和Ⅱ
void backtrack2(vector<int> &candidates, int target, int start, vector<int> &current, vector<vector<int>> &result)
{
    if (target == 0)
    {
        result.push_back(current);
        return;
    }
    for (size_t i = start; i < candidates.size(); i++)
    {
        if (candidates[i] > target)
        {
            break;
        }
        current.push_back(candidates[i]);
        backtrack2(candidates, target - candidates[i], i + 1, current, result); // i + 1 表示为每次递增，以保证选择数字不会相等
        current.pop_back();

        while (i < candidates.size() - 1 && candidates[i] == candidates[i + 1]) // 过滤相同数字
        {
            i++;
        }
    }
}
vector<vector<int>> combinationSum2(vector<int> &candidates, int target)
{
    vector<vector<int>> result;
    vector<int> current;

    std::sort(candidates.begin(), candidates.end());
    backtrack2(candidates, target, 0, current, result);
    return result;
}

// 字符串多项式相乘
string multiply(std::string num1, std::string num2)
{
    string long_num = num1.size() < num2.size() ? num2 : num1;     // 找出位数多
    string short_num = num1.size() < num2.size() ? num1 : num2;    // 找出位处少
    std::deque<int> result(long_num.size() + short_num.size(), 0); // 先定义结果为两者长度之和

    for (int i = short_num.size() - 1; i >= 0; i--)
    {
        int carry = 0;
        for (int j = long_num.size() - 1; j >= 0; j--)
        {
            int product = (short_num[i] - '0') * (long_num[j] - '0') + result[i + j + 1] + carry; // result[i + j + 1] 中存储了上一位相乘的结果
            result[i + j + 1] = product % 10;                                                     // 存储个位数
            carry = product / 10;
        }
        result[i] += carry; // 如果相乘时超出了个位数，则将进位位也存储
    }

    int start = 0;
    while (start < result.size() - 1 && result[start] == 0)
    {
        start++;
    }

    string result_str;
    for (int i = start; i < result.size(); i++)
    {
        result_str.push_back(result[i] + '0');
    }
    return result_str.empty() ? "0" : result_str;
}

// 全排列
void dfs(vector<int> &nums, vector<int> &permutation, vector<int> &used, vector<vector<int>> &res)
{
    if (permutation.size() == nums.size())
    {
        res.push_back(permutation);
        return;
    }
    for (int i = 0; i < nums.size(); ++i)
    {
        if (used[i] == 1)
        {
            continue;
        }

        used[i] = 1;
        permutation.push_back(nums[i]);
        dfs(nums, permutation, used, res);

        permutation.pop_back();
        used[i] = 0;
    }
}
// 全排列 Ⅱ
void DFS(vector<int> &nums, vector<int> &use, vector<vector<int>> &res, vector<int> &adaptation)
{
    // 全排列问题可以用回溯来解决，问题的关键在于如何在一个数组中选出某个数字并且这个数字下次不会被选择
    // 解决这个问题可以用一个use[]数组标记选择的数字，当该数字被选择时就在裁剪阶段判断并去掉
    if (nums.size() == adaptation.size())
    {
        res.push_back(adaptation); // 如果符合就加入 result
        return;
    }

    for (int i = 0; i < nums.size(); i++)
    {
        if (use[i] == 1 || (i > 0 && nums[i] == nums[i - 1] && use[i - 1] == 0)) // 裁剪阶段。use[i] == 1表示已选择,
        {                                                                        // nums[i] == nums[i - 1]表示上一个数字必须和下一个不同, i > 0 限制 i 取值范围
            continue;                                                            //  use[i - 1] == 0表示 上一个未被选择
        }

        use[i] = 1;
        adaptation.push_back(nums[i]);
        DFS(nums, use, res, adaptation);

        adaptation.pop_back();
        use[i] = 0;
    }
}
vector<vector<int>> permute(vector<int> &nums)
{
    vector<vector<int>> res;
    vector<int> permutation;
    vector<int> used(nums.size(), 0);
    std::sort(nums.begin(), nums.end());

    dfs(nums, permutation, used, res);
    return res;
}

// 字典序中的下一个排列
void nextPermutation(vector<int> &nums)
{
    // 这道题的核心思想在于找出尾部第一个突变的顺序(i, j), 之后再找出与 i 相比尾部第一个比 i 大的数值
    // 然后将它们交换, 之后将 i+1 到 尾部排序即可
    // 合理性在于第一个顺序必然是最小突变值，第一个比 i 大的数值必然是最接近 i 的数值
    int tail = nums.size() - 1;
    if (nums.size() - 1 <= 0)
        return;

    while (tail > 0 && nums[tail] <= nums[tail - 1])
    {
        tail--; // 找出突变的顺序(i, j)
    }
    for (int i = nums.size() - 1; i >= 0 && tail > 0; i--)
    {

        if (nums[tail - 1] < nums[i] && i > tail - 1)
        {
            std::swap(nums[tail - 1], nums[i]);         // 交换 与i 相比尾部第一个比 i 大的数值
            std::sort(nums.begin() + tail, nums.end()); // 重新排序
            return;
        }
    }
    sort(nums.begin(), nums.end());
}

// 实现正整数十进制转换二进制
string ToBinary(long n)
{
    string binaryStr = "";
    if (n == 0)
    {
        return "0";
    }
    while (n > 0)
    {
        binaryStr = std::to_string(n & 1) + binaryStr; // n & 1 表示找出 n 的二进制中为 0 的位数
        n = n >> 1;                                    // 将 n 除 2, 缩进 n 值
    }
    return binaryStr;
}

// 实现 Pow(x, n)
double myPow(double x, int n)
{
    double base = x;
    double result = 1;
    long long abs_n = abs((long long)n); // 避免 x = 1.0, n = INT_MIN 时的情况
    string binaryStr = ToBinary(abs_n);

    for (int i = binaryStr.size() - 1; i >= 0; --i)
    {
        if (binaryStr[i] != '0')
        {
            result = result * base; // 当二进制位不为 0 时，将 base 累乘到结果上
        }

        base *= base; // base 一直保持平方
    }
    return n < 0 ? 1 / result : result; // 处理幂为负数
}

// 旋转图像
void rotate(vector<vector<int>> &matrix)
{
    // 水平线交换
    int matrixRowSize = matrix.size();
    for (int i = 0; i < matrixRowSize; i++)
    {
        for (int j = i; j < matrix[i].size(); j++)
        {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
    // 垂直交换
    for (int i = 0; i < matrixRowSize; i++)
    {
        int left = 0, right = matrixRowSize - 1;
        while (left < right)
        {
            swap(matrix[i][left], matrix[i][right]);
            left++;
            right--;
        }
    }
}

// 字母异位词分组
vector<vector<string>> groupAnagrams(vector<string> &strs)
{
    unordered_map<string, vector<string>> map;
    for (auto &str : strs)
    {
        // 排序字符得到一个共同性质的 key
        // 根据排序后的 key 分组
        string key = str;
        sort(key.begin(), key.end());
        map[key].push_back(str);
    }

    vector<vector<string>> result;
    for (auto &pair : map)
    {
        result.push_back(pair.second);
    }
    return result;
}

// 跳跃游戏
bool canJump(vector<int> &nums)
{
    // 跳跃游戏的解决分为两步，首先要保证能够达到末尾，即i + nums[i] >= lastGoodIndex
    // 其次再用贪心算法每次缩短末尾lastGoodIndex的值，最终lastGoodIndex == 0表示能跳到末尾
    int n = nums.size();
    int lastGoodIndex = n - 1;

    for (int i = n - 2; i >= 0; i--)
    {

        if (i + nums[i] >= lastGoodIndex)
        {
            lastGoodIndex = i;
        }
    }
    return lastGoodIndex == 0;
}
// 跳跃游戏Ⅱ
int jump(vector<int> &nums)
{
    // 求跳跃游戏的最短跳跃次数可以用区间内的贪心算法
    // 即在区间border_val内寻找最大跳跃位置max_reach，并将最大跳跃位置[max_reach,i + max_reac]又重新构成区间border_val内
    // 持续统计区间的次数，即步数，最终叠加成最短跳跃次数
    if (nums.size() <= 1)
    {
        return 0;
    }

    int border_val = 0, steps = 0, max_reach = 0;

    for (int i = 0; i < nums.size() - 1; i++)
    {
        max_reach = max(max_reach, i + nums[i]);

        if (i == border_val)
        {
            border_val = max_reach;
            steps++;
        }
    }
    return steps;
}

// 移除元素
int removeElement(vector<int> &nums, int val)
{
    // 双指针
    int n = nums.size();
    int left = 0;
    for (int right = 0; right < n; right++)
    {
        if (nums[right] != val)
        {
            nums[left] = nums[right];
            left++;
        }
    }
    return left;
}

// 回文数
bool isPalindrome(int x)
{
    if (x <= 0)
    {
        return false;
    }
    string s = to_string(x);

    int left = 0, right = s.size() - 1;
    while (left < right)
    {
        if (s[left] != s[right])
        {
            return false;
        }
        left++;
        right--;
    }
    return true;
}

// 快速排序
void QuickSort(vector<vector<int>> &intervals, int left, int right)
{
    if (left > right)
        return;
    int pivot = intervals[left][0], i = left, j = right; // 定义基准值为pivot等于序列[left]

    while (i != j)
    {
        while (intervals[j][0] >= pivot && i < j)
        {
            j--; // 先从右边往左边查找第一个比基准值pivot小的数值
        }
        while (intervals[i][0] <= pivot && i < j)
        {
            i++; // 在从左边往右边查找第一个比基准值pivot大的数值
        }
        if (i < j)
        {
            std::swap(intervals[i], intervals[j]); // 如果i和j不相碰时交换
        }
        // 不断交换，直到分割为两半，右边全部值比基准值pivot大，左边全部值比基准值pivot小
    }

    std::swap(intervals[left], intervals[i]); // 将两半的中点值即相碰的i和j与最左边的pivot交换
    QuickSort(intervals, left, i - 1);        // 递归左边
    QuickSort(intervals, i + 1, right);       // 递归右边
}
// 合并区间
vector<vector<int>> QuickSortMerge(vector<vector<int>> &intervals)
{
    if (intervals.empty())
    {
        return intervals;
    }
    QuickSort(intervals, 0, intervals.size() - 1);
    vector<vector<int>> merge_gather;

    merge_gather.push_back(intervals[0]);      // 定义一个集合并且插入第一个元素
    for (int i = 1; i < intervals.size(); i++) // 如果在intervals中的start大于集合最后一个元素的end，则说明不重叠。否则重叠，更新数值
    {

        if (intervals[i].front() > merge_gather.back().back())
        {
            merge_gather.push_back(intervals[i]);
        }
        else
        {
            merge_gather.back().back() = std::max(merge_gather.back().back(), intervals[i].back());
        }
    }

    return merge_gather;
}

ListNode *SortListMerge(ListNode *l1, ListNode *l2)
{
    ListNode dummy;
    ListNode *tail = &dummy; // 定义一个新链表来用于保存比较的值

    while (l1 != nullptr && l2 != nullptr)
    {
        if (l1->val < l2->val) // 暴力比较，谁小谁先放入新链表中
        {
            tail->next = l1;
            l1 = l1->next;
        }
        else
        {
            tail->next = l2;
            l2 = l2->next;
        }
        tail = tail->next;
    }
    if (l1 != nullptr) // 链接末尾节点的值
        tail->next = l1;
    if (l2 != nullptr)
        tail->next = l2;

    return dummy.next;
}
// 归并排序
ListNode *SortList(ListNode *head)
{
    if (head == nullptr || head->next == nullptr)
    {
        return head;
    }
    ListNode *slow = head, *fast = head, *prev = nullptr;

    while (fast != nullptr && fast->next != nullptr)
    {
        // 利用快慢指针寻找中点
        prev = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    prev->next = nullptr;

    ListNode *l1 = SortList(head); // 左分割
    ListNode *l2 = SortList(slow); // 右分割

    return SortListMerge(l1, l2); // 将一个序列有序排列
}
//  合并链表
ListNode *mergeTwoLists(ListNode *list1, ListNode *list2)
{
    if (list1 == nullptr && list2 == nullptr)
    {
        return nullptr;
    }
    if (list1 == nullptr)
        return list2;
    if (list2 == nullptr)
        return list1;

    ListNode *cur = list1;
    while (cur->next != nullptr)
    {
        cur = cur->next;
    }
    cur->next = list2;
    ListNode *result = SortList(list1);

    return result;
}

// 插入区间
vector<vector<int>> insert(vector<vector<int>> &intervals, vector<int> &newInterval)
{
    if (intervals.empty())
    {
        intervals.push_back(newInterval);
        return intervals;
    }
    for (int i = 0; i < intervals.size(); i++)
    {
        if (newInterval.front() >= intervals[i].front() && (i == intervals.size() - 1 || newInterval.front() < intervals[i + 1].front()))
        {
            intervals.insert(intervals.begin() + (i + 1), newInterval);
            break;
        }
        if (newInterval.front() < intervals[i].front())
        {
            intervals.insert(intervals.begin(), newInterval);
            break;
        }
    }
    vector<vector<int>> result(1, intervals[0]);
    for (int i = 1; i < intervals.size(); i++)
    {
        if (result.back().back() < intervals[i].front())
        {
            result.push_back(intervals[i]);
        }
        else
        {
            result.back().back() = std::max(result.back().back(), intervals[i].back());
        }
    }
    return result;
}

// 螺旋矩阵
vector<int> spiralOrder(vector<vector<int>> &matrix)
{
    vector<int> result;
    int top = 0, left = 0, right = matrix.front().size() - 1, bottom = matrix.size() - 1;

    while (left <= right && top <= bottom)
    {
        for (int row = left; row <= right; row++)
        {
            result.push_back(matrix[top][row]);
        }
        top++;

        for (int column = top; column <= bottom; column++)
        {
            result.push_back(matrix[column][right]);
        }
        right--;

        if (top <= bottom)
        {
            for (int row = right; row >= left; row--)
            {
                result.push_back(matrix[bottom][row]);
            }
            bottom--;
        }

        if (left <= right)
        {
            for (int column = bottom; column >= top; column--)
            {
                result.push_back(matrix[column][left]);
            }
            left++;
        }
    }
    return result;
}
// 螺旋矩阵Ⅱ
vector<vector<int>> generateMatrix(int n)
{
    int top = 0, left = 0, right = n - 1, bottom = n - 1, val = 0;
    vector<vector<int>> result(n, vector<int>(n, 0));

    while (top <= bottom && left <= right && val <= std::pow(n, 2))
    {
        for (int row = left; row <= right; row++)
        {
            result[top][row] = ++val;
        }
        top++;

        for (int column = top; column <= bottom; column++)
        {

            result[column][right] = ++val;
        }
        right--;

        if (top <= bottom)
        {
            for (int row = right; row >= left; row--)
            {
                result[bottom][row] = ++val;
            }
            bottom--;
        }

        if (left <= right)
        {
            for (int column = bottom; column >= top; column--)
            {
                result[column][left] = ++val;
            }
            left++;
        }
    }
    return result;
}

// 旋转链表
ListNode *rotateRight(ListNode *head, int k)
{
    ListNode *cur = head, *tail = nullptr, *new_head = nullptr;
    int length = 0, shift = 0;

    while (cur != nullptr)
    {
        length++;
        cur = cur->next;
        if (cur != nullptr && cur->next == nullptr)
        {
            tail = cur;
        }
    }
    shift = k % length;

    if (shift == 0)
    {
        return head;
    }

    cur = head;
    for (int i = 0; i < (length - shift) - 1; i++)
    {
        cur = cur->next;
    }

    new_head = cur->next;
    tail->next = head;
    cur->next = nullptr;

    return new_head;
}

// 不同路径
int uniquePaths(int m, int n)
{
    int result[m][n];

    for (int row = 0; row < m; row++)
    {
        for (int column = 0; column < n; column++)
        {
            if (row == 0 || column == 0)
            {
                result[row][column] = 1;
            }
            else
            {
                result[row][column] = result[row - 1][column] + result[row][column - 1];
            }
        }
    }
    return result[m - 1][n - 1];
}
//  不同路径Ⅱ
int uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid)
{
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();

    // 如果起点或终点有障碍物，直接返回0。
    if (obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1)
    {
        return 0;
    }

    obstacleGrid[0][0] = 1; // 起点，路径数为1

    // 初始化第一行和第一列
    for (int i = 1; i < m; ++i)
    {
        obstacleGrid[i][0] = (obstacleGrid[i][0] == 0 && obstacleGrid[i - 1][0] == 1) ? 1 : 0;
    }
    for (int j = 1; j < n; ++j)
    {
        obstacleGrid[0][j] = (obstacleGrid[0][j] == 0 && obstacleGrid[0][j - 1] == 1) ? 1 : 0;
    }

    for (int i = 1; i < m; ++i)
    {
        for (int j = 1; j < n; ++j)
        {
            if (obstacleGrid[i][j] == 0)
            {
                obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]; // 只有在当前单元格没有障碍物时才进行计算
            }
            else
            {
                obstacleGrid[i][j] = 0; // 有障碍物的格子路径数为 0
            }
        }
    }

    return obstacleGrid[m - 1][n - 1]; // 返回终点的路径数
}

// 最小路径和
int minPathSum(vector<vector<int>> &grid)
{
    int m = grid.size(), n = grid.front().size();
    int result[m][n];
    result[0][0] = grid[0][0];

    for (int row = 0; row < m; row++)
    {

        for (int column = 0; column < n; column++)
        {
            if (row == 0 && column == 0)
                continue;

            if ((row == 0 || column == 0))
            {
                row == 0 ? result[row][column] = result[row][column - 1] + grid[row][column] : result[row][column] = result[row - 1][column] + grid[row][column];
            }
            else
            {
                result[row][column] = std::min(result[row - 1][column] + grid[row][column], result[row][column - 1] + grid[row][column]);
            }
        }
    }
    return result[m - 1][n - 1];
}

// 在含有数字、'('')'、字符的字符串中计数字符个数
int getNum(const string &s, int &i)
{
    int num = 0;
    while (i < s.length() && (s[i] >= '0' && s[i] <= '9'))
    {
        num = num * 10 + (s[i] - '0');
        ++i;
    }
    --i; // 回到最后一个数字，每次在for循环中都会跳到开始的字符
    return num == 0 ? 1 : num;
}
string countLetters(const string &s)
{
    stack<map<char, int>> st; // 使用栈和哈希表，当是数字是则用一个map<char, int>持续入栈
    st.push({});              // 当找到'(' ,则也用一个map<char, int>将后面的数字持续入栈
                              // 直到找到')' ,将'('栈中的数据添加到已有的数字栈中，然后删除'('栈
    for (int i = 0; i < s.length(); ++i)
    {
        if (isalpha(s[i]))
        {
            int start = i;
            ++i;
            int num = getNum(s, i);
            st.top()[s[start]] += num;
        }
        else if (s[i] == '(') // 遇到'('，在栈顶添加新map
        {
            st.push({});
        }
        else if (s[i] == ')')
        {
            ++i;
            int num = getNum(s, i);
            auto temp = st.top();
            st.pop();
            for (auto &it : temp)
            {
                st.top()[it.first] += it.second * num;
            }
        }
    }
    string result;
    for (auto &it : st.top())
    {
        result += it.first + to_string(it.second);
    }
    return result;
}

// 大数相加
string solve(string s, string t)
{
    string long_nuns = s.size() > t.size() ? s : t;
    string short_nums = s.size() > t.size() ? t : s;
    int carry = 0, n = long_nuns.size(), m = short_nums.size();
    string result;

    while (m > 0)
    {
        int num = (short_nums[m - 1] - '0' + long_nuns[n - 1] - '0' + carry) % 10;
        carry = (short_nums[m - 1] - '0' + long_nuns[n - 1] - '0' + carry) / 10;

        result.push_back(num + '0');
        m--;
        n--;
    }
    while (n > 0)
    {
        int num = (long_nuns[n - 1] - '0' + carry) % 10;
        carry = (long_nuns[n - 1] - '0' + carry) / 10;

        result.push_back(num + '0');
        n--;
    }

    if (carry == 1)
        result.push_back('1');
    std::reverse(result.begin(), result.end());
    return result;
}

// 编辑距离(莱文斯坦距离)
int minDistance(string word1, string word2)
{
    string long_str = word1.size() > word2.size() ? word1 : word2;
    string short_str = word1.size() > word2.size() ? word2 : word1;

    vector<vector<int>> dp(short_str.size() + 1, vector<int>(long_str.size() + 1, 0));

    for (int i = 0; i < dp[0].size(); ++i)
    {
        dp[0][i] = i;
    }
    for (int j = 0; j < dp.size(); ++j)
    {
        dp[j][0] = j;
    }

    for (int y = 1; y <= short_str.size(); y++)
    {
        for (int x = 1; x <= long_str.size(); x++)
        {
            if (short_str[y - 1] == long_str[x - 1])
            {
                dp[y][x] = dp[y - 1][x - 1];
            }
            else
            {
                dp[y][x] = std::min({dp[y][x - 1] + 1, dp[y - 1][x] + 1, dp[y - 1][x - 1] + 1});
            }
        }
    }
    return dp[short_str.size()][long_str.size()];
}

//  矩阵置零(零矩阵变换)
void _setZeroes(vector<vector<int>> &matrix, const vector<int> memory)
{
    for (int i = 0; i < matrix[memory.front()].size(); i++)
    {
        matrix[memory.front()][i] = 0;
    }
    for (int j = 0; j < matrix.size(); j++)
    {
        matrix[j][memory.back()] = 0;
    }
}
void setZeroes(vector<vector<int>> &matrix)
{
    int row_size = matrix.size();
    int column_size = matrix[0].size();
    matrix[0].size();
    vector<vector<int>> memory;
    vector<vector<int>> first_row_memory;

    for (int i = 0; i < column_size; i++)
    {
        if (matrix[0][i] == 0)
        {
            first_row_memory.push_back({0, i});
        }
    }
    for (int j = 1; j < row_size; j++)
    {
        if (matrix[j][0] == 0)
        {
            first_row_memory.push_back({j, 0});
        }
    }

    for (int x = 1; x < row_size; x++)
    {
        for (int y = 1; y < column_size; y++)
        {
            if (matrix[x][y] == 0)
            {
                memory.push_back({x, y});
            }
        }
    }
    for (int i = 0; i < memory.size(); i++)
    {
        _setZeroes(matrix, memory[i]);
    }

    if (!first_row_memory.empty())
    {
        for (int i = 0; i < first_row_memory.size(); i++)
        {
            _setZeroes(matrix, first_row_memory[i]);
        };
    }
}
