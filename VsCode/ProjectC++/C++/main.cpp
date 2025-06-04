#include "my_libraries.h"

// 单链表
//  template<class ListType>
//  class ListNode {
//  public:
//	ListType data;
//	ListNode* next;
//	void size(ListNode* phead);
//	void Print(ListNode* phead);
//	void push_back(ListNode** phead, ListType x);
//	void push_front(ListNode** phead, ListType x);
//	void insetr(ListNode** phead, int position, ListType x);
//	void pop_back(ListNode** phead);
//	void pop_front(ListNode** phead);
//	void clear(ListNode** phead, int position);
//  };
//  template<class ListType>
//  void ListNode<ListType>::size(ListNode* phead) {
//	int sum = 0;
//	if (phead == NULL) {
//		cout << 0 << endl;
//	}
//	else
//	{
//		ListNode* tail = phead;
//		while (tail != NULL)
//		{
//			tail = tail->next;
//			sum++;
//		}
//		cout << sum << endl;
//	}
//  }
//  template<class ListType>
//  void ListNode<ListType>::Print(ListNode* phead) {
//	ListNode* cur = phead;
//	if (phead == NULL)
//		cout << "null";
//	while (cur != NULL) {
//		cout << cur->data << " ";
//		cur = cur->next;
//	}
//	cout << endl;
//  }
////增
// template<class ListType>
// void ListNode<ListType>::push_back(ListNode** phead, ListType x) {
//	ListNode<ListType>* node = new ListNode<ListType>();
//	node->data = x;
//	node->next = NULL;
//
//	if (*phead == NULL) {
//		*phead = node;
//	}
//	else {
//		ListNode* tail = *phead;
//		while (tail->next != NULL) {
//			tail = tail->next;
//		}
//		tail->next = node;
//	}
// }
// template<class ListType>
// void ListNode<ListType>::push_front(ListNode** phead, ListType x) {
//	ListNode<ListType>* node = new ListNode();
//	node->data = x;
//	node->next = NULL;
//
//	node->next = *phead;
//	*phead = node;
// }
// template<class ListType>
// void ListNode<ListType>::insetr(ListNode** phead, int position, ListType x) {
//	ListNode<ListType>* node = new ListNode<ListType>();
//	node->data = x;
//	node->next = NULL;
//	ListNode* tail = *phead;
//	if (*phead == NULL) {
//		*phead = node;
//	}
//	else if (position == 0)
//	{
//		node->next = *phead;
//		*phead = node;
//	}
//	else {
//		for (int i = 0; i < position - 1; i++)
//			tail = tail->next;
//		node->next = tail->next;
//		tail->next = node;
//	}
// }
////删
// template<class ListType>
// void ListNode<ListType>::pop_back(ListNode** phead) {
//	assert(*phead != NULL);
//	if ((*phead)->next == NULL){
//		delete *phead;
//		*phead == NULL;
//	}
//	else{
//		ListNode* prev = NULL;
//		ListNode* cur = *phead;
//		while (cur->next!= NULL) {
//			prev = cur;
//			cur = cur->next;
//		}
//		delete cur;
//		cur = NULL;
//		prev->next = NULL;
//	}
// }
// template<class ListType>
// void ListNode<ListType>::pop_front(ListNode** phead) {
//	assert(*phead != NULL);
//	ListNode* cur = *phead;
//	*phead = cur->next;
//	delete cur;
//	cur = NULL;
// }
// template<class ListType>
// void ListNode<ListType>::clear(ListNode** phead, int position) {
//	assert(*phead != NULL);
//
//	ListNode* cur = *phead;
//	ListNode* prev=NULL;
//	if (position == 0) {
//		*phead = cur->next;
//		delete cur;
//		cur = NULL;
//	}
//	else {
//		for (int i = 0; i < position; i++) {
//			prev = cur;
//			cur = cur->next;
//		}
//		prev->next = cur->next;
//		delete cur;
//		cur = NULL;
//	}
// }
//
//
// int main() {
//	ListNode<int> listObject;
//	ListNode<int>* listPoint = NULL;
//	listObject.push_back(&listPoint, 10);
//	listObject.push_back(&listPoint, 20);
//	listObject.push_front(&listPoint, 30);
//	listObject.push_front(&listPoint, 40);
//	listObject.push_back(&listPoint, 50);
//	listObject.push_back(&listPoint, 60);
//	listObject.size(listPoint);
//	cout<<"删除前:"<<endl;
//	listObject.Print(listPoint);
//	listObject.pop_front(&listPoint);
//	listObject.pop_back(&listPoint);
//	/*listObject.pop_back(&listPoint);*/
//	cout << "删除后:" << endl;
//	listObject.Print(listPoint);
//	listObject.clear(&listPoint, 2);
//	cout << "删除后:" << endl;
//	listObject.Print(listPoint);
//
// }

// 循环双链表
//  class List
//  {
//  public:
//  	int val;
//  	List *prev;
//  	List *next;
//  	List() : val(0), prev(nullptr), next(nullptr) {}
//  	List(int i) : val(0), prev(nullptr), next(nullptr) {}
//  	List *init();				   // 初始化
//  	List *find(List *, int);	   // 查找
//  	int size(List *);			   // 计算大小
//  	void print(List *);			   // 输出
//  	void push_back(List *, int);   // 尾增
//  	void push_front(List *, int);  // 头增
//  	void insert(List *, int, int); // 插入
//  	void pop_back(List *);		   // 尾删
//  	void pop_front(List *);		   // 头删
//  	void remove(List *, int);	   // 删除
//  	void cleared(List *);		   // 清空
//  };
//  void* clearedRecur(List* phead,List *head){
//  	if(phead->next=head){
//  		return nullptr;
//  	}
//  	clearedRecur(phead->next,head);
//  	delete phead->next;
//  	phead->next=nullptr;
//  }
//  List *List::init()
//  {
//  	List *sentry = new List();
//  	sentry->next = sentry;
//  	sentry->prev = sentry;
//  	return sentry;
//  }
//  List *List::find(List *phead, int count)
//  {
//  	// assert(count < size(phead));
//  	if (phead->next == nullptr)
//  		return nullptr;
//  	int i = 0;
//  	List *cur = phead->next;
//  	unordered_map<int, List *> map;
//  	while (cur != phead)
//  	{
//  		map[i] = cur;
//  		cur = cur->next;
//  		i++;
//  	}
//  	return map[count];
//  }
//  int List::size(List *phead)
//  {
//  	/*if (phead->next == phead)
//  	{
//  		return 0;
//  	}
//  	return 1 + size(phead->next);*/
//  }
//  void List::print(List *phead)
//  {
//  	List *cur = phead->next;
//  	while (cur != phead)
//  	{
//  		cout << cur->val << " ";
//  		cur = cur->next;
//  	}
//  	cout << endl;
//  }
//  void List::push_back(List *phead, int x)
//  {
//  	List *newNode = new List();
//  	List *tail = phead->prev;
//  	newNode->val = x;
//  	tail->next = newNode;
//  	newNode->prev = tail;
//  	newNode->next = phead;
//  	phead->prev = newNode;
//  }
//  void List::push_front(List *phead, int x)
//  {
//  	List *newNode = new List();
//  	newNode->val = x;
//  	newNode->next = phead->next;
//  	newNode->prev = phead;
//  	phead->next->prev = newNode;
//  	phead->next = newNode;
//  }
//  void List::insert(List *phead, int count, int x)
//  {
//  	List *p = find(phead, count);
//  	// List *p=(List*)(&turnUp-offsetof(List,val));
//  	List *newNode = new List();
//  	newNode->val = x;
//  	newNode->next = p;
//  	newNode->prev = p->prev;
//  	p->prev->next = newNode;
//  	p->prev = newNode;
//  }
//  void List::pop_front(List *phead)
//  {
//  	assert(phead->next != nullptr);
//  	List *head = phead->next;
//  	head->next->prev = phead;
//  	phead->next = head->next;
//  	delete head;
//  	head = nullptr;
//  }
//  void List::pop_back(List *phead)
//  {
//  	assert(phead != nullptr);
//  	List *tail = phead->prev;
//  	tail->prev->next = phead;
//  	phead->prev = tail->prev;
//  	delete tail;
//  	tail = nullptr;
//  }
//  void List::remove(List *phead, int count)
//  {
//  	assert(phead->next != nullptr);
//  	List *findNode = find(phead, count);
//  	findNode->next->prev = findNode->prev;
//  	findNode->prev->next = findNode->next;
//  	delete findNode;
//  	findNode = nullptr;
//  }
//  void List::cleared(List * phead){
//  	clearedRecur(phead,phead);
//  }
//  int main()
//  {
//  	List p;
//  	List *object = p.init();
//  	p.push_back(object, 1);
//  	p.push_back(object, 2);
//  	p.push_back(object, 5);
//  	p.push_back(object, 6); // 尾节点
//  	p.push_front(object, 9);
//  	p.push_front(object, 7); // 头节点
//  	p.insert(object, 1, 10);
//  	p.insert(object, 1, 3);
//  	p.pop_front(object);
//  	p.pop_back(object);
//  	p.remove(object, 5);
//  	p.cleared(object);
//  	p.print(object);
//  	cout << p.find(object, 3)->val << endl;
//  	 system("pause");
//  }

// 堆
// template <class heaptype>
// class Heap
// {
// public:
//     heaptype *arr_point;
//     int size;
//     int capacity;
//     Heap() : size(0), capacity(0), arr_point(nullptr) {}
//     Heap(int val) : size(val), capacity(val), arr_point(nullptr) {}
//
//     void swap(heaptype &x, heaptype &y)
//     {
//         heaptype temp = x;
//         x = y;
//         y = temp;
//     }
//     void Print(Heap *);                                 // 输出
//     virtual void heapInit(Heap *, heaptype *, int) = 0; // 初始化
//     virtual void heapPush(Heap *, int) = 0;             // 插入
//     virtual void heapPop(Heap *) = 0;                   // 删除
//     virtual void heapSort(Heap *) = 0;                  // 堆降序
// };
// template <class heaptype>
// class LessHeap : public Heap<int>
// {
// public:
//     LessHeap() : Heap() {}
//     LessHeap(int val) : Heap(val) {}
//      // 向下调整算法，用来构建堆
//     void adjustDown(heaptype *outher_arr, int i, Heap<heaptype> *new_arr)
//     {
//         int perent, child;
//         perent = i;
//         child = 2 * i + 1;
//         while ((outher_arr[child] < outher_arr[perent] || outher_arr[child + 1] < outher_arr[perent]) && child < new_arr->size)
//         {
//             if (child + 1 < new_arr->size && outher_arr[child] > outher_arr[child + 1])
//                 child++;
//             if (outher_arr[child] > outher_arr[perent])
//                 break;
//             swap(outher_arr[perent], outher_arr[child]);
//             perent = child;
//             child = 2 * perent + 1;
//         }
//     }
//      //向上调整算法，用来插入数值
//     void adjustUp(heaptype *outher_arr, int n)
//     {
//         int perent, child;
//         child = n - 1;
//         perent = (child - 1) / 2;
//         while (child > 0)
//         {
//             if (outher_arr[perent] > outher_arr[child])
//             {
//                 swap(outher_arr[perent], outher_arr[child]);
//             }
//             else
//             {
//                 break;
//             }
//             child = perent;
//             perent = (child - 1) / 2;
//         }
//     }
//     virtual void heapInit(Heap<heaptype> *heap, heaptype *outher_arr, int n)
//     {
//         heap->arr_point = new int[n];
//         memcpy(heap->arr_point, outher_arr, sizeof(heaptype) * n);
//         for (int i = (n - 2) / 2; i >= 0; i--)
//         {
//             adjustDown(heap->arr_point, i, heap); //使用向下调整算法完成小根堆或者大根堆
//         }
//     }
//     virtual void heapPush(Heap<heaptype> *heap, int x)
//     {
//         if (heap->size == heap->capacity)
//         {
//             heap->capacity *= 2;
//             heap->arr_point = (int *)realloc(heap->arr_point, (heap->capacity) * sizeof(int));
//             heap->arr_point[heap->size++] = x;
//         }
//         adjustUp(heap->arr_point, heap->size);
//     }
//     virtual void heapPop(Heap<heaptype> *heap)
//     {
//         assert(heap->size);
//         swap(heap->arr_point[0], heap->arr_point[--(heap->size)]);
//         heap->arr_point[heap->size] = 0;
//         adjustDown(heap->arr_point, 0, heap);
//     }
//     virtual void heapSort(Heap<heaptype> *heap)
//     {
//         int temp = heap->size;
//         while (heap->size > 0)
//         {
//             swap(heap->arr_point[0], heap->arr_point[heap->size - 1]);
//             heap->size--;
//             adjustDown(heap->arr_point, 0, heap);
//         }
//         heap->size = temp;
//     }
// };
// template <class heaptype>
// class GreatHeap : public Heap<int>
// {
// public:
//     GreatHeap() : Heap() {}
//     GreatHeap(int val) : Heap(val) {}
//
//     void adjustDown(heaptype *outher_arr, int i, Heap<heaptype> *new_arr)
//     {
//         int perent, child;
//         perent = i;
//         child = 2 * i + 1;
//         while ((outher_arr[child] > outher_arr[perent] || outher_arr[child + 1] > outher_arr[perent]) && child < new_arr->size)
//         {
//             if (child + 1 < new_arr->size && outher_arr[child] < outher_arr[child + 1])
//                 child++;
//             if (outher_arr[child] < outher_arr[perent])
//                 break;
//             swap(outher_arr[perent], outher_arr[child]);
//             perent = child;
//             child = 2 * perent + 1;
//         }
//     }
//     void adjustUp(heaptype *outher_arr, int n)
//     {
//         int perent, child;
//         child = n - 1;
//         perent = (child - 1) / 2;
//         while (child > 0)
//         {
//             if (outher_arr[perent] > outher_arr[child])
//             {
//                 swap(outher_arr[perent], outher_arr[child]);
//             }
//             else
//             {
//                 break;
//             }
//             child = perent;
//             perent = (child - 1) / 2;
//         }
//     }
//     virtual void heapInit(Heap<heaptype> *heap, heaptype *outher_arr, int n)
//     {
//         heap->arr_point = new heaptype[n];
//         memcpy(heap->arr_point, outher_arr, sizeof(heaptype) * n);
//         for (int i = (n - 2) / 2; i >= 0; i--)
//         {
//             adjustDown(heap->arr_point, i, heap);
//         }
//     }
//     virtual void heapPush(Heap<heaptype> *heap, int x)
//     {
//         if (heap->size == heap->capacity)
//         {
//             heap->capacity *= 2;
//             heap->arr_point = (int *)realloc(heap->arr_point, (heap->capacity) * sizeof(int));
//             heap->arr_point[heap->size++] = x;
//         }
//         adjustUp(heap->arr_point, heap->size);
//     }
//     virtual void heapPop(Heap<heaptype> *heap)
//     {
//         assert(heap->size);
//         swap(heap->arr_point[0], heap->arr_point[--(heap->size)]);
//         heap->arr_point[heap->size] = 0;
//         adjustDown(heap->arr_point, 0, heap);
//     }
//     virtual void heapSort(Heap<heaptype> *heap)
//     {
//         int temp = heap->size;
//         while (heap->size > 0)
//         {
//             swap(heap->arr_point[0], heap->arr_point[heap->size - 1]);
//             heap->size--;
//             adjustDown(heap->arr_point, 0, heap);
//         }
//         heap->size = temp;
//     }
// };
//
// template <class heaptype>
// void Heap<heaptype>::Print(Heap *heap)
// {
//     for (int i = 0; i < heap->size; i++)
//     {
//         cout << heap->arr_point[i] << " ";
//     }
//     cout << endl;
// }
// int main()
// {
//     Heap<int> *lessheap = new LessHeap<int>(10);
//     Heap<int> *greatheap = new GreatHeap<int>(10);
//     int arr[] = {19, 17, 15, 13, 1, 3, 5, 7, 9, 11};
//     int n = sizeof(arr) / sizeof(int);
//
//     lessheap->heapInit(lessheap, arr, n);
//     // lessheap->heapPush(lessheap, 9);
//     // lessheap->heapPop(lessheap);
//     lessheap->heapSort(lessheap);
//     lessheap->Print(lessheap);
//     greatheap->heapInit(greatheap, arr, n);
//     greatheap->heapSort(greatheap);
//     greatheap->Print(greatheap);
//
//     // system("pause");
// }

// 栈
// template <class StackType>
// class Stack
// {
// public:
//     vector<StackType> *stack_array;
//     int top;
//     Stack() : stack_array(nullptr) {}
//     void stack_initial();       // 初始化
//     void print();               // 输出
//     void stack_push(StackType); // 增
//     void stack_pop();           // 删
//     void stack_destroy();       // 销毁
// };
// template <class StackType>
// void Stack<StackType>::print()
// {
//     if (stack_array == nullptr)
//     {
//         cout << "null" << endl;
//         return;
//     }
//     for (auto it = stack_array->begin(); it != stack_array->end(); ++it)
//     {
//         cout << *it << " ";
//     }
//     cout << endl;
// }
// template <class StackType>
// void Stack<StackType>::stack_initial()
// {
//     top = 0;
//     vector<StackType> *array = new vector<StackType>();
//     stack_array = array;
// }
// template <class StackType>
// void Stack<StackType>::stack_push(StackType x)
// {
//     // assert(this);
//     stack_array->push_back(x);
//     top++;
// }
// template <class StackType>
// void Stack<StackType>::stack_pop()
// {
//     assert(stack_array->size() >= 0);
//     stack_array->pop_back();
//     top--;
// }
// template <class StackType>
// void Stack<StackType>::stack_destroy()
// {
//     assert(stack_array);
//     stack_array->clear();
//     delete stack_array;
//     stack_array = nullptr;
// }

// 队
// template <class type>
// class QueueNode
// {
// public:
//     QueueNode *next;
//     type data;
// };
// template <class type>
// class Queue
// {
// public:
//     QueueNode<type> *head;//维护头节点
//     QueueNode<type> *tail;//维护尾节点
//     void queueInitial();  // 初始化
//     void queuePush(type); // 插入
//     void queuePop();      // 删除
//     int queueSize();      // 计算长度
//     bool queueEmpty();    // 判断空
//     void queueDestroy();  // 销毁
//     type queueFront();    // 返回头
//     type queueBack();     // 返回尾
// };
// template <class type>
// void Queue<type>::queueInitial()
// {
//     // asser(this);
//     head = nullptr;
//     tail = nullptr;
// }
// template <class type>
// void Queue<type>::queuePush(type x)
// {
//     QueueNode<type> *node = new QueueNode<type>();
//     if (!head)
//         head = node;
//     else
//         tail->next = node;
//     tail = node;
//     tail->data = x;
//     tail->next = nullptr;
// }
// template <class type>
// void Queue<type>::queuePop()
// {
//     assert(!queueEmpty());
//     if (head->next == nullptr)
//         tail = nullptr;
//     QueueNode<type> *record = head->next;
//     delete head;
//     //head = nullptr;
//     head = record;
// }
// template <class type>
// bool Queue<type>::queueEmpty()
// {
//     return head == nullptr;
// }
// template <class type>
// int Queue<type>::queueSize()
// {
//     if (head == nullptr)
//         return 0;
//     QueueNode<type> *cur = head;
//     int flag = 0;
//     while (cur)
//     {
//         cur = cur->next;
//         ++flag;
//     }
//     return flag;
// }
// template <class type>
// void Queue<type>::queueDestroy()
// {
//     assert(head);
//     int size = queueSize();
//     while (size--)
//         queuePop();
// }
// template <class type>
// type Queue<type>::queueFront()
// {
//     assert(head);
//     return head->data;
// }
// template <class type>
// type Queue<type>::queueBack()
// {
//     assert(head);
//     return tail->data;
// }

// 搜索树
// template <class T>
// struct BSTreeNode
// {
// public:
//     BSTreeNode<T> *left, *right;
//     T value;
//     BSTreeNode(T val) : value(val), left(nullptr), right(nullptr) {}
// };
// template <class T>
// class BSTree
// {
// public:
//     typedef BSTreeNode<T> node;
//     BSTree() : root(nullptr) {}
//     // 删除
//     void _erase(node *cur, node *parent)
//     {
//         if (cur->left == nullptr)
//         {
//             // 左为空
//             if (parent->right == cur)
//                 parent->right = cur->right;
//             else
//                 parent->left = cur->right;
//             delete cur;
//         }
//         else if (cur->right == nullptr)
//         {
//             // 右为空
//             if (parent->left == cur)
//                 parent->left = cur->left;
//             else
//                 parent->right = cur->left;
//             delete cur;
//         }
//         else
//         {
//             // 左右不为空
//             node *right_min_parent = cur;
//             node *right_min = cur->right;
//             while (right_min->left != nullptr)
//             {
//                 right_min_parent = right_min;
//                 right_min = right_min->left;
//             }
//             // 将待删节点和右边最小替换，则退化成情况1
//             cur->value = right_min->value;
//             if (right_min_parent->right == right_min)
//                 right_min_parent->right = right_min->right;
//             else
//                 right_min_parent->left = right_min->right;
//             delete right_min;
//         }
//     }
//     bool erase(const T &value)
//     {
//         node *parent = nullptr;
//         node *cur = root;
//         while (cur)
//         {
//             if (cur->value < value)
//             {
//                 parent = cur;
//                 cur = cur->right;
//             }
//             else if (cur->value > value)
//             {
//                 parent = cur;
//                 cur = cur->left;
//             }
//             else
//             {
//                 // 找到当前节点，开始删除
//                 // 此时有四种情况：1.左为空，2.右为空，3.左右不为空，4.左右为空
//                 if (root->left == nullptr && root->right == nullptr)
//                 {
//                     // 左右为空的情况
//                     delete root;
//                     root = nullptr;
//                     return true;
//                 }
//                 _erase(cur, parent);
//                 return true;
//             }
//         }
//         return false;
//     }
//     // 查找
//     bool Find(T value)
//     {
//         node *cur = root;
//         while (cur != nullptr)
//         {
//             if (value > cur->value)
//             {
//                 cur = cur->right;
//             }
//             else if (value < cur->value)
//             {
//                 cur = cur->left;
//             }
//             else
//             {
//                 return true;
//             }
//         }
//         return false;
//     }
//     // 前序输出
//     void _perorder(node *root)
//     {
//         if (root != nullptr)
//         {
//             cout << root->value << " ";
//             _perorder(root->left);
//             _perorder(root->right);
//         }
//         // else
//         // {
//         //     cout << "null"
//         //          << " ";
//         // }
//     }
//     void perorder()
//     {
//         _perorder(root);
//     }
//     // 插入
//     bool insert(T value)
//     {
//         if (root == nullptr)
//         {
//             root = new node(value);
//             return true;
//         }
//         node **cur = &root;
//         while (*cur)
//         {
//             if (value < (*cur)->value)
//             {
//                 cur = &((*cur)->left);
//             }
//             else if (value > (*cur)->value)
//             {
//                 cur = &((*cur)->right);
//             }
//             else
//             {
//                 return false;
//             }
//         }
//         *cur = new node(value);
//         return true;
//     }
//     // 销毁
//     void destory(node *root)
//     {
//         if (root == nullptr)
//             return;
//         destory(root->left);
//         destory(root->right);
//         delete root;
//         root = nullptr;
//     }
//     // 重载operator=
//     bool operator=(const vector<int> &other)
//     {
//         if (root != nullptr)
//             destory(root);
//         for (int i = 0; i < other.size(); i++)
//         {
//             insert(other[i]);
//         }
//         return true;
//     }
// private:
//     node *root;
// };

// avl树
template <class k, class v>
struct AvlTreeNode
{
    AvlTreeNode<k, v> *left;
    AvlTreeNode<k, v> *right;
    AvlTreeNode<k, v> *parent;

    int _bf;
    pair<k, v> kv;

    AvlTreeNode() : left(nullptr), right(nullptr), parent(nullptr), _bf(0) {}
    AvlTreeNode(pair<k, v> value) : left(nullptr), right(nullptr), parent(nullptr), _bf(0), kv(value) {}
};
template <class k, class v>
class AvlTree
{
public:
    typedef AvlTreeNode<k, v> node;

    void RotateL(node *parent) // 左单旋算法
    {
        node *subR = parent->right;
        node *subRL = subR->left;

        parent->right = subRL;
        if (subRL)
            subR->parent = parent;

        subR->left = parent;
        node *ppMode = parent->parent;

        parent->parent = subR;
        if (root == parent)
        {
            root = subR;
            subR->parent = nullptr;
        }
        else
        {
            if (ppMode->left == subR)
                ppMode->left = subR;
            else
                ppMode->right = subR;

            subR->parent = ppMode;
        }
        parent->_bf = subR->_bf = 0;
    }
    void RotateR(node *parent) // 右单旋算法
    {
        node *subL = parent->left;
        node *subLR = subL->right;
        node *ppNode = parent->parent;

        parent->left = subLR;
        if (subLR)
            subLR->parent = parent;
        subL->right = parent;
        parent->parent = subL;

        if (root == parent)
        {
            root = subL;
            subL->parent = nullptr;
        }
        else
        {
            if (ppNode->left == parent)
                ppNode->left = subL;
            else
                ppNode->right = subL;
        }
        parent->_bf = subL->_bf = 0;
    }
    bool Insert(const pair<k, v> &kv); // 插入

private:
    node *root;
};

template <class k, class v>
bool AvlTree<k, v>::Insert(const pair<k, v> &_kv)
{
    if (root == nullptr)
    {
        root = new node(_kv);
        return true;
    }

    node *parent = nullptr;
    node *cur = root;
    while (cur)
    {
        // 查找
        if (cur->kv.first > _kv.first)
        {
            parent = cur;
            cur = cur->right;
        }
        else if (cur->kv.first < _kv.first)
        {
            parent = cur;
            cur = cur->left;
        }
        else
        {
            return false;
        }
        // 判断在左子树还是右子树
        cur = new node(_kv);
        if (parent->kv.first > _kv.first)
        {
            parent->left = cur;
            cur->parent = parent;
        }
        else
        {
            parent->right = cur;
            cur->parent = parent;
        }
        // 更新平衡因子
        while (parent)
        {
            if (cur == parent->right)
                parent->_bf++;
            else
                parent->_bf--;

            if (parent->_bf == 0)
            {
                break;
            }
            else if (parent->_bf == 1 || parent->bf == -1)
            {
                cur = parent;
                parent = parent->parent;
            }
            else if (parent->_bf == 2 || parent->bf == -2)
            {
            }
        }

        return true;
    }
}

// 固定内存池的实现
// template <int ObjectSize, int ObjectNum = 20>
// class MemPool
// {
// private:
//     struct FreeNode
//     {
//         FreeNode *next;        // 表示内存块中节点的下一个地址
//         char data[ObjectSize]; // 以char字节来进行内存对齐使得结构体大小变成2*ObjectSize*sizeof(char)
//     };
//     struct MemBlock
//     {
//         MemBlock *next;           // 表示下一个内存块的地址
//         FreeNode data[ObjectNum]; // 表示一个内存块中有多少节点,data[]表示一个节点中的对象
//     };
//     FreeNode *freeNodeHead; // 空闲节点中的首地址
//     MemBlock *memBlockHead; // 内存块中的首地址
// public:
//     Mempool()
//     {
//         freeNodeHead = nullptr;
//         memBlockHead = nullptr;
//     }
//     ~MemPool()
//     {
//         MemBlock *cur;
//         while (memBlockHead)
//         {
//             // 删除内存块
//             cur = memBlockHead->next;
//             delete memBlockHead;
//             memBlockHead = cur;
//         }
//     }
//     void *Malloc();
//     void Free(void *);
// };
// template <int ObjectSize, int ObjectNum>
// void *MemPool<ObjectSize, ObjectNum>::Malloc()
// {
//     if (freeNodeHead == nullptr)
//     {
//         MemBlock *newBlock = new MemBlock(); // 创造一个内存块，里面有data[ObjectNum]个节点
//         newBlock->next = nullptr;            // 将构建的内存块的下一个地址置成null
//         freeNodeHead = &newBlock->data[0]; // 指定块中节点的首地址
//         for (int i = 1; i < ObjectNum; i++)
//         {
//             newBlock->data[i - 1].next = &newBlock->data[i]; // 将所有节点以单链表链接
//         }
//         newBlock->data[ObjectNum - 1].next = nullptr; // 将块中最后节点置成null
//         /*以上为初始化块中数据*/
//         if (memBlockHead == nullptr)
//         {
//             // 如果只有一个块，则将新申请的块作为第一个块
//             memBlockHead = newBlock;
//         }
//         else
//         {
//             // 块节点满后，将新申请的块头插到旧块前
//             newBlock->next = memBlockHead;
//             memBlockHead = newBlock;
//         }
//         /*以上为初始化块与块*/
//     }
//     void *freeNode = freeNodeHead;
//     freeNodeHead = freeNodeHead->next;
//     return freeNode;
// }
// template <int ObjectSize, int ObjectNum>
// void MemPool<ObjectSize, ObjectNum>::Free(void *p)
// {
//     FreeNode *pNode = (FreeNode *)p;
//     pNode->next = freeNodeHead;
//     /*将送进来的节点断链并放到头节点前面*/
//     freeNodeHead = pNode;
// }
//
// class ActualClass
// {
// public:
//     void *operator new(size_t size);
//     void operator delete(void *p);
// };
// void *ActualClass::operator new(size_t size)
// {
//     MemPool<sizeof(ActualClass), 2> pool;
//     cout << "运算符重载";
//     return pool.Malloc();
// }
// void ActualClass::operator delete(void *p)
// {
//     MemPool<sizeof(ActualClass), 2> pool;
//     pool.Free(p);
// }
// int ActualClass::count = 0;

// 离散傅里叶变化

main()
{
    system("pause");
}