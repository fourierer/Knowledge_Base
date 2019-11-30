//链表中倒数第k个结点
//输入一个链表，输出该链表中倒数第k个结点。

#include <iostream>

using namespace std;

struct ListNode{
    int val;
    ListNode*next;
};

//先将链表反转
ListNode* reverse(ListNode* head)
{
    ListNode*pre = NULL;
    ListNode*cur = NULL;
    while(head!=NULL)
    {
        cur = head->next;
        head->next = pre;
        pre = head;
        head = cur;
    }
    return pre;
}

class Solution {
public:
    int FindKthToTail(ListNode* pListHead, unsigned int k) {
        pListHead = reverse(pListHead);//先将链表反转
        int i = 1;
        while(i<k)
        {
            i++;
            pListHead = pListHead->next;
        }
        return pListHead->val;
    }
};

ListNode *Create()
{
	ListNode *head = new ListNode;//构造头结点
	head->val = -1;
	head->next = NULL;//指向NULL

	ListNode *Cur = head;	//构造当前结点，用于记录当前链表构造的位置，初始位置为head

	int data;	//插入链表的数据
	while(1)
	{
		cout << "请输入当前节点的数值：" << endl;
		cin >> data;
		if(data == -1)	//插入-1时结束链表构造
		{
			break;
		}

		ListNode *New = new ListNode;	//构造新结点，用于循环插入链表
		New->val = data;	//新结点数据
		New->next = NULL;	  //新节点指向NULL
		Cur->next = New;	//当前结点指向新构造的结点
		Cur = New;	//当前结点顺移至新结点处，记录链表插入位置
	}
	return head;	//返回头结点
}

int main()
{
    Solution s;
	ListNode*head = Create();
    int data = s.FindKthToTail(head,4);
    cout<<data;
	system("pause");
	return 0;
}