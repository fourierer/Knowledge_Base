//合并两个排序的链表
//输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

#include <iostream>

using namespace std;

struct ListNode {
	int val;
	struct ListNode *next;
};

class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode *head = new ListNode;//构造表头,-1作为标志
	    head->val = -1;
	    head->next = NULL;//指向NULL

        ListNode*cur = head;//记录当前节点的位置

        while(pHead1!=NULL&&pHead2!=NULL)
        {
            if(pHead1->val<=pHead2->val)
            {
                ListNode*New = new ListNode;//新的节点
                New->val = pHead1->val;
                New->next = NULL;
                cur->next = New;
                cur = New;
                pHead1 = pHead1->next;
            }
            else
            {
                ListNode*New = new ListNode;//新的节点
                New->val = pHead2->val;
                New->next = NULL;
                cur->next = New;
                cur = New;
                pHead2 = pHead2->next;
            }
        }
        if(pHead1!=NULL&&pHead2==NULL)
        {
            cur->next = pHead1;
        }
        if(pHead1==NULL&&pHead2!=NULL)
        {
            cur->next = pHead2;
        }
        return head->next->next->next;//构造的链表开头都是-1，需要跳过
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
	ListNode*head1 = Create();
    ListNode*head2 = Create();
    ListNode*phead = s.Merge(head1,head2);
	while(phead!=NULL)
	{
		cout<<phead->val<<" ";
		phead = phead->next;
	}
	system("pause");
	return 0;
}
