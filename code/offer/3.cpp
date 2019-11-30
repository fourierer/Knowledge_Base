//从尾到头打印链表
//输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
#include <iostream>
#include <vector>

using namespace std;

//链表结构体
struct ListNode{
    int val;
    ListNode*next;
};

class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> A;
        while(head!=NULL){
            A.push_back(head->val);
            head = head->next;
        }
        int len = A.size();
        vector<int>B(len);
        for(int i=0;i<len;i++)
        {
            B[i] = A[len-i-1];
        }
        return B;
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
    ListNode*list = Create();
    while(list!=NULL)
    {
        cout<<list->val<<" ";
        list = list->next;
    }
    system("pause");
    return 0;
}







