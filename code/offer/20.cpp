//从上往下打印二叉树
//从上往下打印出二叉树的每个节点，同层节点从左至右打印。

#include <iostream>
#include <vector>
#include <queue>

using namespace std;

struct TreeNode{
    int val;
    TreeNode*left;
    TreeNode*right;
};

class Solution{
public:
    vector<int> PrintFromTopToBottom(TreeNode*root)
    {
        vector<int> v;//要返回的数组
        queue<TreeNode*> q;//队列
        q.push(root);
        while(!q.empty())
        {
            TreeNode* T = q.front();//队列中的树取出来
            q.pop();//删除最前面的树
            if(T==NULL)
            {
                continue;
            }
            v.push_back(T->val);
            q.push(T->left);
            q.push(T->right);
        }
        return v;
    }
};

