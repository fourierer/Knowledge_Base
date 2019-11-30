//重建二叉树
//输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
//例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

#include <iostream>
#include <vector>

using namespace std;

struct TreeNode 
{
    int val;
    TreeNode*left;
    TreeNode*right;
};

//后序输出二叉树
void PostOrderBiTree(TreeNode*T)
{
    if(T==NULL)
    {
        return;
    }
    else
    {
        PostOrderBiTree(T->left);
        PostOrderBiTree(T->right);
        cout<<T->val<<" ";
    }
}

class Solution 
{
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        int len = vin.size();
        if (len==0)
        {
            return NULL;
        }
        TreeNode *head = new TreeNode;
        head->val = pre[0];
        vector<int>left_pre,left_vin,right_pre,right_vin;//分别存储左右子树的先序和中序

        //在中序里面寻找根节点的位置
        int gen = 0;
        for(int i=0;i<len;i++)
        {
            if(vin[i]==pre[0])
            {
                gen = i;
                break;
            }
        }

        //左子树的先序和中序
        for(int i=0;i<gen;i++)
        {
            left_vin.push_back(vin[i]);
            left_pre.push_back(pre[i+1]);
        }

        //右子树的先序和中序
        for(int i=gen+1;i<len;i++)
        {
            right_vin.push_back(vin[i]);
            right_pre.push_back(pre[i]);
        }

        head->left = reConstructBinaryTree(left_pre,left_vin);
        head->right = reConstructBinaryTree(right_pre,right_vin);
        return head;
    }
};


int main()
{
    Solution s;
    vector<int>pre,vin;
    pre.push_back(1);
    pre.push_back(2);
    pre.push_back(3);
    vin.push_back(2);
    vin.push_back(1);
    vin.push_back(3);
    TreeNode*T = s.reConstructBinaryTree(pre,vin);
    PostOrderBiTree(T);//后序输出，验证构建的正确性

    system("pause");
    return 0;
}



