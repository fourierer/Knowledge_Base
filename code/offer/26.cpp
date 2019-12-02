//平衡二叉树
//输入一棵二叉树，判断该二叉树是否是平衡二叉树。
//平衡二叉树的左右子树也是平衡二叉树，那么所谓平衡就是左右子树的高度差不超过1
#include<iostream>
#include<math.h>
#include<algorithm>

using namespace std;

struct TreeNode{
    int val;
    TreeNode*left;
    TreeNode*right;
};

int TreeDepth(TreeNode*root)
{
    if(root==NULL)
    {
        return 0;
    }
    int dleft = TreeDepth(root->left);
    if(dleft==-1)
    {
        return -1;
    }
    int dright = TreeDepth(root->right);
    if(dright==-1)
    {
        return -1;
    }
    return abs(dright-dleft)>1?-1:max(dright,dleft)+1;
}

class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if(TreeDepth(pRoot)==-1)
        {
            return false;
        }
        return true;
    }
};