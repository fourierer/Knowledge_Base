//输入两棵二叉树A，B，判断B是不是A的子结构或者子树
//子树的意思是包含了一个结点，就得包含这个结点下的所有节点，一棵大小为n的二叉树有n个子树，就是分别以每个结点为根的子树;
//子结构的意思是包含了一个结点，可以只取左子树或者右子树，或者都不取。
#include <iostream>

using namespace std;

struct TreeNode{
    int val;
    TreeNode*lchild;
    TreeNode*rchild;
};


bool doesTree1HaveTree2(TreeNode* node1,TreeNode* node2)
{
    //如果Tree2已经遍历完了都能对应的上，返回true
    if(node2==NULL)
    {
        return true;
    }
    //如果Tree2还没有遍历完，Tree1却遍历完了。返回false
    if(node1==NULL)
    {
        return false;
    }
    //如果其中有一个点没有对应上，返回false
    if(node1->val!=node2->val)
    {
        return false;
    }
    //如果根节点对应的上，那么就分别去子节点里面匹配
    return doesTree1HaveTree2(node1->lchild,node2->lchild)&&doesTree1HaveTree2(node1->rchild,node2->rchild);
}


//判断是否为子结构
bool HasSubTree(TreeNode* root1,TreeNode* root2)
{
    bool result = false;
    //当Tree1和Tree2都不为零的时候，才进行比较。否则直接返回false
    if (root2 != NULL && root1 != NULL)
    {
        //如果找到了对应Tree2的根节点的点
        if(root1->val == root2->val)
        {
            //以这个根节点为为起点判断是否包含Tree2
            result = doesTree1HaveTree2(root1,root2);
        }
        //如果找不到，那么就再去root的左儿子当作起点，去判断时候包含Tree2
        if(!result)
        {
            result = HasSubTree(root1->lchild,root2);
        }
        //如果还找不到，那么就再去root的右儿子当作起点，去判断时候包含Tree2
        if(!result)
        {
            result = HasSubTree(root1->rchild,root2);
        }
    }
    return result;
}

//判断是否为子树




//先序创建二叉树
void CreateBiTree(TreeNode **T)
{
    int ch;
    cin >> ch;
    if (ch == -1)
    {
        *T = NULL;
        return;
    }
    else
    {
        *T = new TreeNode;
        (*T)->val = ch;
        cout << "input" << ch << "'s left son node:";
        CreateBiTree(&((*T)->lchild));
        cout << "input" << ch << "'s right son node:";
        CreateBiTree((&(*T)->rchild));
    }

    return;
}


int main(){
    TreeNode*A;
    TreeNode*B;
    CreateBiTree(&A);
    CreateBiTree(&B);
    cout<<HasSubTree(A,B)<<endl;;
    system("pause");
    return 0;
}

