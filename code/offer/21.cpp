//二叉搜索树的后序遍历序列
//输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

#include <iostream>
#include <vector>

using namespace std;

bool isBST(vector<int> sequence,int start,int end)
{
    if(start>=end)
    {
        return true;
    }
    int val = sequence[end];//根节点
    int split;//划分的索引，第一个for循环为了找到划分的位置
    for(int i=start;i<end;i++)
    {
        if(sequence[i]>val)
        {
            split = i;
            break;
        }
    }
    //判断后面是否还有小于根节点的数，如果有，返回false
    for(int i=split;i<end;i++)
    {
        if(sequence[i]<val)
        {
            return false;
        }
    }
    return isBST(sequence,start,split-1)&&isBST(sequence,split,end-1);
}


class Solution{
public:
    bool VerifySquenceOfBST(vector<int> sequence)
    {
        int len = sequence.size();
        if(len==NULL)
        {
            return false;
        }
        return isBST(sequence,0,len-1);
    }
};


