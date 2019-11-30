#include <iostream>

using namespace std;

class Solution{
public:
    int jumpFloor(int n)
    {
        int pre1 = 1;
        int pre2 = 2;
        int cur = 0;//当前计算结果
        for(int i=3;i<=n;i++)
        {
            cur = pre1+pre2;
            pre1 = pre2;
            pre2 = cur;
        }
        if(n<3)
        {
            return n;
        }
        return cur;
    }    
};





