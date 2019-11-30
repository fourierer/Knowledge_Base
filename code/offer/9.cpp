//变态跳台阶
//一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

#include<iostream>

using namespace std;

class Solution{
public:
    int jumpFloorII(int number)
    {
        if(number<2)
        {
            return number;
        }
        else
        {
            int pre = 0;
            int cur = 1;
            for(int i=2;i<=number;i++)
            {
                pre = cur;
                cur = 2*pre;                
            }
            return cur;
        }        
    }
};

