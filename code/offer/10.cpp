//矩形覆盖
//我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

#include <iostream>

using namespace std;

class Solution{
public:
    int rectCover(int number)
    {
        int pre1 = 1;
        int pre2 = 2;
        int cur = 0;
        for(int i=3;i<=number;i++)
        {
            int cur = pre1+pre2;
            pre1 = pre2;
            pre2 = cur;
        }
        if(number<3)
        {
            return number;
        }
        return cur;
    }
};


