//数组中出现次数超过一半的数字
//数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
//由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2，如果不存在则输出0。

#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int len = numbers.size();
        if(len==0)
        {
            return 0;
        }
        int prevalue = numbers[0];
        int count = 1;
        for(int i=1;i<len;i++)
        {
            if(numbers[i]==prevalue)
            {
                count++;
            }
            else
            {
                count--;
                if(count==0)
                {
                    prevalue = numbers[i];
                    count = 1;
                }
            }         
        }

        //判断prevalue出现的次数是否真的大于数组规模的一半
        count = 0;
        for(int i=0;i<len;i++)
        {
            if(numbers[i]==prevalue)
            {
                count++;
            }
        }
        if(count>len/2)
        {
            return prevalue;
        }
        return 0;
    }
};


