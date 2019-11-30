//数字在排序数组中出现的次数
//统计一个数字在排序数组中出现的次数。
#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        int count=0;
        int len = data.size();
        for(int i=0;i<len;i++)
        {
            if(data[i]==k)
            {
                count++;
            }
            if(data[i]==k&&data[i+1]!=k)
            {
                break;
            }
        }
        return count;
    }
};