//连续子数组的最大和
//{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和(子向量的长度至少是1)。

#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        int len = array.size();
        vector<int> dp(len);
        int Max = array[0];
        dp[0] = array[0];
        for(int i=1;i<len;i++)
        {
            int max = dp[i-1]+array[i];
            if(max>array[i])
            {
                dp[i]=max;
            }
            else
            {
                dp[i]=array[i];
            }
            if(dp[i]>Max)
            {
                Max = dp[i];
            }            
        }
        return Max;    
    }
};

