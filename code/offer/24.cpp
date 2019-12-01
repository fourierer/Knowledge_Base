//最小的K个数
//输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4

#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        vector<int> v;
        if(k>input.size())
        {
            return v;
        }
        sort(input.begin(),input.end());
        for(int i=0;i<k;i++)
        {
            v.push_back(input[i]);
        }
        return v;
    }
};

