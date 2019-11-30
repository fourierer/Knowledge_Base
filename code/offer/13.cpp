//调整数组顺序使奇数位于偶数前面
//输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
//并保证奇数和奇数，偶数和偶数之间的相对位置不变。

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int len = array.size();
        for(int i=0;i<len;i++)
        {
            for(int j=0;j<len;j++)
            {
                if(array[j]%2==0&&array[j+1]%2==1)
                {
                    int temp = array[j];
                    array[j] = array[j+1];
                    array[j+1] = temp;
                }
            }
        }
    }
};

int main()
{
    Solution s;
    vector<int> array;
    array.push_back(1);
    array.push_back(2);
    array.push_back(3);
    array.push_back(4);
    array.push_back(5);
    array.push_back(6);
    s.reOrderArray(array);
    for(int i=0;i<array.size();i++)
    {
        cout<<array[i]<<" ";
    }
    cout<<endl;
    system("pause");
    return 0;
}
