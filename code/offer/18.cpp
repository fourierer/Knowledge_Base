//˳ʱ���ӡ����
//����һ�����󣬰��մ���������˳ʱ���˳�����δ�ӡ��ÿһ�����֣����磬�����������4 X 4���� 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
//�����δ�ӡ������1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> ssz;
        if(matrix.size()==0||matrix[0].size()==0)
        {
            return ssz;
        }
        int up = 0;
        int down = matrix.size()-1;
        int left = 0;
        int right = matrix[0].size()-1;

        while(true)
        {
            for(int col=left;col<=right;col++)
            {
                ssz.push_back(matrix[up][col]);
            }
            up++;
            if(up>down)
            {
                break;
            }

            for(int row=up;row<=down;row++)
            {
                ssz.push_back(matrix[row][right]);
            }
            right--;
            if(left>right)
            {
                break;
            }

            for(int col=right;col>=left;col--)
            {
                ssz.push_back(matrix[down][col]);
            }
            down--;
            if(up>down)
            {
                break;
            }

            for(int row=down;row>=up;row--)
            {
                ssz.push_back(matrix[row][left]);
            }
            left++;
            if(left>right)
            {
                break;
            }
        }
        return ssz;
    }
};
