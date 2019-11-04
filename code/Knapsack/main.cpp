#include <iostream>
#include <string.h>
#include <stdio.h>
#include <algorithm>

using namespace std;

int main()
{
    int k,W;//物品的数量和包的总容量
    cin>>k>>W;//输入i和W
    int w[k] = {0};//第i个位置代表第i+1个物品的重量w(i+1)
    int v[k] = {0};//第i个位置代表第i+1个物品的价值v(i+1)
    int OPT[k+1][W+1];
    memset(OPT, 0, sizeof(OPT));//在头文件#include<string.h>中

    for(int i=0;i<k;i++)
    {
          scanf("%d",&w[i]);//输入各个物品的重量
    }
    for(int i=0;i<k;i++)
    {
          scanf("%d",&v[i]);//输入各个物品的价值
    }

    //动态规划求解，实际上在计算一个二维矩阵
    for(int i =1;i<=k;i++)
    {
          for(int j = 1;j<=W;j++)
          {
                if(j-w[i-1]<0)
                  OPT[i][j] = OPT[i-1][j];
                else
                  OPT[i][j] = max(OPT[i-1][j],OPT[i-1][j-w[i-1]]+v[i-1]);
          }
    }
    int O = OPT[k][W];
    for(int i = 0;i<k+1;i++)
    {
          for(int j = 0;j<W+1;j++)
          {
                cout<<OPT[i][j]<<"  ";
          }
          cout<<endl;
    }
    cout<<O<<endl;



    return 0;
}
