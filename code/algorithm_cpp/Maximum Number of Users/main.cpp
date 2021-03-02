#include <iostream>
#include <string.h>
#include <stdio.h>
#include <algorithm>

using namespace std;

int main()
{
    int k,M,N;//任务的数量和服务器的空间和内存
    cin>>k>>M>>N;//输入i和M,N
    int m[k] = {0};//第i个位置代表第i+1个任务需求的空间m(i+1)
    int n[k] = {0};//第i个位置代表第i+1个任务需求的内存(i+1)
    int u[k] = {0};//第i个位置代表第i+1个任务可以服务的人数u(i+1)
    int OPT[k+1][M+1][N+1];//三维数组
    memset(OPT, 0, sizeof(OPT));//在头文件#include<string.h>中

    for(int i=0;i<k;i++)
    {
          scanf("%d",&m[i]);//输入各个任务的空间
    }
        for(int i=0;i<k;i++)
    {
          scanf("%d",&n[i]);//输入各个任务的内存
    }
    for(int i=0;i<k;i++)
    {
          scanf("%d",&u[i]);//输入各个任务可以服务的人数
    }

    //动态规划求解，实际上在计算一个三维矩阵
    for(int i =1;i<=k;i++)
    {
          for(int j = 1;j<=M;j++)
          {
                for(int s = 1;s<=N;s++)
                {
                      if((j-m[i-1]<0)||(s-n[i-1]<0))//如果没有这一条件，下面else中的语句很容易超出索引
                        OPT[i][j][s] = OPT[i-1][j][s];
                      else
                        OPT[i][j][s] = max(OPT[i-1][j][s],OPT[i-1][j-m[i-1]][s-n[i-1]]+u[i-1]);
                }
          }
    }
    int O = OPT[k][M][N];
    cout<<O<<endl;
    return 0;
}
