#include <iostream>
#include <string.h>
#include <stdio.h>
#include <algorithm>

using namespace std;

int main()
{
    int k,M,N;//����������ͷ������Ŀռ���ڴ�
    cin>>k>>M>>N;//����i��M,N
    int m[k] = {0};//��i��λ�ô����i+1����������Ŀռ�m(i+1)
    int n[k] = {0};//��i��λ�ô����i+1������������ڴ�(i+1)
    int u[k] = {0};//��i��λ�ô����i+1��������Է��������u(i+1)
    int OPT[k+1][M+1][N+1];//��ά����
    memset(OPT, 0, sizeof(OPT));//��ͷ�ļ�#include<string.h>��

    for(int i=0;i<k;i++)
    {
          scanf("%d",&m[i]);//�����������Ŀռ�
    }
        for(int i=0;i<k;i++)
    {
          scanf("%d",&n[i]);//�������������ڴ�
    }
    for(int i=0;i<k;i++)
    {
          scanf("%d",&u[i]);//�������������Է��������
    }

    //��̬�滮��⣬ʵ�����ڼ���һ����ά����
    for(int i =1;i<=k;i++)
    {
          for(int j = 1;j<=M;j++)
          {
                for(int s = 1;s<=N;s++)
                {
                      if((j-m[i-1]<0)||(s-n[i-1]<0))//���û����һ����������else�е��������׳�������
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
