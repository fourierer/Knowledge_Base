#include <iostream>
#include <string.h>
#include <stdio.h>
#include <algorithm>

using namespace std;

int main()
{
    int k,W;//��Ʒ�������Ͱ���������
    cin>>k>>W;//����i��W
    int w[k] = {0};//��i��λ�ô����i+1����Ʒ������w(i+1)
    int v[k] = {0};//��i��λ�ô����i+1����Ʒ�ļ�ֵv(i+1)
    int OPT[k+1][W+1];
    memset(OPT, 0, sizeof(OPT));//��ͷ�ļ�#include<string.h>��

    for(int i=0;i<k;i++)
    {
          scanf("%d",&w[i]);//���������Ʒ������
    }
    for(int i=0;i<k;i++)
    {
          scanf("%d",&v[i]);//���������Ʒ�ļ�ֵ
    }

    //��̬�滮��⣬ʵ�����ڼ���һ����ά����
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
