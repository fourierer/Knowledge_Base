#include <iostream>
#include <string>
#include <string.h>
#include <algorithm>
#include <stdio.h>

using namespace std;

int main()
{
      string str1,str2;
      getline(cin,str1);
      getline(cin,str2);
      int m = str1.length();//��һ���ַ����ĳ���
      int n = str2.length();//�ڶ����ַ����ĳ���
      int OPT[m+1][n+1];//0����
      memset(OPT, 0, sizeof(OPT));//��ͷ�ļ�#include<string.h>��
      //�����ά���飬��ģΪmn

      for(int i=1;i<m+1;i++)
      {
            for(int j=1;j<n+1;j++)
            {
                  if(str1[i-1]==str2[j-1])
                  {
                        OPT[i][j] = OPT[i-1][j-1] + 1;
                  }
                  if(str1[i-1]!=str2[j-1])
                  {
                        OPT[i][j] = max(OPT[i-1][j],OPT[i][j-1]);
                  }
            }
      }
      for(int i=0;i<m+1;i++)
      {
            for(int j=0;j<n+1;j++)
            {

            cout<<OPT[i][j]<<" ";
            }
            cout<<endl;
      }
      printf("%d",OPT[m][n]);

      return 0;
}
