#include <iostream>
#include <stdio.h>
using namespace std;

long int merge(int a[], int left, int mid, int right,int b[])
{
   int i = mid;
   int j = right;
   long int lcount = 0;
   while (i >= left && j > mid)
   {
         if(a[i] > (long long) 3 * a[j])
         {
               lcount += j-mid;
               i--;
         }
         else
         {
               j--;
         }
   }
   i = mid;
   j = right;
   int k = 0;
   while (i >= left && j > mid)
   {
         if(a[i] > a[j])
         {
               b[k++] = a[i--];
         }
         else
         {
               b[k++] = a[j--];
         }
   }
   while (i >= left)
   {
         b[k++] = a[i--];
   }
   while (j > mid)
   {
         b[k++] = a[j--];
   }
   for (i = 0; i < k; i++)
   {
         a[right - i] = b[i];
   }
return lcount;
}

long int solve(int a[],int left, int right,int b[])
{
      long int cnt = 0;
      if(right > left)
      {
            int mid = (right+left) / 2;
            cnt += solve(a,left, mid,b);
            cnt += solve(a,mid + 1, right,b);
            cnt += merge(a,left, mid, right,b);
      }
      return cnt;
}

long int InversePairs(int a[],int len)

{
    int *b=new int[len];
    long int count=solve(a,0,len-1,b);
    delete[] b;
    return count;
}

int main()
{
    long int n;//数组维度
    int *array;//数组
    scanf("%d", &n);
    array  = new int[n];
    for(long int i=0;i<n;i++)
      scanf("%d", &array[i]);

    //寻找显著逆序数
    /*
    long int count = 0;
    for(int i=0;i<n;i++)
    {
          for(int j=i;j<n;j++)
          {
                if(array[i]>3*array[j])
                {
                      count++;
                }
          }
    }
    */
    long int count = InversePairs(array, n);
    printf("%d", count);
    return 0;
}
