#include <iostream>
#include <stdio.h>

using namespace std;

long int merge(int a[], int left, int mid, int right,int b[])
{
	int i = mid;
	int j = right;
	int k = 0;
	while (i >= left && j >= mid+1)
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
	while (j >= mid+1)
	{
		b[k++] = a[j--];
	}
	for (i = 0; i < k; i++)
	{
		a[right - i] = b[i];
	}
}

long int solve(int a[],int left, int right,int b[])
{
	if(right > left)
	{
	int mid = (right+left) / 2;
	solve(a,left, mid,b);
	solve(a,mid + 1, right,b);
	merge(a,left, mid, right,b);
	}
}


int main()
{
	long int n;//数组维度
	scanf("%d", &n);
	int *a = new int[n];
	int *b = new int[n];
	for(long int i=0;i<n;i++)
	{
		scanf("%d", &a[i]);//scanf的速度要比cin的速度快
	}

	solve(a,0,n-1,b);//归并排序

	for(int i = 0;i<n;i++)
		cout<<a[i]<<' ';
	return 0;
}
