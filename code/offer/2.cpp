//字符串替换
//请实现一个函数，将一个字符串中的每个空格替换成“%20”。
//例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

#include <iostream>
#include <string>

using namespace std;

class Solution
{
public:
    char * replaceSpace(string str)
    {
        int len = str.length();
        //先计算有多少个空格
        int count = 0;
        for(int i=0;i<len;i++)
        {
            if(str[i]==' ')
            {
                count++;
            }
        }
        char *new_str = new char[len+2*count];//新的字符串
        int N = 0;//记录当前有几个空格
        for(int i=0;i<len;i++)
        {
            if(str[i]!=' ')
            {
                new_str[i+2*N] = str[i];
            }
            else
            {
                new_str[i+2*N] = '%';
                new_str[i+2*N+1] = '2';
                new_str[i+2*N+2] = '0';
                N++;
            } 
        }
        return new_str;
    }
};

int main()
{
    Solution s;
    string str= "ab cd ds";
    char* new_str = s.replaceSpace(str);
    string t(new_str);
    cout<<t;
    system("pause");
    return 0;
}

