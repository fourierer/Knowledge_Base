//数值的整数次方
//给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。保证base和exponent不同时为0 
#include <iostream>

using namespace std;

class Solution{
public:
    double Power(double base,int exponent)
    {
        if(base==0.0)
        {
            return 0.0;
        }
        else if(exponent==0)
        {
            return 1.0;
        }
        else
        {
            double result;
            int e = exponent>0?exponent:-exponent;
            for(int i=1;i<=e;i++)
            {
                result = result*base;
            }
            return exponent>0?result:1/result;
        }        
    }
};



