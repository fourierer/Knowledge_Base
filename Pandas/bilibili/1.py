import pandas as pd
import numpy as np



############################### 1.Selection（b站视频P4） #######################
# Series[label] # 返回scalar value
# DataFrame[columns] # 返回Series corresponding to colname
# Series.loc[index]
# DataFrame.loc[row_index, col_columns] # 通过index和columns值访问
# DataFrame.iloc[integer]
# DataFrame.iloc[row_index, col_columns] # 通过数字索引访问

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
# print(df[(df.AAA <= 6) & (df.index.isin([0, 2, 4]))])
# df.index.isin[0, 2, 4]，index是否在在[0，2，4]中间


data = {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
df2 = pd.DataFrame(data=data, index=[1, 2, 3, 4]) # index从1开始
# print(df2.iloc[1:3]) # 按照index位置取行，和numpy一样，不包括最后一个，所以是索引位置1，2所在的行，即label为2，3所在的行
'''
   AAA  BBB  CCC
2    5   20   50
3    6   30  -30
'''
# print(df2.loc[1:3]) # 按照label取行，当按照label取行时，包括首尾两个元素，即label为1，2，3所在的行
'''
   AAA  BBB  CCC
1    4   10  100
2    5   20   50
3    6   30  -30
'''


df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
# print((df['AAA']<=6) & (df.index.isin([0,2,4]))) # 返回长度和df一样的bool Series数据结构
# print(df[~((df.AAA <= 6) & (df.index.isin([0, 2, 4])))]) # ~ 表示取反
'''
   AAA  BBB  CCC
1    5   20   50
3    7   40  -50
'''
# print(df.loc[lambda df: (df['AAA']>4) & (df['BBB']<40), :]) # 用lambda函数的形式来表示查询结果

def query(df):
    return (df['AAA']>4) & (df['BBB']<40)
# print(df.loc[query, :]) # 使用真正的函数来代替查询条件或者lambda形式的函数

df = pd.DataFrame(
    {"AAA": [1, 1, 1, 2, 2, 2, 3, 3], "BBB": [2, 1, 3, 4, 5, 1, 2, 3]}
)
# print(df)
'''
   AAA  BBB
0    1    2
1    1    1
2    1    3
3    2    4
4    2    5
5    2    1
6    3    2
7    3    3
'''


############################### 2.添加新的列（b站视频P5） #######################
df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
#(1)直接添加列
# df['DDD'] = df['AAA'] - df['BBB']
df.loc[:, 'DDD'] = df['AAA'] - df['BBB']
# df.loc[:, 'DDD'] = 3 # 利用Series的广播属性
# print(df) # 直接改变df的值

#(2)通过assign函数添加列
def query(df):
    if df['AAA']>4:
        return 'Good'
    return 'Bad'
df.loc[:, 'DDD'] = df.apply(query, axis=1) # axis表示增加列
# print(df) # 直接改变df的值

#(3)通过assign函数添加列
df2 = df.assign(
    EEE = lambda x: x['AAA'] + 3,
    FFF = lambda x: x['BBB'] + 3
) # EEE和FFF就是新加的列，assign可以同时添加过个列
# print(df) # assign不改变原先df的值
# print(df2)

#(4)通过条件分组来添加列
df['GGG'] = 'x'
df.loc[df['AAA']-df['BBB']>5, 'GGG'] = 'big'
df.loc[df['AAA']-df['BBB']<=5, 'GGG'] = 'small'
# print(df)


############################### 3.统计函数（b站视频P6） #######################
df = pd.DataFrame(
    {"AAA": [4, 4, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
#(1)一般统计属性
# print(df.describe()) # 输出所有统计信息
# print(df.mean()) # 统计每列的均值
# print(df['AAA'].mean()) # 统计'AAA'列的均值

#(2)去重和按值计数
# print(df['AAA'].unique()) # 去重，输出[4,6,7]
# print(df['AAA'].value_counts()) # 计数，输出每个数出现的次数

#(3)相关系数和协方差
# print(df.cov()) # 计算列两两之间的协方差
# print(df.corr()) # 计算列两两之间的相关系数
# print(df['AAA'].corr(df['BBB'])) # 计算特定列之间的相关系数
# print(df['AAA'].corr(df['BBB'] - df['CCC']))



############################### 4.缺失值的处理（b站视频P7） #######################
# isnull()和notnull():检测是否为空值

# dropna():丢弃、删除缺失值，删除行还是删除列见axis参数

# fillna():填充缺失值，具体填充的值和填充方式见value和method参数


############################### 5.不允许修改子DataFrame（b站视频P8） #######################

df = pd.DataFrame(
    {"AAA": [8, 4, 4, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
# print(df['AAA']>4)
# df[df['AAA']>4]['BBB'] = 3
# 这一步会报错：
# /Users/yihe/Documents/donkey/Alibaba/learning_note/Pandas/Cookbook/1.py:138: SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
# 即df['AAA']筛选的结果是一个子DataFrame，不确定直接是view（直接影响源DataFrame）还是copy
# 建议使用：
df.loc[df['AAA']>4, 'BBB'] = 3
# print(df)


############################### 6.Pandas数据排序（b站视频P9） #######################
# Series.sort_values(ascending=True, inplace=True) # ascending默认为升序排列，inplace表示是否修改源Series
# DataFrame.sort_values(y, ascending=True, inplace=True) # by为字符串或者字符串列表，表示单列排序还是多列排序
df = pd.DataFrame(
    {"AAA": [4, 4, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
# print(df['AAA'].sort_values()) # 对Series进行排序
# print(df.sort_values(by='AAA', ascending=False)) # 对DataFrame进行排序
# print(df.sort_values(by=['AAA', 'BBB'], ascending=False)) # 对多列进行降序排列
# print(df.sort_values(by=['AAA', 'BBB'], ascending=[True, False])) # 对'AAA'列进行升序，对'BBB'列进行降序


############################### 7.Pandas字符串处理（b站视频P10） #######################
# 先获取Series的字符串属性，然后在属性上调用函数
# 只能在字符串列上使用，不能在数字列上使用
# DataFrame没有str属性和方法
# Series.str不是原生的python字符串，而是有自己的一套方法，不过大部分都很类似
# 在API reference中，找到Series，然后搜索string handling即可看到Series.str自己的方法

#(1)Series.str的基本函数
df = pd.DataFrame(
    {"AAA": ['4C', '4C', '6C', '7C'], "BBB": [10, 20, 30, 40], "CCC": ['-100', '-50', '-30', '-50']}
)
# print(df['AAA'].str) # <pandas.core.strings.accessor.StringMethods object at 0x7fe954878a30>，str属性方法，就是指string handling中过的方法
# print(df['AAA'].str.replace('C', '')) # 替换'AAA'列中的符号'C'，不改变源df
# df.loc[:, 'AAA'] = df['AAA'].str.replace('C', '') # 改变源df
# print(df['AAA'].str.isnumeric())
# print(df['AAA'].str.len())

#(2)使用str的startswith和contains得到bool的Series可以做条件查询
condition = df['AAA'].str.startswith('4')
# print(condition)
# print(df[condition].head())

#(3)需要多次的str的链式处理
df['CCC'].str.replace('-', '') # 去掉'-'
# df['CCC'].str.replace('-', '').slice(0, 2) # 报错AttributeError: 'Series' object has no attribute 'slice'，因为slice是str属性上的方法，不是Series上的方法
df['CCC'].str.replace('-', '').str.slice(0, 2)# 获取字符串从索引0到1的字符串，或者：
df['CCC'].str.replace('-', '').str[0:2]


############################### 8.Pandas的axis参数（b站视频P11） #######################
# axis=0 # 行，index
# axis=1 # 列，columns

df = pd.DataFrame(
    np.arange(12).reshape(3, 4),
    columns=list('ABCD')
)
# print(df)
df.drop('A', axis=1) # 删除列'A'
df.drop(1, axis=0)
# print(df.mean(axis=0)) # 按行求均值，不是求每一行的均值，所以结果是4个数，每个数代表列平均数
def get(x):
    return x['A'] + x['B'] + x['C'] + x['D']
df['sum'] = df.apply(get, axis=1) # 按列求和，即生成每一行的和作为新列
# print(df)


############################### 9.Pandas的索引index（b站视频P12） #######################

df = pd.DataFrame(
    np.arange(12).reshape(3, 4),
    columns=list('ABCD')
)
# print(df.loc[4]) # 报错，DataFrame.loc不能按index值取行
df.set_index('A', inplace=True, drop=False) # 使用列'A'的值作为新的index，drop为False表示'A'列保留
# print(df)
# print(df.loc[4]) # 可以按照index值取行
# print(df.sort_index())
# pandas查询index，当index的值是唯一的时候，查询使用哈希方法；当index是有序的时候，查询使用二分方法；


############################### 10.Pandas的Merge语法（b站视频P13） #######################
# Pandas的Merge语法和Sql的join，将不同的表按照key关联到一个表

# #(1)电影数据集中的join实例
# # 评分数据
# df_ratings = pd.read_csv(
#     '../dataset/ml-1m/ratings.dat',
#     sep='::',
#     engine='python',
#     names='UserID::MovieID::Rating::Timestamp'.split('::')
# )
# # print(df_rating.head())


# # 用户数据
# df_users = pd.read_csv(
#     '../dataset/ml-1m/users.dat',
#     sep='::',
#     engine='python',
#     names='UserID::Gender::Age::Occupation::Zip-code'.split('::')
# )
# # print(df_users.head())


# # 电影数据
# df_movies = pd.read_csv(
#     '../dataset/ml-1m/movies.dat',
#     sep='::',
#     engine='python',
#     names='MovieID::Title::Genres'.split('::'),
#     encoding='ISO-8859-1'
# )
# # print(df_movies.head())


# # 对用户数据和评分数据按照UserID进行join
# df_ratings_users = pd.merge(
#     df_ratings, df_users, left_on='UserID', right_on='UserID', how='inner'
# )
# # print(df_ratings_users.head())
# # inner参数表示取交集，即两个表中同时有的数据才会保留，how默认参数也是inner


# # 对用户数据和评分数据merge之后的数据和电影数据进行join
# df_ratings_users = pd.merge(
#     df_ratings_users, df_movies, left_on='MovieID', right_on='MovieID', how='inner'
# )
# # print(df_ratings_users.head())


#(2)merge时数量的对齐关系
# one-to-one:1对1的关系，关联的key都是唯一的
# 如（学号，姓名）merge（学号，年龄）
# 结果条数1*1
left = pd.DataFrame({'id':[1, 2, 3, 4],
                      'name':['name_a', 'name_b', 'name_c', 'name_d']})
right = pd.DataFrame({'id':[1, 2, 3, 4],
                      'age':[21, 22, 23, 24]})
# print(pd.merge(left, right, on='id'))
# print(pd.merge(left, right, left_on='id', right_on='id'))



# one-to-many:1对多的关系，左边唯一key，右边不唯一key
# 如（学号，姓名）merge（学号，[语文成绩，数学成绩]）
# 结果条数1*N
left = pd.DataFrame({'id':[1, 2, 3, 4],
                      'name':['name_a', 'name_b', 'name_c', 'name_d']})
right = pd.DataFrame({'id':[1, 1, 2, 3],
                      'grade':['语文78', '数学87', '语文90', '语文85']})
# print(pd.merge(left, right, on='id'))


# many-to-many:多对多的关系，左右都不唯一key
# 如（学号，[语文成绩，数学成绩]）merge（学号，[篮球，足球]）
# 结果条数M*N
left = pd.DataFrame({'id':[1, 2, 3, 3],
                      'hobby':['篮球', '篮球', '篮球', '足球']})
right = pd.DataFrame({'id':[1, 1, 2, 3],
                      'grade':['语文78', '数学87', '语文90', '语文85']})
# print(pd.merge(left, right, on='id'))


#(3)理解left join, right join, inner join和outer join的区别
left = pd.DataFrame({'key':['k0', 'k1', 'k2', 'k3'],
                     'A':['A1', 'A2', 'A3', 'A4'],
                     'B':['B1', 'B2', 'B3', 'B4']})
right = pd.DataFrame({'key':['k0', 'k1', 'k4', 'k5'],
                     'C':['C1', 'C2', 'C3', 'C4'],
                     'D':['D1', 'D2', 'D3', 'D4']})
# inner join，左右都有才会保留在join中
# print(pd.merge(left, right, how='inner')) # on默认为两个表columns的交集，即key

# left join，左边的出现在join，右边如果无法匹配则为Null
# print(pd.merge(left, right, how='left'))

# right join，右边的出现在join，左边如果无法匹配则为Null
# print(pd.merge(left, right, how='right'))

# outer join，左右都会出现在join中，如果无法匹配则为Null
# print(pd.merge(left, right, how='outer'))


#(4)非key的字段出现重名
left = pd.DataFrame({'key':['k0', 'k1', 'k2', 'k3'],
                     'A':['A1', 'A2', 'A3', 'A4'],
                     'B':['B1', 'B2', 'B3', 'B4']})
right = pd.DataFrame({'key':['k0', 'k1', 'k4', 'k5'],
                     'A':['A5', 'A6', 'A7', 'A8'],
                     'D':['D1', 'D2', 'D3', 'D4']})
# print(pd.merge(left, right, on='key'))
# print(pd.merge(left, right, on='key', suffixes=('_left', '_right')))

df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo1'],
                    'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo2'],
                    'value': [5, 6, 7, 8]})
# print(pd.merge(df1, df2, left_on='lkey', right_on='rkey', how='inner'))




############################### 11.Pandas的Concat合并（b站视频P14） #######################
# pandas.concat(objs, axis=0, join='outer', ignore_index=False)
# objs:要合并的列表，可以是DataFrame和Series，可以混合
# axis:0表示按行合并，1表示按列合并
# join:合并时索引的对齐方式
# ignore_index:是否忽略原来的数据索引

df1 = pd.DataFrame({'A':['A0', 'A1', 'A2', 'A3'],
                    'B':['B0', 'B1', 'B2', 'B3'],
                    'C':['C0', 'C1', 'C2', 'C3'],
                    'D':['D0', 'D1', 'D2', 'D3'],
                    'E':['E0', 'E1', 'E2', 'E3']
                  })
df2 = pd.DataFrame({'A':['A4', 'A5', 'A6', 'A7'],
                    'B':['B4', 'B5', 'B6', 'B7'],
                    'C':['C4', 'C5', 'C6', 'C7'],
                    'D':['D4', 'D5', 'D6', 'D7'],
                    'F':['F4', 'F5', 'F6', 'F7']
                  })
# axis=0按行合并
# print(pd.concat([df1, df2])) # 按行合并，同时保留原来的索引
# print(pd.concat([df1, df2], ignore_index=True)) # 忽略原来的索引
# print(pd.concat([df1, df2], ignore_index=True, join='inner'))

# axis=1相当于添加列
# print(pd.concat([df1, df2], axis=1))


# DataFrame.append(obj, ignore_index=False)
# append只有按行合并，没有按列合，相当于concat按行合并的简化形式
df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
# print(df1.append(df2))
# print(df1.append(df2, ignore_index=True))




############################### 12.Pandas批量拆分与合并Excel文件（b站视频P15） #######################



############################### 13.Pandas实现groupby分组统计（b站视频P16） #######################
df = pd.DataFrame({'A':['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B':['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C':np.random.randn(8),
                   'D':np.random.randn(8)
})
# print(df.groupby('A').sum()) # 列'A'变成索引对数据分组，即列'A'中相同值所在的行会变成一组，列'B'不是数字，故被自动忽略
# print(df.groupby(['A', 'B']).mean()) # 列'A'和'B'变成二级索引
# print(df.groupby(['A', 'B'], as_index=False).mean()) # 列'A'和'B'不变成索引
df1 = df.groupby('A').agg([np.sum, np.mean, np.std])
# print(df1) # 列索引变成多级索引
# print(df1.loc['bar', 'C']) # Series

# print(df1.loc[:, 'C']) # DataFrame
# 等价于
# print(df1['C'])
# 等价于
# print(df.groupby('A')['C'].agg([np.sum, np.mean, np.std]))

# 对不同的列使用不同的统计函数
# print(df.groupby('A').agg({'C':np.max, 'D':np.min}))
# 实例：对每个月统计最高气温，最低气温和平均空气指数
# group_data = df.groupby('month').agg({'bWendu':np.max, 'yWendu':np.min, 'aqi':np.mean})



# 理解groupby
# g = df.groupby('A')
# for name, group in g:
#     print(name)
#     print(group)
# print(g.get_group('bar')) # 通过get_group函数获取'bar'对应的数据

# g = df.groupby(['A', 'B'])
# for name, group in g:
#     print(name) # name变成了元祖，如('foo', 'three')
#     print(group)
# print(g.get_group(('foo', 'three')))


############################### 14.Pandas中的分层索引MultiIndex（b站视频P17） #######################
# 一般groupby之后会创建的多层索引数据，具体用法见视频

#(1)Series的分层索引
#(2)Series分层索引筛选数据
#(3)DataFrame的分层索引
#(4)DataFrame分层索引筛选数据


############################### 15.Pandas的数据转换函数map,apply,applymap（b站视频P18） #######################

#(1)map只用于Series，实现值到值的映射
# Series.map(dict)
# Series.map(function) # 其中function的参数是Series中的每个值 


#(2)apply用于Series处理某个值，也可以处理DataFrame的某个轴方向的Series
# Series.apply(function) # 其中function的参数是Series中的每个值
# DataFrame.apply(function) # 其中function的参数是DataFrame中某个方向上的Series


#(3)applymap只能用于DataFrame，用于DataFrame某个值的处理
# DataFrame.applymap(function) # 其中function的参数是DataFrame中的每个值


############################### 16.Pandas对groupby之后的每个分组应用apply函数（b站视频P19） #######################
# pandas的groupby函数遵从split-apply-combine规则，其中在apply阶段可以定义各种函数
# df.groupby('x').apply(function) # function的参数可以是DataFrame，可以是Series，也可以是某个值


# 实例1:对数值列按分组的归一化
# 评分数据
df_ratings = pd.read_csv(
    '../dataset/ml-1m/ratings.dat',
    sep='::',
    engine='python',
    names='UserID::MovieID::Rating::Timestamp'.split('::')
)
# print(df_rating.head())

def rating_norm(df):
    # 参数df是每个分组的DataFrame，最终只需返回一个新的DataFrame即可
    min_value = df['Rating'].min() # 求列Rating的Series的最小值
    max_value = df['Rating'].max()
    # 新增一列，对df['Rating']的Series应用apply，apply中函数的参数是Series中的某个值
    df['Rating_norm'] = df['Rating'].apply(lambda x: (x - min_value)/(max_value - min_value))
    return df
ratings = df_ratings.groupby('UserID').apply(rating_norm)
# print(ratings.head())


############################### 17.Pandas使用stack和pivot进行数据透视（b站视频P20） #######################
#(1)使用stack进行数据透视
df = pd.read_csv(
    '../dataset/ml-1m/ratings.dat',
    sep='::',
    engine='python',
    names='UserID::MovieID::Rating::Timestamp'.split('::')
)
df['pdate'] = pd.to_datetime(df['Timestamp'], unit='s')
# print(df.head())
df_group = df.groupby([df['pdate'].dt.month, 'Rating'])['UserID'].agg(pv=np.sum)
# print(df_group)

df_stack = df_group.unstack()
# print(df_stack)
# df_stack.plot()
# import matplotlib.pyplot as plt
# plt.show()
# print(df_stack.stack().head()) # stack和unstack互为逆操作


#(2)或者使用pivot进行数据透视
df_reset = df_group.reset_index()
# print(df_reset.head())
df_pivot = df_reset.pivot('pdate', 'Rating', 'pv') # 等价于df_stack
# print(df_pivot.head())
# df_pivot.plot()
# import matplotlib.pyplot as plt
# plt.show()


############################### 18.Pandas使用apply函数给表格添加多列（b站视频P21） #######################
df = pd.DataFrame(
    np.arange(12).reshape(3, 4),
    columns=list('ABCD')
)
# 添加一列的方法
def get(df):
    new_column =  df['A'] + df['B'] + df['C'] + df['D']
    return new_column
df['sum'] = df.apply(get, axis=1) # 按列求和，即生成每一行的和作为新列
# print(df)


df = pd.DataFrame(
    np.arange(12).reshape(3, 4),
    columns=list('ABCD')
)
# 添加多列的方法
def Get(df):
    new_column1, new_column2 = df['A'] + df['B'] + df['C'] + df['D'], (df['A'] + df['B'] + df['C'] + df['D'])/4
    return new_column1, new_column2

df[['sum', 'avg']] = df.apply(Get, axis=1, result_type='expand')
# print(df)

