import numpy as np
import pandas as pd
from io import StringIO
import json


########################1.csv############################
data = 'col1,col2,col3\na,b,1\na,b,2\nc,d,3'
# print(StringIO(data))
# print(pd.read_csv(StringIO(data)))
# print(pd.read_csv(StringIO(data), usecols=['col1', 'col2']))


########################2.json############################
# print(list('ABC')) # ['A', 'B', 'C']
df = pd.DataFrame(np.random.randn(5, 3), columns=list('ABC'))
# print(df)
dfj = df.to_json('./dfj_test.json')
# print(type(dfj)) # <class 'str'>
# with open('dfj.json', 'w') as f:
#     json.dump(dfj, f)


########################3.html############################
url = (
    "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/io/data/html/spam.html"
)
# dfs = pd.read_html(url)
# print(dfs)

df = pd.DataFrame(np.random.randn(2, 3))
# print(df)
html = df.to_html()
# print(type(html)) # <class 'str'>
# print(html)


########################4.latex############################
df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["c", "d"])
# print(df.style.to_latex())


########################5.xml############################
xml = """<?xml version="1.0" encoding="UTF-8"?>
<bookstore>
  <book category="cooking">
    <title lang="en">Everyday Italian</title>
    <author>Giada De Laurentiis</author>
    <year>2005</year>
    <price>30.00</price>
  </book>
  <book category="children">
    <title lang="en">Harry Potter</title>
    <author>J K. Rowling</author>
    <year>2005</year>
    <price>29.99</price>
  </book>
  <book category="web">
    <title lang="en">Learning XML</title>
    <author>Erik T. Ray</author>
    <year>2003</year>
    <price>39.95</price>
  </book>
</bookstore>"""

# df1 = pd.read_xml(xml)
# df2 = pd.read_xml("https://www.w3schools.com/xml/books.xml") # ?
# print(df1)
# print(df2)

geom_df = pd.DataFrame(
    {
        "shape": ["square", "circle", "triangle"],
        "degrees": [360, 360, 180],
        "sides": [4, np.nan, 3],
    }
)
# print(geom_df)
# print(geom_df.to_xml())


########################6.excel############################
# pd.read_excel("path_to_file.xls", sheet_name="Sheet1")

df = pd.DataFrame(
    {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
)
df.to_excel('./path_to_file.xlsx')





