## 数据库

### 一、基础知识

#### 1.mysql在mac上的安装

参考（https://blog.csdn.net/baidu_26315231/article/details/88963558）

安装好mysql之后在终端输入：

```shell
mysql -u root -p
```

随后输入密码即可启动mysql，界面如下：

![mysql启动页面](/Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/mysql启动页面.png)



#### 2.基本mysql指令

查看mysql中有哪些数据库：show databases;

选择使用某个数据库：use test;（表示使用名字叫test的数据库）

创建数据库：create database bjpowernode;（表示创建一个名字是bjpowernode的数据库）

查看某个数据库下有哪些表：show tables;

查看mysql数据库的版本号：select version();

查看当前用的哪个数据库：select database();

mysql不见分号不执行，可以输入\c结束输入当前指令；

退出mysql：exit

注意：以上命令不区分大小写



#### 3.数据库基本单位：表（table）

任何一张表都有行和列：

行表示数据/记录，列表示字段。每个字段都有字段名，数据类型，约束等属性（比如唯一性约束，则该字段下的数据不能重复）。



#### 4.SQL语言分类

DQL：数据查询语言，凡是带有select关键字的都是DQL；

```mysql
select ...
```



DML：数据操作语言，凡是对表中数据进行增删改的都是DML；

```mysql
insert ...
delete ...
update ...
```

（前两个最常用）





DDL：数据定义语言，凡是带有create，drop，alter的都是DDL，DDL主要操作的是表的结构，不是表中数据。

```mysql
create ...
drop ...
alter ...
```



TCL：事务控制语言，包括事务提交：commit；事务回滚：rollback；

DCL：数据控制语言，例如授权grant；撤销权限revoke；



#### 5.导入sql数据

```mysql
source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/document/bjpowernode.sql
```



运行show tables;，显示其中的几张表

```bash
mysql> show tables;
+-----------------------+
| Tables_in_bjpowernode |
+-----------------------+
| DEPT                  |
| EMP                   |
| SALGRADE              |
+-----------------------+
3 rows in set (0.00 sec)
```



查看表中数据：select * from emp;（从emp里查询所有数据）

```bash
mysql> select * from emp;
+-------+--------+-----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB       | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+-----------+------+------------+---------+---------+--------+
|  7369 | SMITH  | CLERK     | 7902 | 1980-12-17 |  800.00 |    NULL |     20 |
|  7499 | ALLEN  | SALESMAN  | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN  | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER   | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN  | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER   | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER   | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7788 | SCOTT  | ANALYST   | 7566 | 1987-04-19 | 3000.00 |    NULL |     20 |
|  7839 | KING   | PRESIDENT | NULL | 1981-11-17 | 5000.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN  | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
|  7876 | ADAMS  | CLERK     | 7788 | 1987-05-23 | 1100.00 |    NULL |     20 |
|  7900 | JAMES  | CLERK     | 7698 | 1981-12-03 |  950.00 |    NULL |     30 |
|  7902 | FORD   | ANALYST   | 7566 | 1981-12-03 | 3000.00 |    NULL |     20 |
|  7934 | MILLER | CLERK     | 7782 | 1982-01-23 | 1300.00 |    NULL |     10 |
+-------+--------+-----------+------+------------+---------+---------+--------+
14 rows in set (0.00 sec)
```



不看表中数据，只看表的结构：desc emp;（describe emp）

```bash
mysql> desc emp;
+----------+-------------+------+-----+---------+-------+
| Field    | Type        | Null | Key | Default | Extra |
+----------+-------------+------+-----+---------+-------+
| EMPNO    | int         | NO   | PRI | NULL    |       |
| ENAME    | varchar(10) | YES  |     | NULL    |       |
| JOB      | varchar(9)  | YES  |     | NULL    |       |
| MGR      | int         | YES  |     | NULL    |       |
| HIREDATE | date        | YES  |     | NULL    |       |
| SAL      | double(7,2) | YES  |     | NULL    |       |
| COMM     | double(7,2) | YES  |     | NULL    |       |
| DEPTNO   | int         | YES  |     | NULL    |       |
+----------+-------------+------+-----+---------+-------+
8 rows in set (0.01 sec)
```



### 二、SQL语句

### DQL（数据查询语言）

#### 1.简单查询

1.1. 查询一个字段

```sql
select 字段名 from 表名;
```

```bash
mysql> select * from dept;
+--------+------------+----------+
| DEPTNO | DNAME      | LOC      |
+--------+------------+----------+
|     10 | ACCOUNTING | NEW YORK |
|     20 | RESEARCH   | DALLAS   |
|     30 | SALES      | CHICAGO  |
|     40 | OPERATIONS | BOSTON   |
+--------+------------+----------+
4 rows in set (0.00 sec)

mysql> select dname from dept;
+------------+
| dname      |
+------------+
| ACCOUNTING |
| RESEARCH   |
| SALES      |
| OPERATIONS |
+------------+
4 rows in set (0.00 sec)
```



1.2. 查询多个字段

对字段名使用逗号隔开，如：

```sql
select 字段1,字段2 from 表名;
```

```bash
mysql> select deptno,dname from dept;
+--------+------------+
| deptno | dname      |
+--------+------------+
|     10 | ACCOUNTING |
|     20 | RESEARCH   |
|     30 | SALES      |
|     40 | OPERATIONS |
+--------+------------+
4 rows in set (0.00 sec)
```



1.3. 查询所有字段（尽量不要写到代码里面，效率低，可读性差）

```sql
select * from 表名;
```



1.4. 给查询的列取别名

```sql
select dname as deptname from dept;
select dname deptname from dept;
```

使用as关键字起别名（也可以省略），更改列名dname为deptname，只是将显示的查询结果列名显示为deptname，原列表还是叫：dname。select语句永远不会进行修改操作，只负责查询。

```bash
mysql> select dname as deptname from dept;
+------------+
| deptname   |
+------------+
| ACCOUNTING |
| RESEARCH   |
| SALES      |
| OPERATIONS |
+------------+
4 rows in set (0.00 sec)

mysql> select * from dept;
+--------+------------+----------+
| DEPTNO | DNAME      | LOC      |
+--------+------------+----------+
|     10 | ACCOUNTING | NEW YORK |
|     20 | RESEARCH   | DALLAS   |
|     30 | SALES      | CHICAGO  |
|     40 | OPERATIONS | BOSTON   |
+--------+------------+----------+
4 rows in set (0.00 sec)
```

取别名时，里面有空格，可以用单引号（或者双引号）括起别名：

```sql
select dname 'dept name' from dept;
```

在所有数据库中，字符串统一使用单引号括起来，单引号是标准，双引号在oracle数据库中用不了，但是在mysql中可以使用。



1.5. 列参与运算，比如计算员工的年薪=月薪*12

```bash
mysql> select * from emp;
+-------+--------+-----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB       | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+-----------+------+------------+---------+---------+--------+
|  7369 | SMITH  | CLERK     | 7902 | 1980-12-17 |  800.00 |    NULL |     20 |
|  7499 | ALLEN  | SALESMAN  | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN  | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER   | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN  | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER   | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER   | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7788 | SCOTT  | ANALYST   | 7566 | 1987-04-19 | 3000.00 |    NULL |     20 |
|  7839 | KING   | PRESIDENT | NULL | 1981-11-17 | 5000.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN  | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
|  7876 | ADAMS  | CLERK     | 7788 | 1987-05-23 | 1100.00 |    NULL |     20 |
|  7900 | JAMES  | CLERK     | 7698 | 1981-12-03 |  950.00 |    NULL |     30 |
|  7902 | FORD   | ANALYST   | 7566 | 1981-12-03 | 3000.00 |    NULL |     20 |
|  7934 | MILLER | CLERK     | 7782 | 1982-01-23 | 1300.00 |    NULL |     10 |
+-------+--------+-----------+------+------------+---------+---------+--------+
14 rows in set (0.00 sec)

mysql> select ename, sal from emp;
+--------+---------+
| ename  | sal     |
+--------+---------+
| SMITH  |  800.00 |
| ALLEN  | 1600.00 |
| WARD   | 1250.00 |
| JONES  | 2975.00 |
| MARTIN | 1250.00 |
| BLAKE  | 2850.00 |
| CLARK  | 2450.00 |
| SCOTT  | 3000.00 |
| KING   | 5000.00 |
| TURNER | 1500.00 |
| ADAMS  | 1100.00 |
| JAMES  |  950.00 |
| FORD   | 3000.00 |
| MILLER | 1300.00 |
+--------+---------+
14 rows in set (0.00 sec)
```

```bash
mysql> select ename, sal*12 from emp;
+--------+----------+
| ename  | sal*12   |
+--------+----------+
| SMITH  |  9600.00 |
| ALLEN  | 19200.00 |
| WARD   | 15000.00 |
| JONES  | 35700.00 |
| MARTIN | 15000.00 |
| BLAKE  | 34200.00 |
| CLARK  | 29400.00 |
| SCOTT  | 36000.00 |
| KING   | 60000.00 |
| TURNER | 18000.00 |
| ADAMS  | 13200.00 |
| JAMES  | 11400.00 |
| FORD   | 36000.00 |
| MILLER | 15600.00 |
+--------+----------+
14 rows in set (0.01 sec)
```

```bash
mysql> select ename, sal*12 as year_sal from emp;
+--------+----------+
| ename  | year_sal |
+--------+----------+
| SMITH  |  9600.00 |
| ALLEN  | 19200.00 |
| WARD   | 15000.00 |
| JONES  | 35700.00 |
| MARTIN | 15000.00 |
| BLAKE  | 34200.00 |
| CLARK  | 29400.00 |
| SCOTT  | 36000.00 |
| KING   | 60000.00 |
| TURNER | 18000.00 |
| ADAMS  | 13200.00 |
| JAMES  | 11400.00 |
| FORD   | 36000.00 |
| MILLER | 15600.00 |
+--------+----------+
14 rows in set (0.00 sec)
```



#### 2.条件查询

```sql
select
    字段1，字段2，字段3，...
from
    表名
where
    条件;
```

例如在emp表中查询：

查询薪资等于800的员工姓名和编号：

```bash
mysql> select empno,ename from emp where sal=800;
+-------+-------+
| empno | ename |
+-------+-------+
|  7369 | SMITH |
+-------+-------+
1 row in set (0.01 sec)
```

查询薪资不等于800的员工姓名和编号：

```bash
mysql> select empno, ename from emp where sal!=800;
+-------+--------+
| empno | ename  |
+-------+--------+
|  7499 | ALLEN  |
|  7521 | WARD   |
|  7566 | JONES  |
|  7654 | MARTIN |
|  7698 | BLAKE  |
|  7782 | CLARK  |
|  7788 | SCOTT  |
|  7839 | KING   |
|  7844 | TURNER |
|  7876 | ADAMS  |
|  7900 | JAMES  |
|  7902 | FORD   |
|  7934 | MILLER |
+-------+--------+
13 rows in set (0.01 sec)

mysql> select empno, ename from emp where sal<>800;
+-------+--------+
| empno | ename  |
+-------+--------+
|  7499 | ALLEN  |
|  7521 | WARD   |
|  7566 | JONES  |
|  7654 | MARTIN |
|  7698 | BLAKE  |
|  7782 | CLARK  |
|  7788 | SCOTT  |
|  7839 | KING   |
|  7844 | TURNER |
|  7876 | ADAMS  |
|  7900 | JAMES  |
|  7902 | FORD   |
|  7934 | MILLER |
+-------+--------+
13 rows in set (0.00 sec)
```

查询SMITH的员工编号和薪资：

```bash
 mysql> select empno, sal from emp where ename='smith';
+-------+--------+
| empno | sal    |
+-------+--------+
|  7369 | 800.00 |
+-------+--------+
1 row in set (0.00 sec)
```

查询薪资在2450到3000之间（包括2450和3000）的员工信息：

```bash
mysql> select empno,ename,sal from emp where sal>=2450 and sal<=3000;
+-------+-------+---------+
| empno | ename | sal     |
+-------+-------+---------+
|  7566 | JONES | 2975.00 |
|  7698 | BLAKE | 2850.00 |
|  7782 | CLARK | 2450.00 |
|  7788 | SCOTT | 3000.00 |
|  7902 | FORD  | 3000.00 |
+-------+-------+---------+
5 rows in set (0.00 sec)
```

查询哪些员工的津贴为null：

```bash
mysql> select empno,ename,sal,comm from emp where comm is null;
+-------+--------+---------+------+
| empno | ename  | sal     | comm |
+-------+--------+---------+------+
|  7369 | SMITH  |  800.00 | NULL |
|  7566 | JONES  | 2975.00 | NULL |
|  7698 | BLAKE  | 2850.00 | NULL |
|  7782 | CLARK  | 2450.00 | NULL |
|  7788 | SCOTT  | 3000.00 | NULL |
|  7839 | KING   | 5000.00 | NULL |
|  7876 | ADAMS  | 1100.00 | NULL |
|  7900 | JAMES  |  950.00 | NULL |
|  7902 | FORD   | 3000.00 | NULL |
|  7934 | MILLER | 1300.00 | NULL |
+-------+--------+---------+------+
10 rows in set (0.00 sec)
```

在数据库中，null不能使用等号进行衡量，需要使用is null。null在数据库中表示什么也没有，不是一个值。

查询哪些员工的津贴不为null：

```bash
mysql> select empno,ename,sal,comm from emp where comm is not null;
+-------+--------+---------+---------+
| empno | ename  | sal     | comm    |
+-------+--------+---------+---------+
|  7499 | ALLEN  | 1600.00 |  300.00 |
|  7521 | WARD   | 1250.00 |  500.00 |
|  7654 | MARTIN | 1250.00 | 1400.00 |
|  7844 | TURNER | 1500.00 |    0.00 |
+-------+--------+---------+---------+
4 rows in set (0.00 sec)
```

查询工资大于2500，并且部门编号为10或20的员工：（涉及到and和or优先级问题）

```bash
mysql> select * from emp where sal >2500 and deptno = 10 or deptno = 20;
+-------+-------+-----------+------+------------+---------+------+--------+
| EMPNO | ENAME | JOB       | MGR  | HIREDATE   | SAL     | COMM | DEPTNO |
+-------+-------+-----------+------+------------+---------+------+--------+
|  7369 | SMITH | CLERK     | 7902 | 1980-12-17 |  800.00 | NULL |     20 |
|  7566 | JONES | MANAGER   | 7839 | 1981-04-02 | 2975.00 | NULL |     20 |
|  7788 | SCOTT | ANALYST   | 7566 | 1987-04-19 | 3000.00 | NULL |     20 |
|  7839 | KING  | PRESIDENT | NULL | 1981-11-17 | 5000.00 | NULL |     10 |
|  7876 | ADAMS | CLERK     | 7788 | 1987-05-23 | 1100.00 | NULL |     20 |
|  7902 | FORD  | ANALYST   | 7566 | 1981-12-03 | 3000.00 | NULL |     20 |
+-------+-------+-----------+------+------------+---------+------+--------+
6 rows in set (0.00 sec)
```

这个语句表示工资大于2500并且部门编号为10，或者20部门所有员工。因为and优先级高于or，修正如下：

```bash
mysql> select * from emp where sal >2500 and (deptno = 10 or deptno = 20);
+-------+-------+-----------+------+------------+---------+------+--------+
| EMPNO | ENAME | JOB       | MGR  | HIREDATE   | SAL     | COMM | DEPTNO |
+-------+-------+-----------+------+------------+---------+------+--------+
|  7566 | JONES | MANAGER   | 7839 | 1981-04-02 | 2975.00 | NULL |     20 |
|  7788 | SCOTT | ANALYST   | 7566 | 1987-04-19 | 3000.00 | NULL |     20 |
|  7839 | KING  | PRESIDENT | NULL | 1981-11-17 | 5000.00 | NULL |     10 |
|  7902 | FORD  | ANALYST   | 7566 | 1981-12-03 | 3000.00 | NULL |     20 |
+-------+-------+-----------+------+------------+---------+------+--------+
4 rows in set (0.00 sec)
```

查询工作岗位是manager或者salesman的员工：

```bash
mysql> select * from emp where job='manager' or job='salesman';
+-------+--------+----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB      | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+----------+------+------------+---------+---------+--------+
|  7499 | ALLEN  | SALESMAN | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER  | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER  | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER  | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
+-------+--------+----------+------+------------+---------+---------+--------+
7 rows in set (0.00 sec)

mysql> select * from emp where job in ('manager','salesman');
+-------+--------+----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB      | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+----------+------+------------+---------+---------+--------+
|  7499 | ALLEN  | SALESMAN | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER  | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER  | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER  | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
+-------+--------+----------+------+------------+---------+---------+--------+
7 rows in set (0.00 sec)
```

in 相当于多个or，后面接具体的值，而不是一个区间，比如下面查询语句表示找出800和5000薪资，而不是800到5000。

```bash
select * from emp where sal in (800,5000);
```



模糊查询like：支持%和下划线匹配，%匹配任意多个字符，_匹配一个字符：

查询名字里面含有o的员工：

```bash
mysql> select ename from emp where ename like '%o%';
+-------+
| ename |
+-------+
| JONES |
| SCOTT |
| FORD  |
+-------+
3 rows in set (0.00 sec)
```

查询名字第三个字母是r的员工：

```bash
mysql> select ename from emp where ename like '__r%';
+--------+
| ename  |
+--------+
| WARD   |
| MARTIN |
| TURNER |
| FORD   |
+--------+
4 rows in set (0.00 sec)
```

查询名字中有_的员工：

```sql
select ename from tmp where ename like '%\_%'; # \表示转义
```



#### 3.排序

3.1. 查询所有员工薪资并排序

```bash
mysql> select
    ->     ename,sal
    -> from 
    ->     emp
    -> order by
    ->     sal;
+--------+---------+
| ename  | sal     |
+--------+---------+
| SMITH  |  800.00 |
| JAMES  |  950.00 |
| ADAMS  | 1100.00 |
| WARD   | 1250.00 |
| MARTIN | 1250.00 |
| MILLER | 1300.00 |
| TURNER | 1500.00 |
| ALLEN  | 1600.00 |
| CLARK  | 2450.00 |
| BLAKE  | 2850.00 |
| JONES  | 2975.00 |
| SCOTT  | 3000.00 |
| FORD   | 3000.00 |
| KING   | 5000.00 |
+--------+---------+
14 rows in set (0.00 sec)
```

```sql
select ename,sal from emp order by sal desc; # 降序
select ename,sal from emp order by sal asc; # 升序
```



3.2.按照多个字段进行排序

如查询员工名字和薪资，要求按照薪资升序；如果薪资一样，再按照名字升序排列。

```bash
mysql> select ename,sal from emp order by sal asc, ename asc;
+--------+---------+
| ename  | sal     |
+--------+---------+
| SMITH  |  800.00 |
| JAMES  |  950.00 |
| ADAMS  | 1100.00 |
| MARTIN | 1250.00 |
| WARD   | 1250.00 |
| MILLER | 1300.00 |
| TURNER | 1500.00 |
| ALLEN  | 1600.00 |
| CLARK  | 2450.00 |
| BLAKE  | 2850.00 |
| JONES  | 2975.00 |
| FORD   | 3000.00 |
| SCOTT  | 3000.00 |
| KING   | 5000.00 |
+--------+---------+
14 rows in set (0.00 sec)
```

sal在前起主导作用，只有当sal相等的时候才会考虑启用ename排序。



3.3.根据字段的位置排序

```sql
select ename,sal from emp order by 2;
```

按照查询结果的第2列进行排序，不建议再开发中这样写。



#### 4.综合案例

找出工资在1250到3000之间的员工信息，要求按照薪资降序排列。

```bash
mysql> select 
    ->     ename,sal
    -> from
    ->     emp
    -> where
    ->     sal between 1250 and 3000
    -> order by
    ->     sal desc;
+--------+---------+
| ename  | sal     |
+--------+---------+
| SCOTT  | 3000.00 |
| FORD   | 3000.00 |
| JONES  | 2975.00 |
| BLAKE  | 2850.00 |
| CLARK  | 2450.00 |
| ALLEN  | 1600.00 |
| TURNER | 1500.00 |
| MILLER | 1300.00 |
| WARD   | 1250.00 |
| MARTIN | 1250.00 |
+--------+---------+
10 rows in set (0.00 sec)
```

关键字顺序不能变化，即先查询出来再排序。



#### 5.数据处理函数

5.1.数据处理函数又称为单行处理函数（一个输入对应一个输出）。和单行处理函数相对的是多行处理函数（多个输入对应一个输出）。

5.2.常用的单行处理函数

（1）转换大小写lower()和upper()

```sql
select lower(ename) as ename from emp; # 转换小写
select upper(ename) as ename from emp; # 转换大写
```



（2）取子串substrate()

```sql
select substr(ename, x, y) as ename from emp; # 取子串，x是起始位置从1开始，y是要取的长度
```

找出员工名字第一个字母是'A'的员工信息

第一种方法：模糊查询

```sql
select ename from emp where ename like 'A%';
```

第二种方法：使用substr()

```sql
select ename from emp where substr(ename, 1, 1) = 'A';
```



（3）字符串长度length()

```sql
select length(ename) as enamelength from emp;
```



（4）字符串拼接concat()

```sql
select concat(empno,ename) from emp;
```

首字母大写：

```bash
mysql> select concat(substr(ename,1,1), lower(substr(ename,2,length(ename)-1))) as ename from emp;
+--------+
| ename  |
+--------+
| Smith  |
| Allen  |
| Ward   |
| Jones  |
| Martin |
| Blake  |
| Clark  |
| Scott  |
| King   |
| Turner |
| Adams  |
| James  |
| Ford   |
| Miller |
+--------+
14 rows in set (0.00 sec)
```



（5）去空格trim()

```bash
mysql> select * from emp where ename = trim('  KING');
+-------+-------+-----------+------+------------+---------+------+--------+
| EMPNO | ENAME | JOB       | MGR  | HIREDATE   | SAL     | COMM | DEPTNO |
+-------+-------+-----------+------+------------+---------+------+--------+
|  7839 | KING  | PRESIDENT | NULL | 1981-11-17 | 5000.00 | NULL |     10 |
+-------+-------+-----------+------+------------+---------+------+--------+
1 row in set (0.01 sec)
```



（6）日期处理函数

str_to_date()：将字符串转换成日期

date_format()：将日期转换成特定格式的字符串



（7）四舍五入函数round()

（8）生成随机数函数rand()

（9）ifnull()函数可以将null数据转换成一个具体值

在数据库中，只要有null参与的运算，结果都是null，如：

```bash
mysql> select ename,sal+comm as salcomm from emp;
+--------+---------+
| ename  | salcomm |
+--------+---------+
| SMITH  |    NULL |
| ALLEN  | 1900.00 |
| WARD   | 1750.00 |
| JONES  |    NULL |
| MARTIN | 2650.00 |
| BLAKE  |    NULL |
| CLARK  |    NULL |
| SCOTT  |    NULL |
| KING   |    NULL |
| TURNER | 1500.00 |
| ADAMS  |    NULL |
| JAMES  |    NULL |
| FORD   |    NULL |
| MILLER |    NULL |
+--------+---------+
14 rows in set (0.00 sec)
```

计算每个员工的年薪：

```bash
mysql> select ename, (sal + comm) * 12 as yeahsal from emp;
+--------+----------+
| ename  | yeahsal  |
+--------+----------+
| SMITH  |     NULL |
| ALLEN  | 22800.00 |
| WARD   | 21000.00 |
| JONES  |     NULL |
| MARTIN | 31800.00 |
| BLAKE  |     NULL |
| CLARK  |     NULL |
| SCOTT  |     NULL |
| KING   |     NULL |
| TURNER | 18000.00 |
| ADAMS  |     NULL |
| JAMES  |     NULL |
| FORD   |     NULL |
| MILLER |     NULL |
+--------+----------+
14 rows in set (0.00 sec)
```

使用ifnull函数来计算每个员工的年薪：

```sql
ifnull(x,y) # 当x为null时，把数据x当作y
```

```bash
mysql> select ename, (sal + ifnull(comm,0)) * 12 as yeahsal from emp;
+--------+----------+
| ename  | yeahsal  |
+--------+----------+
| SMITH  |  9600.00 |
| ALLEN  | 22800.00 |
| WARD   | 21000.00 |
| JONES  | 35700.00 |
| MARTIN | 31800.00 |
| BLAKE  | 34200.00 |
| CLARK  | 29400.00 |
| SCOTT  | 36000.00 |
| KING   | 60000.00 |
| TURNER | 18000.00 |
| ADAMS  | 13200.00 |
| JAMES  | 11400.00 |
| FORD   | 36000.00 |
| MILLER | 15600.00 |
+--------+----------+
14 rows in set (0.00 sec)
```



（10）case函数

```sql
# 简单case，枚举某个字段（或者是该字段的函数）所有可能的值
case <col_name>
    when <value1> then <result1>
    when <value2> then <result2>
    ...
    else <resule>
end

# case搜索函数
case
    when <condition1> then <result1>
    when <condition2> then <result2>
    ...
    else <result>
end
```

当员工的工作岗位是MANAGE时，工资上调10%，其他正常：

```sql
select
    ename,job,sal as oldsal, (case job when 'MANAGE' then sal*1.1 else sal end) as newsal
from
    emp;
```

```bash
+--------+-----------+---------+---------+
| ename  | job       | oldsal  | newsal  |
+--------+-----------+---------+---------+
| SMITH  | CLERK     |  800.00 |  800.00 |
| ALLEN  | SALESMAN  | 1600.00 | 1600.00 |
| WARD   | SALESMAN  | 1250.00 | 1250.00 |
| JONES  | MANAGER   | 2975.00 | 2975.00 |
| MARTIN | SALESMAN  | 1250.00 | 1250.00 |
| BLAKE  | MANAGER   | 2850.00 | 2850.00 |
| CLARK  | MANAGER   | 2450.00 | 2450.00 |
| SCOTT  | ANALYST   | 3000.00 | 3000.00 |
| KING   | PRESIDENT | 5000.00 | 5000.00 |
| TURNER | SALESMAN  | 1500.00 | 1500.00 |
| ADAMS  | CLERK     | 1100.00 | 1100.00 |
| JAMES  | CLERK     |  950.00 |  950.00 |
| FORD   | ANALYST   | 3000.00 | 3000.00 |
| MILLER | CLERK     | 1300.00 | 1300.00 |
+--------+-----------+---------+---------+
14 rows in set (0.00 sec)
```



5.3.多行处理函数（分组函数）

5个：count计数，sum求和，avg平均值，max最大值，min最小值

分组函数在使用的时候必须先进行分组，然后才能用。如果没有对数据进行分组，整张表默认为一组。



（1）找出最高工资：

```bash
mysql> select max(sal) as max_val from emp;
+---------+
| max_val |
+---------+
| 5000.00 |
+---------+
1 row in set (0.06 sec)
```

（2）计算员工数量：

```bash
mysql> select count(ename) as count_emp from emp;
+-----------+
| count_emp |
+-----------+
|        14 |
+-----------+
1 row in set (0.01 sec)
```

（3）计算员工补助和：（分组函数自动忽略null）

```bash
mysql> select * from emp;
+-------+--------+-----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB       | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+-----------+------+------------+---------+---------+--------+
|  7369 | SMITH  | CLERK     | 7902 | 1980-12-17 |  800.00 |    NULL |     20 |
|  7499 | ALLEN  | SALESMAN  | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN  | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER   | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN  | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER   | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER   | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7788 | SCOTT  | ANALYST   | 7566 | 1987-04-19 | 3000.00 |    NULL |     20 |
|  7839 | KING   | PRESIDENT | NULL | 1981-11-17 | 5000.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN  | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
|  7876 | ADAMS  | CLERK     | 7788 | 1987-05-23 | 1100.00 |    NULL |     20 |
|  7900 | JAMES  | CLERK     | 7698 | 1981-12-03 |  950.00 |    NULL |     30 |
|  7902 | FORD   | ANALYST   | 7566 | 1981-12-03 | 3000.00 |    NULL |     20 |
|  7934 | MILLER | CLERK     | 7782 | 1982-01-23 | 1300.00 |    NULL |     10 |
+-------+--------+-----------+------+------------+---------+---------+--------+
14 rows in set (0.00 sec)

mysql> select sum(comm) as sum_comm from emp;
+----------+
| sum_comm |
+----------+
|  2200.00 |
+----------+
1 row in set (0.00 sec)
```

count(具体字段)：统计该字段下所有不为NULL的元素总数；

count(*)：统计表中总行数，只要有一行数据，则count++；

在数据库中每一行记录不可能都为NULL，一行数据有一列不为NULL，则这行数据就是有效的。

```bash
mysql> select count(*) from emp;
+----------+
| count(*) |
+----------+
|       14 |
+----------+
1 row in set (0.01 sec)

mysql> select count(comm) from emp;
+-------------+
| count(comm) |
+-------------+
|           4 |
+-------------+
1 row in set (0.00 sec)
```

（4）分组函数不能直接用在where子句中：

找出比最低工资高的员工信息：

```sql
select ename,sal from emp where sal > min(sal); # 报错，在执行where时，还没有分组，而分组函数必须分组才可以使用
```

where在执行的时候还没有分组，所以where后面不能出现分组函数。

但是select max(sal) from emp;可以执行，因为select在执行的时候，group by已经结束了（默认整张表），所以分组函数可以在select后面执行。



（5）分组函数可以组合起来一起用

```sql
mysql> select sum(sal),min(sal),avg(sal) from emp;
+----------+----------+-------------+
| sum(sal) | min(sal) | avg(sal)    |
+----------+----------+-------------+
| 29025.00 |   800.00 | 2073.214286 |
+----------+----------+-------------+
1 row in set (0.00 sec)
```



#### 6.分组查询（非常重要）

```sql
select
    ...
from
    ...
where
    ...
group by # 分组查询
    ...
order by
    ...
```

**执行顺序：1.from；2.where；3.group by；4.select；5.order by**

（1）计算每个工作岗位的最高薪资？

```bash
mysql> select job,max(sal) from emp group by job;
+-----------+----------+
| job       | max(sal) |
+-----------+----------+
| CLERK     |  1300.00 |
| SALESMAN  |  1600.00 |
| MANAGER   |  2975.00 |
| ANALYST   |  3000.00 |
| PRESIDENT |  5000.00 |
+-----------+----------+
5 rows in set (0.00 sec)

mysql> select job,max(sal),count(ename) from emp group by job;
+-----------+----------+--------------+
| job       | max(sal) | count(ename) |
+-----------+----------+--------------+
| CLERK     |  1300.00 |            4 |
| SALESMAN  |  1600.00 |            4 |
| MANAGER   |  2975.00 |            3 |
| ANALYST   |  3000.00 |            2 |
| PRESIDENT |  5000.00 |            1 |
+-----------+----------+--------------+
5 rows in set (0.01 sec)
```

**在一条select语句中，如果有group by语句，则select后面只能跟：参加分组的字段，以及分组函数，其他一律不能跟。**



（2）找出每个部门的最高薪资

```bash
mysql> select deptno,max(sal) from emp group by deptno;
+--------+----------+
| deptno | max(sal) |
+--------+----------+
|     20 |  3000.00 |
|     30 |  2850.00 |
|     10 |  5000.00 |
+--------+----------+
3 rows in set (0.00 sec)
```



（3）找出每个部门，不同工作岗位的最高薪资

```bash
mysql> select deptno,job,max(sal) from emp group by deptno,job order by deptno;
+--------+-----------+----------+
| deptno | job       | max(sal) |
+--------+-----------+----------+
|     10 | CLERK     |  1300.00 |
|     10 | MANAGER   |  2450.00 |
|     10 | PRESIDENT |  5000.00 |
|     20 | ANALYST   |  3000.00 |
|     20 | CLERK     |  1100.00 |
|     20 | MANAGER   |  2975.00 |
|     30 | CLERK     |   950.00 |
|     30 | MANAGER   |  2850.00 |
|     30 | SALESMAN  |  1600.00 |
+--------+-----------+----------+
9 rows in set (0.00 sec)
```



（4）找出每个部门均值薪资，要求显示均值薪资大于2000的

使用having子句，对分完组之后的数据进一步过滤，havin不能代替where，并且不能单独使用，必须和group by联合使用。

第一步：找出每个部门平均薪资

```bash
mysql> select deptno,avg(sal) from emp group by deptno;
+--------+-------------+
| deptno | avg(sal)    |
+--------+-------------+
|     20 | 2175.000000 |
|     30 | 1566.666667 |
|     10 | 2916.666667 |
+--------+-------------+
3 rows in set (0.00 sec)
```

第二步：过滤

```bash
mysql> select deptno,avg(sal) from emp group by deptno having avg(sal)>2000;
+--------+-------------+
| deptno | avg(sal)    |
+--------+-------------+
|     20 | 2175.000000 |
|     10 | 2916.666667 |
+--------+-------------+
2 rows in set (0.00 sec)
```

having效率不如where，如果where完成不了的，则需要使用having，比如上述的均值。



#### 7.单表查询大总结

单表查询：

```sql
select
    ...
from
    ...
where
    ...
group by # 分组查询
    ...
having
    ...
order by
    ...
```

**执行顺序：1.from；2.where；3.group by；4.having；5.select；6.order by**

找出每个岗位的平均薪资，要求显示平均薪资大于1500的，除MANAGE岗位之外，要求按照平均薪资降序排列。

```bash
mysql> select job,avg(sal) as avg_sal from emp where job!='manager' group by job having avg(sal)>1500 order by avg(sal) desc;
+-----------+-------------+
| job       | avg_sal     |
+-----------+-------------+
| PRESIDENT | 5000.000000 |
| ANALYST   | 3000.000000 |
+-----------+-------------+
2 rows in set (0.00 sec)
```



#### 8.去除重复记录

使用distinct关键字去重，只能出现在所有字段的最前方

```bash
mysql> select distinct job from emp;
+-----------+
| job       |
+-----------+
| CLERK     |
| SALESMAN  |
| MANAGER   |
| ANALYST   |
| PRESIDENT |
+-----------+
5 rows in set (0.00 sec)

mysql> select distinct job,deptno from emp;
+-----------+--------+
| job       | deptno |
+-----------+--------+
| CLERK     |     20 |
| SALESMAN  |     30 |
| MANAGER   |     20 |
| MANAGER   |     30 |
| MANAGER   |     10 |
| ANALYST   |     20 |
| PRESIDENT |     10 |
| CLERK     |     30 |
| CLERK     |     10 |
+-----------+--------+
9 rows in set (0.00 sec)
```

统计工作岗位的数量：

```bash
mysql> select count(distinct job) from emp;
+---------------------+
| count(distinct job) |
+---------------------+
|                   5 |
+---------------------+
1 row in set (0.01 sec)
```



#### 9.连接查询（非常重要）

（1）定义

从一张表中单独查询，成为单表查询。emp表和dept表联合起来查询数据，从emp表中取员工名字，从dept表中取部门名字，这种跨表查询，多张表联合起来查询数据成为连接查询。

（2）分类

根据表连接方式分类：

内连接：：等值连接，非等值连接，自连接

外连接：左外连接（左连接），右外连接（右连接）

全连接：（不讲了）

（3）当两张表进行连接查询时，没有任何条件的限制会发生什么现象？

查询每个员工所在的部门名称：

```bash
mysql> select ename,dname from emp,dept;
+--------+------------+
| ename  | dname      |
+--------+------------+
| SMITH  | OPERATIONS |
| SMITH  | SALES      |
| SMITH  | RESEARCH   |
| SMITH  | ACCOUNTING |
| ALLEN  | OPERATIONS |
| ALLEN  | SALES      |
| ALLEN  | RESEARCH   |
| ALLEN  | ACCOUNTING |
| WARD   | OPERATIONS |
| WARD   | SALES      |
| WARD   | RESEARCH   |
| WARD   | ACCOUNTING |
| JONES  | OPERATIONS |
| JONES  | SALES      |
| JONES  | RESEARCH   |
| JONES  | ACCOUNTING |
| MARTIN | OPERATIONS |
| MARTIN | SALES      |
| MARTIN | RESEARCH   |
| MARTIN | ACCOUNTING |
| BLAKE  | OPERATIONS |
| BLAKE  | SALES      |
| BLAKE  | RESEARCH   |
| BLAKE  | ACCOUNTING |
| CLARK  | OPERATIONS |
| CLARK  | SALES      |
| CLARK  | RESEARCH   |
| CLARK  | ACCOUNTING |
| SCOTT  | OPERATIONS |
| SCOTT  | SALES      |
| SCOTT  | RESEARCH   |
| SCOTT  | ACCOUNTING |
| KING   | OPERATIONS |
| KING   | SALES      |
| KING   | RESEARCH   |
| KING   | ACCOUNTING |
| TURNER | OPERATIONS |
| TURNER | SALES      |
| TURNER | RESEARCH   |
| TURNER | ACCOUNTING |
| ADAMS  | OPERATIONS |
| ADAMS  | SALES      |
| ADAMS  | RESEARCH   |
| ADAMS  | ACCOUNTING |
| JAMES  | OPERATIONS |
| JAMES  | SALES      |
| JAMES  | RESEARCH   |
| JAMES  | ACCOUNTING |
| FORD   | OPERATIONS |
| FORD   | SALES      |
| FORD   | RESEARCH   |
| FORD   | ACCOUNTING |
| MILLER | OPERATIONS |
| MILLER | SALES      |
| MILLER | RESEARCH   |
| MILLER | ACCOUNTING |
+--------+------------+
56 rows in set (0.00 sec)
```

当两张表进行连接查询，没有任何条件限制的时候，最终查询结果条数是两张表条数的乘积，这种现象称为笛卡尔积现象。



（4）避免笛卡尔积现象

连接时加条件，满足这个条件的记录被筛选出来：

```bash
mysql> select ename,dname from emp,dept where emp.deptno=dept.deptno;
+--------+------------+
| ename  | dname      |
+--------+------------+
| SMITH  | RESEARCH   |
| ALLEN  | SALES      |
| WARD   | SALES      |
| JONES  | RESEARCH   |
| MARTIN | SALES      |
| BLAKE  | SALES      |
| CLARK  | ACCOUNTING |
| SCOTT  | RESEARCH   |
| KING   | ACCOUNTING |
| TURNER | SALES      |
| ADAMS  | RESEARCH   |
| JAMES  | SALES      |
| FORD   | RESEARCH   |
| MILLER | ACCOUNTING |
+--------+------------+
14 rows in set (0.01 sec)
```

最终查询的结果条数是14条，但是在匹配的过程中，匹配的次数还是56次。

表起别名：（SQL92语法）

```bash
select e.ename,d.dname from emp e,dept d where e.deptno=d.deptno;
+--------+------------+
| ename  | dname      |
+--------+------------+
| SMITH  | RESEARCH   |
| ALLEN  | SALES      |
| WARD   | SALES      |
| JONES  | RESEARCH   |
| MARTIN | SALES      |
| BLAKE  | SALES      |
| CLARK  | ACCOUNTING |
| SCOTT  | RESEARCH   |
| KING   | ACCOUNTING |
| TURNER | SALES      |
| ADAMS  | RESEARCH   |
| JAMES  | SALES      |
| FORD   | RESEARCH   |
| MILLER | ACCOUNTING |
+--------+------------+
14 rows in set (0.00 sec)
```



#### 10.内连接

（1）等值连接（筛选条件是等值关系，所以是等值连接）

查询每个员工所在的部门名称，显示员工名与部门名？

对emp e表和dept d表进行连接，条件是e.deptno=d.deptno

SQL92写法：

```bash
select e.ename,d.dname from emp e,dept d where e.deptno=d.deptno;
+--------+------------+
| ename  | dname      |
+--------+------------+
| SMITH  | RESEARCH   |
| ALLEN  | SALES      |
| WARD   | SALES      |
| JONES  | RESEARCH   |
| MARTIN | SALES      |
| BLAKE  | SALES      |
| CLARK  | ACCOUNTING |
| SCOTT  | RESEARCH   |
| KING   | ACCOUNTING |
| TURNER | SALES      |
| ADAMS  | RESEARCH   |
| JAMES  | SALES      |
| FORD   | RESEARCH   |
| MILLER | ACCOUNTING |
+--------+------------+
14 rows in set (0.00 sec)
```

SQL92：表连接的条件在where后面，和后期进一步筛选的条件都放到了where后面，导致结构不清晰。

SQL99写法：

```bash
mysql> select e.ename,d.dname from emp e join dept d on e.deptno=d.deptno;
+--------+------------+
| ename  | dname      |
+--------+------------+
| SMITH  | RESEARCH   |
| ALLEN  | SALES      |
| WARD   | SALES      |
| JONES  | RESEARCH   |
| MARTIN | SALES      |
| BLAKE  | SALES      |
| CLARK  | ACCOUNTING |
| SCOTT  | RESEARCH   |
| KING   | ACCOUNTING |
| TURNER | SALES      |
| ADAMS  | RESEARCH   |
| JAMES  | SALES      |
| FORD   | RESEARCH   |
| MILLER | ACCOUNTING |
+--------+------------+
14 rows in set (0.07 sec)
```

SQL99：表连接的条件是独立的，连接之后如果进一步筛选，则往后继续添加where。

SQL99语法的内连接：

```sql
select
    ...
from
    a
inner join #（inner可以省略，加上可读性更好）
    b
on
    a和b筛选条件
where
    筛选条件
...

```



（2）非等值连接（条件不是一个等量关系，称为非等值连接）

找出每个员工的薪资等级，要求显示员工名、薪资、薪资等级？

```bash
mysql> select e.ename,e.sal,s.grade from emp e join salgrade s on e.sal>=s.losal and e.sal<=s.hisal;
+--------+---------+-------+
| ename  | sal     | grade |
+--------+---------+-------+
| SMITH  |  800.00 |     1 |
| ALLEN  | 1600.00 |     3 |
| WARD   | 1250.00 |     2 |
| JONES  | 2975.00 |     4 |
| MARTIN | 1250.00 |     2 |
| BLAKE  | 2850.00 |     4 |
| CLARK  | 2450.00 |     4 |
| SCOTT  | 3000.00 |     4 |
| KING   | 5000.00 |     5 |
| TURNER | 1500.00 |     3 |
| ADAMS  | 1100.00 |     1 |
| JAMES  |  950.00 |     1 |
| FORD   | 3000.00 |     4 |
| MILLER | 1300.00 |     2 |
+--------+---------+-------+
14 rows in set (0.00 sec)
```

或者用between...and...



（3）自连接

查询员工的上级领导，要求显示员工名和对应的领导名？

```bash
mysql> select empno,ename,mgr from emp;
+-------+--------+------+
| empno | ename  | mgr  |
+-------+--------+------+
|  7369 | SMITH  | 7902 |
|  7499 | ALLEN  | 7698 |
|  7521 | WARD   | 7698 |
|  7566 | JONES  | 7839 |
|  7654 | MARTIN | 7698 |
|  7698 | BLAKE  | 7839 |
|  7782 | CLARK  | 7839 |
|  7788 | SCOTT  | 7566 |
|  7839 | KING   | NULL |
|  7844 | TURNER | 7698 |
|  7876 | ADAMS  | 7788 |
|  7900 | JAMES  | 7698 |
|  7902 | FORD   | 7566 |
|  7934 | MILLER | 7782 |
+-------+--------+------+
14 rows in set (0.00 sec)
```

自连接技巧：emp一张表看成两张表，一张作为员工表，一张作为领导表

```bash
mysql> select a.ename as '员工名',b.ename as '领导名' from emp a join emp b on a.mgr=b.empno;
+-----------+-----------+
| 员工名    | 领导名    |
+-----------+-----------+
| SMITH     | FORD      |
| ALLEN     | BLAKE     |
| WARD      | BLAKE     |
| JONES     | KING      |
| MARTIN    | BLAKE     |
| BLAKE     | KING      |
| CLARK     | KING      |
| SCOTT     | JONES     |
| TURNER    | BLAKE     |
| ADAMS     | SCOTT     |
| JAMES     | BLAKE     |
| FORD      | JONES     |
| MILLER    | CLARK     |
+-----------+-----------+
13 rows in set (0.00 sec)
```



#### 11.外连接

内连接特点：完全能够匹配上这个条件的数据查询出来。要连接的两张表是平等关系。

外连接：把没有匹配上条件的数据也查询出来

右外连接（right join）：right代表将join关键字右边的这张表看成主表，主要是为了将这张表的数据全部查询出来，顺便关联查询左边的表。在外连接中，两张表产生了主次关系。

```bash
mysql> select e.ename,d.dname from emp e right join dept d on e.deptno=d.deptno;
+--------+------------+
| ename  | dname      |
+--------+------------+
| MILLER | ACCOUNTING |
| KING   | ACCOUNTING |
| CLARK  | ACCOUNTING |
| FORD   | RESEARCH   |
| ADAMS  | RESEARCH   |
| SCOTT  | RESEARCH   |
| JONES  | RESEARCH   |
| SMITH  | RESEARCH   |
| JAMES  | SALES      |
| TURNER | SALES      |
| BLAKE  | SALES      |
| MARTIN | SALES      |
| WARD   | SALES      |
| ALLEN  | SALES      |
| NULL   | OPERATIONS |
+--------+------------+
15 rows in set (0.00 sec)
```

左外连接：

```bash
mysql> select e.ename,d.dname from dept d left join emp e on e.deptno=d.deptno;
+--------+------------+
| ename  | dname      |
+--------+------------+
| MILLER | ACCOUNTING |
| KING   | ACCOUNTING |
| CLARK  | ACCOUNTING |
| FORD   | RESEARCH   |
| ADAMS  | RESEARCH   |
| SCOTT  | RESEARCH   |
| JONES  | RESEARCH   |
| SMITH  | RESEARCH   |
| JAMES  | SALES      |
| TURNER | SALES      |
| BLAKE  | SALES      |
| MARTIN | SALES      |
| WARD   | SALES      |
| ALLEN  | SALES      |
| NULL   | OPERATIONS |
+--------+------------+
15 rows in set (0.00 sec)

-- 内连接结果
mysql> select e.ename,d.dname from emp e join dept d on e.deptno=d.deptno;
+--------+------------+
| ename  | dname      |
+--------+------------+
| SMITH  | RESEARCH   |
| ALLEN  | SALES      |
| WARD   | SALES      |
| JONES  | RESEARCH   |
| MARTIN | SALES      |
| BLAKE  | SALES      |
| CLARK  | ACCOUNTING |
| SCOTT  | RESEARCH   |
| KING   | ACCOUNTING |
| TURNER | SALES      |
| ADAMS  | RESEARCH   |
| JAMES  | SALES      |
| FORD   | RESEARCH   |
| MILLER | ACCOUNTING |
+--------+------------+
14 rows in set (0.07 sec)
```

和内连接一样，join前面的outer省略了，带着可读性强。

**注意：外连接的查询结果条数一定是大于等于内连接的查询结果条数。**

```bash
mysql> select a.ename as '员工表',b.ename as '领导表' from emp a left join emp b on a.mgr=b.empno;
+-----------+-----------+
| 员工表    | 领导表    |
+-----------+-----------+
| SMITH     | FORD      |
| ALLEN     | BLAKE     |
| WARD      | BLAKE     |
| JONES     | KING      |
| MARTIN    | BLAKE     |
| BLAKE     | KING      |
| CLARK     | KING      |
| SCOTT     | JONES     |
| KING      | NULL      |
| TURNER    | BLAKE     |
| ADAMS     | SCOTT     |
| JAMES     | BLAKE     |
| FORD      | JONES     |
| MILLER    | CLARK     |
+-----------+-----------+
14 rows in set (0.01 sec)

-- 内连接结果
mysql> select a.ename as '员工名',b.ename as '领导名' from emp a join emp b on a.mgr=b.empno;
+-----------+-----------+
| 员工名    | 领导名    |
+-----------+-----------+
| SMITH     | FORD      |
| ALLEN     | BLAKE     |
| WARD      | BLAKE     |
| JONES     | KING      |
| MARTIN    | BLAKE     |
| BLAKE     | KING      |
| CLARK     | KING      |
| SCOTT     | JONES     |
| TURNER    | BLAKE     |
| ADAMS     | SCOTT     |
| JAMES     | BLAKE     |
| FORD      | JONES     |
| MILLER    | CLARK     |
+-----------+-----------+
13 rows in set (0.00 sec)
```



```
mysql> select * from emp;
+-------+--------+-----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB       | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+-----------+------+------------+---------+---------+--------+
|  7369 | SMITH  | CLERK     | 7902 | 1980-12-17 |  800.00 |    NULL |     20 |
|  7499 | ALLEN  | SALESMAN  | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN  | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER   | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN  | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER   | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER   | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7788 | SCOTT  | ANALYST   | 7566 | 1987-04-19 | 3000.00 |    NULL |     20 |
|  7839 | KING   | PRESIDENT | NULL | 1981-11-17 | 5000.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN  | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
|  7876 | ADAMS  | CLERK     | 7788 | 1987-05-23 | 1100.00 |    NULL |     20 |
|  7900 | JAMES  | CLERK     | 7698 | 1981-12-03 |  950.00 |    NULL |     30 |
|  7902 | FORD   | ANALYST   | 7566 | 1981-12-03 | 3000.00 |    NULL |     20 |
|  7934 | MILLER | CLERK     | 7782 | 1982-01-23 | 1300.00 |    NULL |     10 |
+-------+--------+-----------+------+------------+---------+---------+--------+
14 rows in set (0.00 sec)


mysql> select * from dept;
+--------+------------+----------+
| DEPTNO | DNAME      | LOC      |
+--------+------------+----------+
|     10 | ACCOUNTING | NEW YORK |
|     20 | RESEARCH   | DALLAS   |
|     30 | SALES      | CHICAGO  |
|     40 | OPERATIONS | BOSTON   |
+--------+------------+----------+
4 rows in set (0.00 sec)
```



#### 12.多表连接

```sql
select
    ...
from 
    a
join
    b
on
    a和b的连接条件
join
    c
on
    a和c的连接条件
right join
    d
on
    a和d的连接条件
```

一条SQL语句中内连接和外连接可以混合都可以出现。

找出每个员工的部门名称以及工资等级，要求显示员工名、部门名、薪资、薪资等级？

```bash
mysql> select 
           e.ename,e.sal,d.dname,s.grade 
       from 
           emp e 
       join 
           dept d 
       on 
           e.deptno=d.deptno 
       join 
           salgrade s 
       on 
           e.sal between s.losal and s.hisal;
+--------+---------+------------+-------+
| ename  | sal     | dname      | grade |
+--------+---------+------------+-------+
| SMITH  |  800.00 | RESEARCH   |     1 |
| ALLEN  | 1600.00 | SALES      |     3 |
| WARD   | 1250.00 | SALES      |     2 |
| JONES  | 2975.00 | RESEARCH   |     4 |
| MARTIN | 1250.00 | SALES      |     2 |
| BLAKE  | 2850.00 | SALES      |     4 |
| CLARK  | 2450.00 | ACCOUNTING |     4 |
| SCOTT  | 3000.00 | RESEARCH   |     4 |
| KING   | 5000.00 | ACCOUNTING |     5 |
| TURNER | 1500.00 | SALES      |     3 |
| ADAMS  | 1100.00 | RESEARCH   |     1 |
| JAMES  |  950.00 | SALES      |     1 |
| FORD   | 3000.00 | RESEARCH   |     4 |
| MILLER | 1300.00 | ACCOUNTING |     2 |
+--------+---------+------------+-------+
14 rows in set (0.00 sec)
```



找出每个员工的部门名称，工资等级以及上级领导，要求显示员工名、领导名、部门名、薪资、薪资等级？（加上左外自连接）

```bash
mysql> select e.ename,e.sal,d.dname,s.grade,leader.ename from emp e join dept d on e.deptno=d.deptno join salgrade s on e.sal between s.losal and s.hisal left join emp leader on e.mgr=leader.empno;
+--------+---------+------------+-------+-------+
| ename  | sal     | dname      | grade | ename |
+--------+---------+------------+-------+-------+
| SMITH  |  800.00 | RESEARCH   |     1 | FORD  |
| ALLEN  | 1600.00 | SALES      |     3 | BLAKE |
| WARD   | 1250.00 | SALES      |     2 | BLAKE |
| JONES  | 2975.00 | RESEARCH   |     4 | KING  |
| MARTIN | 1250.00 | SALES      |     2 | BLAKE |
| BLAKE  | 2850.00 | SALES      |     4 | KING  |
| CLARK  | 2450.00 | ACCOUNTING |     4 | KING  |
| SCOTT  | 3000.00 | RESEARCH   |     4 | JONES |
| KING   | 5000.00 | ACCOUNTING |     5 | NULL  |
| TURNER | 1500.00 | SALES      |     3 | BLAKE |
| ADAMS  | 1100.00 | RESEARCH   |     1 | SCOTT |
| JAMES  |  950.00 | SALES      |     1 | BLAKE |
| FORD   | 3000.00 | RESEARCH   |     4 | JONES |
| MILLER | 1300.00 | ACCOUNTING |     2 | CLARK |
+--------+---------+------------+-------+-------+
14 rows in set (0.01 sec)
```



#### 13.子查询

（1）定义

select语句中嵌套select语句，被嵌套的select语句称为子查询。

（2）子查询的语句可以出现在哪里

```sql
select
    ..(select).
from
    ..(select).
where
    ..(select).
```

（3）出现在where后的子查询

找出比最低工资高的员工姓名和工资？

```bash
mysql> select ename,sal from emp where sal > (select min(sal) from emp);
+--------+---------+
| ename  | sal     |
+--------+---------+
| ALLEN  | 1600.00 |
| WARD   | 1250.00 |
| JONES  | 2975.00 |
| MARTIN | 1250.00 |
| BLAKE  | 2850.00 |
| CLARK  | 2450.00 |
| SCOTT  | 3000.00 |
| KING   | 5000.00 |
| TURNER | 1500.00 |
| ADAMS  | 1100.00 |
| JAMES  |  950.00 |
| FORD   | 3000.00 |
| MILLER | 1300.00 |
+--------+---------+
13 rows in set (0.01 sec)
```

（4）from中的子查询

from后面的子查询可以将子查询的查询结果当成一张临时表。

找出每个岗位的平均工资的薪资等级？

第一步：找出每个岗位的平均工资

```bash
mysql> select job,avg(sal) from emp group by job;
+-----------+-------------+
| job       | avg(sal)    |
+-----------+-------------+
| CLERK     | 1037.500000 |
| SALESMAN  | 1400.000000 |
| MANAGER   | 2758.333333 |
| ANALYST   | 3000.000000 |
| PRESIDENT | 5000.000000 |
+-----------+-------------+
5 rows in set (0.01 sec)
```

第二步：把以上的查询结果就当作一张真实的表t，将表t和工资表salgrade s进行表连接

思路如下：

```sql
select
    t.*,s.grade
from 
    t
join
    salgrade s
on
    t.avg(sal) between s.losal and s.hisal;
```

将第一步中的表替换t，下面这种写法会报错，因为代替t表的语句中avg(sal)没有起别名，在on后面的t.avg会报错

```sql
select
    t.*,s.grade
from 
    (select job,avg(sal) from emp group by job) t
join
    salgrade s
on
    t.avg(sal) between s.losal and s.hisal;
```

改为如下：

```sql
select
    t.*,s.grade
from 
    (select job,avg(sal) as avg_sal from emp group by job) t
join
    salgrade s
on
    t.avg_sal between s.losal and s.hisal;
```

```bash
mysql> select
    ->     t.*,s.grade
    -> from 
    ->     (select job,avg(sal) as avg_sal from emp group by job) t
    -> join
    ->     salgrade s
    -> on
    ->     t.avg_sal between s.losal and s.hisal;
+-----------+-------------+-------+
| job       | avg_sal     | grade |
+-----------+-------------+-------+
| CLERK     | 1037.500000 |     1 |
| SALESMAN  | 1400.000000 |     2 |
| MANAGER   | 2758.333333 |     4 |
| ANALYST   | 3000.000000 |     4 |
| PRESIDENT | 5000.000000 |     5 |
+-----------+-------------+-------+
5 rows in set (0.00 sec)
```



（3）select后面出现的子查询



#### 14.union合并查询结果集

查询工作岗位是'MANAGER'和'SALESMAN'的员工？

```bash
mysql> select ename,job from emp where job='manager'
    -> union
    -> select ename,job from emp where job='salesman';
+--------+----------+
| ename  | job      |
+--------+----------+
| JONES  | MANAGER  |
| BLAKE  | MANAGER  |
| CLARK  | MANAGER  |
| ALLEN  | SALESMAN |
| WARD   | SALESMAN |
| MARTIN | SALESMAN |
| TURNER | SALESMAN |
+--------+----------+
7 rows in set (0.07 sec)
```

union的效率高一些，对于表连接来说，每连接一次新表都会满足笛卡尔积。但是union可以减少匹配的次数，在减少匹配次数的情况下，可以完成两个结果集的拼接。

注意：union在进行结果集合并的时候要求两个结果集的列数相同，且合并时列和列的数据类型也要一致（mysql中不需要类型一致，oracle中需要）



#### 15.limit

（1）概述

limit是将查询结果集的一部分取出来，通常使用在分页查询当中（比如百度搜索结果有分页，每页显示10页记录）。

分页的作用是提高用户的体验，因为一次全部都查出来用户体验差，可以一页一页翻着看。

（2）用法

完整用法：limit startIndex, lenghth；从startIndex（起始下标是0）下标开始取length个；

缺省用法：limit k；取top k；

按照薪资降序，取出排名在前五名的员工？

```bash
mysql> select 
           ename,sal 
       from 
           emp 
       order by 
           sal desc 
       limit 5; # 或者limit 0,5;
+-------+---------+
| ename | sal     |
+-------+---------+
| KING  | 5000.00 |
| SCOTT | 3000.00 |
| FORD  | 3000.00 |
| JONES | 2975.00 |
| BLAKE | 2850.00 |
+-------+---------+
5 rows in set (0.00 sec)
```

注意：mysql中limit在order by之后执行

取出工资排名在3-5名的员工？

```bash
mysql> select 
    ->     ename,sal
    -> from
    ->     emp
    -> order by
    ->     sal desc
    -> limit
    ->     2,3;
+-------+---------+
| ename | sal     |
+-------+---------+
| FORD  | 3000.00 |
| JONES | 2975.00 |
| BLAKE | 2850.00 |
+-------+---------+
3 rows in set (0.00 sec)
```



（3）使用limit分页

每页显示pageSize条记录，第pageNo页：limit (pageNo - 1) * pageSize, pageSize;



#### 16.DQL总结

```sql
select
    ...
from
    ...
where
    ...
group by
    ...
having
    ...
order by
    ...
limit
    ...
```

执行顺序：

（1）from

（2）where

（3）group by

（4）having

（5）select

（6）order by

（7）limit



### DDL（数据定义语言）

#### 1.表的创建

（1）建表的语法格式

```sql
create table 表名(字段名1 数据类型,字段名2 数据类型,字段名3 数据类型)；
```

```sql
create table 表名(
    字段名1 数据类型,
    字段名2 数据类型,
    字段名3 数据类型
);
```



（2）mysql中的数据类型

varchar：可变长度的字符串，比较智能，节省空间，会根据实际的数据长度动态分配空间，最长255；

优点：节省空间

缺点：需要动态非配空间，速度慢



char：定长字符串，不管实际的数据长度是多少，分配固定长度的空间去存储数据，使用不恰当的时候可能会导致空间的浪费，最长255。

优点：不需要动态分配空间，速度快

缺点：使用不当可能会导致空间的浪费

varchar和char应该怎么选择？

性别字段：性别是固定长度的字符串，选择char；

姓名字段：每一个人的名字长度不同，选择varchar；



int：数字中的整数型，最长11

bigint：数字中的长整型

float：单精度浮点型数据

double：双精度浮点型数据

date：短日期

datetime：长日期

clob：（character large object）字符大对象，最多可以存储4G的字符串，比如存储一篇文章，存储一个说明，超过255个字符都要采用clob字符大对象来存储

blob：（binary large object）二进制大对象，专门用来存储图片、声音、视频等流媒体数据。往blob类型的字段上插入数据的时候，例如插入图片视频等，需要使用IO流



（3）创建一个学生表

学号、姓名、性别、年龄、邮箱地址

```sql
create table t_student(
    no int,
    name varchar(32),
    gender char(1),
    age int(3),
    email varchar(255)
);
```



删除表：

```sql
drop table 表名;
drop table if exists 表名;
```



### DML（数据操作语言）

#### 1.插入语句insert

```sql
insert into 表名(字段名1, 字段名2, 字段名3...) values(值1, 值2, 值3...);
```

字段名和值要一一对应，包括数量对应和数据类型对应。

```bash
mysql> insert into t_student(no,name,gender,age,email) values(1,'zhangsan','m',20,'zhangsan@123.com');
Query OK, 1 row affected (0.00 sec)

mysql> select * from t_student;
+------+----------+--------+------+------------------+
| no   | name     | gender | age  | email            |
+------+----------+--------+------+------------------+
|    1 | zhangsan | m      |   20 | zhangsan@123.com |
+------+----------+--------+------+------------------+
1 row in set (0.00 sec)
```

```bash
mysql> select * from t_student;
+------+----------+--------+------+------------------+
| no   | name     | gender | age  | email            |
+------+----------+--------+------+------------------+
|    1 | zhangsan | m      |   20 | zhangsan@123.com |
|    3 | NULL     | NULL   | NULL | NULL             |
+------+----------+--------+------+------------------+
2 rows in set (0.00 sec)
```

没有给别的字段赋值，则别的字段都为NULL；insert语句但凡执行成功了，那么必然会多一条记录。

```bash
drop table if exists t_student;
create table t_student(
    no int,
    name varchar(32),
    gender char(1) default 'm',
    age int(3),
    email varchar(255)
);
mysql> desc t_student;
+--------+--------------+------+-----+---------+-------+
| Field  | Type         | Null | Key | Default | Extra |
+--------+--------------+------+-----+---------+-------+
| no     | int          | YES  |     | NULL    |       |
| name   | varchar(32)  | YES  |     | NULL    |       |
| gender | char(1)      | YES  |     | m       |       |
| age    | int          | YES  |     | NULL    |       |
| email  | varchar(255) | YES  |     | NULL    |       |
+--------+--------------+------+-----+---------+-------+
5 rows in set (0.00 sec)
```

使用default方法指定默认值。



insert前面的字段省略的话，等同于所有的字段都写上。

```sql
insert into t_student values(2,'lisi','f',20,'lisi@123.com');
```



#### 2.插入日期

（1）数字格式函数format

```bash
mysql> select ename,format(sal, '$999,999') as sal from emp;
+--------+-------+
| ename  | sal   |
+--------+-------+
| SMITH  | 800   |
| ALLEN  | 1,600 |
| WARD   | 1,250 |
| JONES  | 2,975 |
| MARTIN | 1,250 |
| BLAKE  | 2,850 |
| CLARK  | 2,450 |
| SCOTT  | 3,000 |
| KING   | 5,000 |
| TURNER | 1,500 |
| ADAMS  | 1,100 |
| JAMES  | 950   |
| FORD   | 3,000 |
| MILLER | 1,300 |
+--------+-------+
14 rows in set, 14 warnings (0.00 sec)
```



（2）日期函数

mysql日期格式：

```sql
%Y:年
%m:月
%d:日
%h:时
%i:分
%s:秒
```

str_to_date：将字符串类型varchar类型转换为date类型，通常用于insert语句插入日期类型的数据，通过该函数将字符串转成日期，如果日期的字符串格式是'%Y-%m-%d'，则不需要str_to_date函数。

```bash
create table t_user(
    id int,
    name varchar(10),
    birth date # birth char(10)
);
insert into t_user(id,name,birth) values(1,'zhangsan',str_to_date('01-10-1990','%d-%m-%Y'));
insert into t_user(id,name,birth) values(1,'lisi','1990-10-01');
```



date_format：将date类型转换为具有一定格式的varchar类型，通常用于查询时以某个特定的日期格式展示。

```bash
mysql> select id,name,date_format(birth, '%m/%d/%Y') as birth from t_user;
+------+----------+------------+
| id   | name     | birth      |
+------+----------+------------+
|    1 | zhangsan | 10/01/1990 |
|    1 | lisi     | 10/01/1990 |
+------+----------+------------+
2 rows in set (0.00 sec)

mysql> select id,name,birth from t_user;
+------+----------+------------+
| id   | name     | birth      |
+------+----------+------------+
|    1 | zhangsan | 1990-10-01 |
|    1 | lisi     | 1990-10-01 |
+------+----------+------------+
2 rows in set (0.00 sec)
```

第二条SQL语句实际上是进行了默认的日期格式化，自动将数据库中的date类型转换成varchar类型，并且采用的格式是mysql默认的日期类型：'%Y-%m-%d'。



#### 3.date和datetime两个类型的区别

date是短日期，包括年月日，默认格式是'%Y-%m-%d'

datetime是长日期，包括年月日时分秒，默认格式是'%Y-%m-%d %h:%i:%s'

```bash
drop table if exists t_user;
create table t_user(
    id int,
    name varchar(10),
    birth date, # birth char(10)
    create_time datetime
);
insert into t_user(id,name,birth,create_time) values(1,'lisi','1990-10-01','2020-03-18 15:49:50');
insert into t_user(id,name,birth,create_time) values(2,'zhangsan','1990-10-01',now()); # 使用now()函数获取当前时间

mysql> select * from t_user;
+------+----------+------------+---------------------+
| id   | name     | birth      | create_time         |
+------+----------+------------+---------------------+
|    1 | lisi     | 1990-10-01 | 2020-03-18 15:49:50 |
|    2 | zhangsan | 1990-10-01 | 2022-01-25 21:24:50 |
+------+----------+------------+---------------------+
2 rows in set (0.00 sec)
```



#### 4.修改update

```sql
update 表名 set 字段名1=值1,字段名2=值2,字段名3=值3... where 条件;
```

**注意：没有条件限制会导致所有数据全部更新**

```bash
mysql> update t_user set name='jack',birth='2000-10-11' where id=2;
Query OK, 1 row affected (0.01 sec)
Rows matched: 1  Changed: 1  Warnings: 0

mysql> select * from t_user;
+------+------+------------+---------------------+
| id   | name | birth      | create_time         |
+------+------+------------+---------------------+
|    1 | lisi | 1990-10-01 | 2020-03-18 15:49:50 |
|    2 | jack | 2000-10-11 | 2022-01-25 21:24:50 |
+------+------+------------+---------------------+
2 rows in set (0.00 sec)

mysql> select * from t_user;
+------+------+------------+---------------------+
| id   | name | birth      | create_time         |
+------+------+------------+---------------------+
|    1 | abc  | 1990-10-01 | 2020-03-18 15:49:50 |
|    2 | abc  | 2000-10-11 | 2022-01-25 21:24:50 |
+------+------+------------+---------------------+
2 rows in set (0.00 sec)
```



#### 5.删除数据delete

```sql
delete from 表名 where 条件;
```

**注意：没有条件限制，整张表的数据会全部删除**

```bash
mysql> delete from t_user where id=2;
Query OK, 1 row affected (0.01 sec)

mysql> select * from t_user;
+------+------+------------+---------------------+
| id   | name | birth      | create_time         |
+------+------+------------+---------------------+
|    1 | abc  | 1990-10-01 | 2020-03-18 15:49:50 |
+------+------+------------+---------------------+
1 row in set (0.00 sec)

mysql> insert into t_user(id,name) values(2,'jack');
Query OK, 1 row affected (0.01 sec)

mysql> delete from t_user where name='jack';
Query OK, 1 row affected (0.00 sec)

mysql> select * from t_user;
+------+------+------------+---------------------+
| id   | name | birth      | create_time         |
+------+------+------------+---------------------+
|    1 | abc  | 1990-10-01 | 2020-03-18 15:49:50 |
+------+------+------------+---------------------+
1 row in set (0.00 sec)

mysql> delete from t_user;
Query OK, 1 row affected (0.00 sec)

mysql> select * from t_user;
Empty set (0.00 sec)
```



#### 6.insert插入多条记录

```sql
insert into t_user(字段1，字段2) values(),(),()...;
```

```bash
mysql> insert into t_user(id,name,birth,create_time) values(1,'zs','1980-10-11',now()),(2,'ls','1981-10-11',now());
Query OK, 2 rows affected (0.00 sec)
Records: 2  Duplicates: 0  Warnings: 0

mysql> select * from t_user;
+------+------+------------+---------------------+
| id   | name | birth      | create_time         |
+------+------+------------+---------------------+
|    1 | zs   | 1980-10-11 | 2022-01-27 09:57:08 |
|    2 | ls   | 1981-10-11 | 2022-01-27 09:57:08 |
+------+------+------------+---------------------+
2 rows in set (0.00 sec)
```



#### 7.快速创建表

原理是将查询结果当成一张表新建，可以完成表的快速复制。

```bash
mysql> create table emp2 as select * from emp;
Query OK, 14 rows affected, 2 warnings (0.06 sec)
Records: 14  Duplicates: 0  Warnings: 2

mysql> select * from emp2;
+-------+--------+-----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB       | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+-----------+------+------------+---------+---------+--------+
|  7369 | SMITH  | CLERK     | 7902 | 1980-12-17 |  800.00 |    NULL |     20 |
|  7499 | ALLEN  | SALESMAN  | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN  | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER   | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN  | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER   | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER   | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7788 | SCOTT  | ANALYST   | 7566 | 1987-04-19 | 3000.00 |    NULL |     20 |
|  7839 | KING   | PRESIDENT | NULL | 1981-11-17 | 5000.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN  | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
|  7876 | ADAMS  | CLERK     | 7788 | 1987-05-23 | 1100.00 |    NULL |     20 |
|  7900 | JAMES  | CLERK     | 7698 | 1981-12-03 |  950.00 |    NULL |     30 |
|  7902 | FORD   | ANALYST   | 7566 | 1981-12-03 | 3000.00 |    NULL |     20 |
|  7934 | MILLER | CLERK     | 7782 | 1982-01-23 | 1300.00 |    NULL |     10 |
+-------+--------+-----------+------+------------+---------+---------+--------+
14 rows in set (0.00 sec)
```

```bash
mysql> create table mytable as select empno,ename from emp where job='manager';
Query OK, 3 rows affected (0.02 sec)
Records: 3  Duplicates: 0  Warnings: 0

mysql> select * from mytable;
+-------+-------+
| empno | ename |
+-------+-------+
|  7566 | JONES |
|  7698 | BLAKE |
|  7782 | CLARK |
+-------+-------+
3 rows in set (0.00 sec)
```

省略as相当于将查询结果插入到表中，并不是创建表。



#### 8.快速删除表中的数据

delete删除数据原理：表中数据被删除了，但是这个数据在硬盘上的真实存储空间不会被释放，缺点是删除效率比较低，优点是支持事物回滚，可以恢复数据。属于DML语句。

truncate语句删除数据原理，这种删除效率比较高，表被一次截断，物理删除。缺点是不支持事物回滚，优点是快速。属于DDL语句，删除数据，不删除表。drop是删除整张表。

```bash
mysql> truncate table mytable;
Query OK, 0 rows affected (0.02 sec)

mysql> select * from mytable;
Empty set (0.00 sec)
```

有一张大表，使用delete可能需要很久，效率较低，可以选择使用truncate删除，只需要不到1秒钟删除。



#### 9.对表结构进行增删改

对表结构进行修改：添加一个字段，删除一个字段，修改一个字段（使用alter）。

（1）在实际的开发中，需求一旦被确定，表格就设计好了，很少进行表的结构的修改，成本比较高。这个责任应该由设计人员来承担；

（2）由于修改表结构的操作很少，如果真的需要修改，可以使用工具修改。



#### 10.约束（非常重要）

（1）定义

在创建表的时候，给字段加一些约束，来保证这个表中数据的完整性和有效性。

（2）分类

非空约束：not null

唯一性约束：unique

主键约束：primary key（简称PK）

外键约束：foreign key（简称FK）

检查约束：check（mysql不支持，oracle支持）

（3）非空约束

约束的字段不能为NULL：

```sql
drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255) not null # 非空约束
);
insert into t_vip(id, name) values(1,'zhangsan');
insert into t_vip(id) value(2);
```

把以上语句放到xxx.sql文件当中，直接在mysql中输入source xxx.sql（绝对路径）即可执行该sql脚本。

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/vip_null.sql
Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 1 row affected (0.00 sec)

ERROR 1364 (HY000): Field 'name' doesn't have a default value
mysql> select * from t_vip;
+------+----------+
| id   | name     |
+------+----------+
|    1 | zhangsan |
+------+----------+
1 row in set (0.00 sec)
```



（4）唯一性约束

唯一性约束的字段不能重复，但是可以为NULL，某一个字段被约束不能为NULL，但仍可以都为NULL。

```sql
drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255) unique, # 唯一性约束
    email varchar(255)
);
insert into t_vip(id, name, email) values(1,'zhangsan','zhangsan@123.com');
insert into t_vip(id, name, email) values(2,'lisi','lisi@123.com');

select * from t_vip;

insert into t_vip(id, name, email) values(3,'lisi','lisi@123.com');

insert into t_vip(id) values(4);
insert into t_vip(id) values(5);

select * from t_vip;
```

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/vip_unique.sql
Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 1 row affected (0.01 sec)

Query OK, 1 row affected (0.00 sec)

+------+----------+------------------+
| id   | name     | email            |
+------+----------+------------------+
|    1 | zhangsan | zhangsan@123.com |
|    2 | lisi     | lisi@123.com     |
+------+----------+------------------+
2 rows in set (0.00 sec)

ERROR 1062 (23000): Duplicate entry 'lisi' for key 't_vip.name'
Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

+------+----------+------------------+
| id   | name     | email            |
+------+----------+------------------+
|    1 | zhangsan | zhangsan@123.com |
|    2 | lisi     | lisi@123.com     |
|    4 | NULL     | NULL             |
|    5 | NULL     | NULL             |
+------+----------+------------------+
4 rows in set (0.00 sec)
```

新需求：name和email两个字段联合起来具有唯一性？

```sql
drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255),
    email varchar(255),
    unique(name,email) # name和email两个字段联合起来唯一，表级约束
);
insert into t_vip(id, name, email) values(1,'zhangsan','zhangsan@123.com');
insert into t_vip(id, name, email) values(2,'zhangsan','lisi@456.com');
insert into t_vip(id, name, email) values(3,'zhangsan','lisi@456.com');

select * from t_vip;
```

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/vip_union_unique.sql
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

ERROR 1062 (23000): Duplicate entry 'zhangsan-lisi@456.com' for key 't_vip.name'
+------+----------+------------------+
| id   | name     | email            |
+------+----------+------------------+
|    1 | zhangsan | zhangsan@123.com |
|    2 | zhangsan | lisi@456.com     |
+------+----------+------------------+
2 rows in set (0.00 sec)
```



（5）unique和not null联合

```sql
drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255) not null unique
);

desc t_vip;

insert into t_vip(id, name) values(1,'zhangsan');
insert into t_vip(id, name) values(2,'zhangsan');
insert into t_vip(id) values(2);

select * from t_vip;
```

在mysql中，一个字段加了not null和unique约束之后自动成为主键，oracle不一样。

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/vip_null_unique.sql
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

+-------+--------------+------+-----+---------+-------+
| Field | Type         | Null | Key | Default | Extra |
+-------+--------------+------+-----+---------+-------+
| id    | int          | YES  |     | NULL    |       |
| name  | varchar(255) | NO   | PRI | NULL    |       |
+-------+--------------+------+-----+---------+-------+
2 rows in set (0.00 sec)

Query OK, 1 row affected (0.00 sec)

ERROR 1062 (23000): Duplicate entry 'zhangsan' for key 't_vip.name'
ERROR 1364 (HY000): Field 'name' doesn't have a default value
+------+----------+
| id   | name     |
+------+----------+
|    1 | zhangsan |
+------+----------+
1 row in set (0.00 sec)
```



#### 11.主键约束（非常重要）

（1）定义

主键约束：一种约束

主键字段：该字段上添加了主键约束，这样的字段叫做主键字段

主键值：主键字段中的每一个值叫做主键值

主键是每一行记录的唯一标识，是每一行记录的身份证号。任何一张表都应该有主键，没有主键就会报错。并且一张表的主键约束只能有一个，可以多个字段联合起来添加一个。主键值建议使用int,bigint,char类型，不建议使用varchar做主键，主键一般都是定长的。

（2）主键语句

主键的特征：not null + unique

```sql
drop table if exists t_vip;
create table t_vip(
    id int primary key,
    name varchar(255)
);

-- 可以使用表级约束
-- create table t_vip(
--     id int,
--     name varchar(255),
--     primary key(int)  
-- );

desc t_vip;

insert into t_vip(id, name) values(1,'zhangsan');
insert into t_vip(id, name) values(1,'lisi');

select * from t_vip;

```

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/vip_primary.sql
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.02 sec)

+-------+--------------+------+-----+---------+-------+
| Field | Type         | Null | Key | Default | Extra |
+-------+--------------+------+-----+---------+-------+
| id    | int          | NO   | PRI | NULL    |       |
| name  | varchar(255) | YES  |     | NULL    |       |
+-------+--------------+------+-----+---------+-------+
2 rows in set (0.00 sec)

Query OK, 1 row affected (0.00 sec)

ERROR 1062 (23000): Duplicate entry '1' for key 't_vip.PRIMARY'
+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
+----+----------+
1 row in set (0.00 sec)
```



（3）复合主键

```sql
drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255),
    email varchar(255),
    -- id和name联合起来做主键，复合主键
    primary key(id,name)
);

desc t_vip;

insert into t_vip(id, name, email) values(1,'zhangsan','zhangsan@123.com');
insert into t_vip(id, name, email) values(1,'lisi','lisi@123.com');
insert into t_vip(id, name, email) values(1,'lisi','lisi@123.com');

select * from t_vip;

```

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/vip_union_primary.sql
Query OK, 0 rows affected, 1 warning (0.01 sec)

Query OK, 0 rows affected (0.01 sec)

+-------+--------------+------+-----+---------+-------+
| Field | Type         | Null | Key | Default | Extra |
+-------+--------------+------+-----+---------+-------+
| id    | int          | NO   | PRI | NULL    |       |
| name  | varchar(255) | NO   | PRI | NULL    |       |
| email | varchar(255) | YES  |     | NULL    |       |
+-------+--------------+------+-----+---------+-------+
3 rows in set (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

ERROR 1062 (23000): Duplicate entry '1-lisi' for key 't_vip.PRIMARY'
+----+----------+------------------+
| id | name     | email            |
+----+----------+------------------+
|  1 | lisi     | lisi@123.com     |
|  1 | zhangsan | zhangsan@123.com |
+----+----------+------------------+
2 rows in set (0.00 sec)
```



实际开发中，不建议用复合主键，建议使用单一主键。因为主键存在的意义就是使这行唯一标识。



（4）其它分类

自然主键：主键是一个自然数，和业务没关系

业务主键：主键值和业务紧密关联，例如拿银行卡账号做主键。

在实际开发中，自然主键使用比较多，因为主键只要做到不重复就行，不需要有意义。主键一旦和业务挂钩，当业务发生变动的时候，可能影响到主键值，所以业务主键不建议使用。



（5）自动维护主键

```sql
drop table if exists t_vip;
create table t_vip(
    -- 自动维护主键
    id int primary key auto_increment,
    name varchar(255)
);

desc t_vip;

insert into t_vip(name) values('zhangsan');
insert into t_vip(name) values('zhangsna');
insert into t_vip(name) values('lisi');

select * from t_vip;
```

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/vip_primary_auto.sql
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

+-------+--------------+------+-----+---------+----------------+
| Field | Type         | Null | Key | Default | Extra          |
+-------+--------------+------+-----+---------+----------------+
| id    | int          | NO   | PRI | NULL    | auto_increment |
| name  | varchar(255) | YES  |     | NULL    |                |
+-------+--------------+------+-----+---------+----------------+
2 rows in set (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.01 sec)

Query OK, 1 row affected (0.00 sec)

+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
|  2 | zhangsna |
|  3 | lisi     |
+----+----------+
3 rows in set (0.00 sec)
```





#### 12.外键约束（非常重要）

（1）定义

外键约束：一种约束（foreign）

外键字段：添加外键约束的字段

外键值：外键字段当中的每一个值

使用另一张表字段的值来约束当前表字段，则当前表的字段就是外键字段，字段中的每一个值都是外键值。另一张表被引用的字段不一定是主键，但至少具备唯一性unique。

```sql
-- 先删除子，再删除父;
drop table if exists t_student;
drop table if exists t_class;

-- 先创建父，再创建子;
create table t_class(
    classno int primary key,
    classname varchar(255)
);

-- 引用t_class表中的classno来做当前表外键字段的外键约束;
create table t_student(
    no int primary key auto_increment,
    name varchar(255),
    cno int,
    foreign key(cno) references t_class(classno)
);


-- 先插入父，再插入子;
insert into t_class(classno, classname) values(100, 'A');
insert into t_class(classno, classname) values(101, 'B');

insert into t_student(name, cno) values('jack',100);
insert into t_student(name, cno) values('lucy',100);
insert into t_student(name, cno) values('lilei',100);
insert into t_student(name, cno) values('hanmeimei',100);
insert into t_student(name, cno) values('zhangsan',101);
insert into t_student(name, cno) values('lisi',101);
insert into t_student(name, cno) values('wangwu',101);
insert into t_student(name, cno) values('zhaoliu',101);
insert into t_student(name, cno) values('sunqi',null); -- 外键值可以为null
insert into t_student(name, cno) values('qianba',102); -- 报错，因为外键约束只能是100，101;

select * from t_student;
select * from t_class;
```

```bash
mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/test/student_class_foreign_key.sql
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

Query OK, 1 row affected (0.00 sec)

ERROR 1452 (23000): Cannot add or update a child row: a foreign key constraint fails (`bjpowernode`.`t_student`, CONSTRAINT `t_student_ibfk_1` FOREIGN KEY (`cno`) REFERENCES `t_class` (`classno`))
+----+-----------+------+
| no | name      | cno  |
+----+-----------+------+
|  1 | jack      |  100 |
|  2 | lucy      |  100 |
|  3 | lilei     |  100 |
|  4 | hanmeimei |  100 |
|  5 | zhangsan  |  101 |
|  6 | lisi      |  101 |
|  7 | wangwu    |  101 |
|  8 | zhaoliu   |  101 |
|  9 | sunqi     | NULL |
+----+-----------+------+
9 rows in set (0.00 sec)

+---------+-----------+
| classno | classname |
+---------+-----------+
|     100 | A         |
|     101 | B         |
+---------+-----------+
2 rows in set (0.00 sec)
```





### 三、存储引擎（简单了解）

1.定义

存储引擎是MySQL中特有的一个术语，其他数据库中没有。实际上是一个表存储/组织数据的方式，不同的存储引擎，表存储数据的方式不同。



2.指定存储引擎

```bash
mysql> show create table t_student;
CREATE TABLE `t_student` (
  `no` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) DEFAULT NULL,
  `cno` int DEFAULT NULL,
  PRIMARY KEY (`no`),
  KEY `cno` (`cno`),
  CONSTRAINT `t_student_ibfk_1` FOREIGN KEY (`cno`) REFERENCES `t_class` (`classno`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci |
```

在建表的时候可以在最后小括号的右边使用使用，ENGINE来指定存储引擎，CHARSET来指定这张表的字符编码方式。

mysql默认的存储引擎是InnoDB，默认的字符编码方式是uft8



3.存储引擎类别

mysql支持9大存储引擎：

```bash
mysql> show engines \G
*************************** 1. row ***************************
      Engine: ARCHIVE
     Support: YES
     Comment: Archive storage engine
Transactions: NO
          XA: NO
  Savepoints: NO
*************************** 2. row ***************************
      Engine: BLACKHOLE
     Support: YES
     Comment: /dev/null storage engine (anything you write to it disappears)
Transactions: NO
          XA: NO
  Savepoints: NO
*************************** 3. row ***************************
      Engine: MRG_MYISAM
     Support: YES
     Comment: Collection of identical MyISAM tables
Transactions: NO
          XA: NO
  Savepoints: NO
*************************** 4. row ***************************
      Engine: FEDERATED
     Support: NO
     Comment: Federated MySQL storage engine
Transactions: NULL
          XA: NULL
  Savepoints: NULL
*************************** 5. row ***************************
      Engine: MyISAM
     Support: YES
     Comment: MyISAM storage engine
Transactions: NO
          XA: NO
  Savepoints: NO
*************************** 6. row ***************************
      Engine: PERFORMANCE_SCHEMA
     Support: YES
     Comment: Performance Schema
Transactions: NO
          XA: NO
  Savepoints: NO
*************************** 7. row ***************************
      Engine: InnoDB
     Support: DEFAULT
     Comment: Supports transactions, row-level locking, and foreign keys
Transactions: YES
          XA: YES
  Savepoints: YES
*************************** 8. row ***************************
      Engine: MEMORY
     Support: YES
     Comment: Hash based, stored in memory, useful for temporary tables
Transactions: NO
          XA: NO
  Savepoints: NO
*************************** 9. row ***************************
      Engine: CSV
     Support: YES
     Comment: CSV storage engine
Transactions: NO
          XA: NO
  Savepoints: NO
9 rows in set (0.00 sec)
```



4.MyISAM存储引擎

使用格式文件xxx.frm，数据文件xxx.MYD，索引文件xxx.MYI来存储一张表，优点是可以被转换为压缩、只读表来节省空间，缺点是安全性低。



5.InnoDB存储引擎

InnoDB是mysql默认的存储引擎，是一个重量级的存储引擎。支持事物，数据库崩溃后自动恢复机制，优点是非常安全，缺点是效率低，不支持压缩。



6.MEMORY存储引擎

数据存储在内存当中，断电数据就消失。目的就是快，查询快，不能包含TEXT和BLOB字段。



### 四、事务（非常重要）

#### 1.定义

一个事物就是一个完整的业务逻辑，以转账为例，A账户向B账户转账1w，A账户的钱减去1w，B账户的钱加上1w，两个update语句同时成功，同时失败，不可以再分。



#### 2.应用场景

（1）只有DML语句才会涉及到事物insert，delete，update，只有这三个语句涉及到数据的改动，就一定要考虑安全问题；

（2）假设所有的业务只要一条DML语句就可以完成，就没有存在事物机制的必要了。某件复杂的事情需要多条DML语句联合起来才能完成，所以需要事物的存在；

（3）本质上，一个事物其实就是批量的DML语句同时成功，同时失败。



#### 3.事务机制原理

InnoDB提供一组用来记录事物性活动的日志文件：

```
事务开始：
insert ...
insert ...
delete ...
update ...
事务结束
```

在事物执行过程中，每一条DML的操作都会记录到“事务性活动的日志文件”中，如果提交事务，则会清空事务性活动的日志文件，将数据全部彻底持久化道数据库表中，是一种成功的结束；如果回滚事务，将之前所有的DML操作全部撤销，并清空事务性活动的日志文件，是一种失败的结束。



#### 4.提交和回滚事务

提交事务：commit

回滚事务：rollback

mysql默认情况下是支持自动提交事务的，每执行一次DML语句，则提交一次。回滚永远都是只能回滚到上一次的提交点。比如现在使用insert,delete等语句插入或删除数据，再输入rollback无法回滚，因为再输入这些DML语句的时候已经提交事务了。（这种自动提交实际上不符合我们的开发习惯，因为一个业务通常是需要多条DML语句共同执行才能完成的，为了保证数据的安全，必须要求同时成功之后再提交，所以不能执行一条就提交一条）

关闭mysql的自动提交机制：

```sql
start transaction;
```

表示开启事务，同时关闭自动提交机制，回滚事务：

```bash
mysql> select * from t_vip;
+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
|  2 | zhangsna |
|  3 | lisi     |
+----+----------+
3 rows in set (0.00 sec)

mysql> start transaction;
Query OK, 0 rows affected (0.00 sec)

mysql> insert into t_vip values(5,'wangwu');
Query OK, 1 row affected (0.00 sec)

mysql> select * from t_vip;
+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
|  2 | zhangsna |
|  3 | lisi     |
|  5 | wangwu   |
+----+----------+
4 rows in set (0.00 sec)

mysql> rollback;
Query OK, 0 rows affected (0.00 sec)

mysql> select * from t_vip;
+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
|  2 | zhangsna |
|  3 | lisi     |
+----+----------+
3 rows in set (0.00 sec)
```

提交事务：

```bash
mysql> select * from t_vip;
+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
|  2 | zhangsna |
|  3 | lisi     |
+----+----------+
3 rows in set (0.00 sec)

mysql> insert into t_vip values(5, 'wangwu');
Query OK, 1 row affected (0.00 sec)

mysql> commit;
Query OK, 0 rows affected (0.00 sec)

mysql> select * from t_vip;
+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
|  2 | zhangsna |
|  3 | lisi     |
|  5 | wangwu   |
+----+----------+
4 rows in set (0.00 sec)

mysql> rollback;
Query OK, 0 rows affected (0.00 sec)

mysql> select * from t_vip;
+----+----------+
| id | name     |
+----+----------+
|  1 | zhangsan |
|  2 | zhangsna |
|  3 | lisi     |
|  5 | wangwu   |
+----+----------+
4 rows in set (0.00 sec)
```



#### 5.事务特性

（1）原子性：说明事务是最小的工作单元，不可再分；

（2）一致性：在一个事物中，所有操作必须同时成功，或者同时失败；

（3）隔离性：事物之间具有一定的隔离性；

（4）持久性：事务最终结束的一个保障，事务提交相当于将没有保存到硬盘上的数据保存到硬盘上；



#### 6.事务的隔离性

事务隔离级别：

（1）读未提交：事务A可以读取到事务B未提交的数据，这种隔离级别存在的问题是脏读现象。这种隔离级别一般都是理论上的，大多数数据库都是（2）级别起。

（2）读已提交：事务A可以读取事务B提交之后的数据。这种隔离级别解决了脏读现象。这种隔离级别存在的问题是不可重复读取数据，比如第一次读取的数据是3条，当前事务还没有结束，第二次再读取的时候，读到的数据是4条，称为不可重复读取。

（3）可重复读：事务A开启之后，不管是多久，每一次在事务A中读取到的数据都是一致的。即使事务B将数据已经修改，并且提交了，事务A读取到的数据还是没有发生变化，称为可重复读。解决了不可重复读问题，但存在的问题是出现幻影读，即每一次读到的数据都是幻象，不够真实。mysql中默认隔离级别就是这个。

（4）序列化/串行化：表示事务排队，不能并发，最高隔离级别，效率最低，解决了所有的问题。



#### 7.验证各种隔离级别

查看隔离级别：（8.0版本之后）

```
mysql> select @@transaction_isolation;
+-------------------------+
| @@transaction_isolation |
+-------------------------+
| REPEATABLE-READ         |
+-------------------------+
1 row in set (0.00 sec)
```

见b站视频p110。类似于github中多个用户操作同一个项目的git push，commit和git pull。



### 五、索引（理解内容）

#### 1.定义

索引是在数据库的表的字段上添加的，是为了提高查询效率存在的一种机制，一个字段可以添加一个索引，多个字段联合起来也可以添加索引。

#### 2.索引的实现原理

在任何数据库中主键上都会自动添加索引，id字段上自动有索引，因为id是PK。另外在mysql当中，一个字段上如果有unique约束，也会自动创建索引对象。使用一个自平衡二叉树来存储索引。

#### 3.添加索引的情况

（1）数据量庞大；（多大算庞大，需要具体测试硬件环境）

（2）该字段经常出现where后面，以条件形式存在，也就是说这个字段总是被扫描；

（3）该字段很少出现DML操作，因为DML之后，索引需要重新排序。

#### 4.创建删除索引

```bash
mysql> select * from emp;
+-------+--------+-----------+------+------------+---------+---------+--------+
| EMPNO | ENAME  | JOB       | MGR  | HIREDATE   | SAL     | COMM    | DEPTNO |
+-------+--------+-----------+------+------------+---------+---------+--------+
|  7369 | SMITH  | CLERK     | 7902 | 1980-12-17 |  800.00 |    NULL |     20 |
|  7499 | ALLEN  | SALESMAN  | 7698 | 1981-02-20 | 1600.00 |  300.00 |     30 |
|  7521 | WARD   | SALESMAN  | 7698 | 1981-02-22 | 1250.00 |  500.00 |     30 |
|  7566 | JONES  | MANAGER   | 7839 | 1981-04-02 | 2975.00 |    NULL |     20 |
|  7654 | MARTIN | SALESMAN  | 7698 | 1981-09-28 | 1250.00 | 1400.00 |     30 |
|  7698 | BLAKE  | MANAGER   | 7839 | 1981-05-01 | 2850.00 |    NULL |     30 |
|  7782 | CLARK  | MANAGER   | 7839 | 1981-06-09 | 2450.00 |    NULL |     10 |
|  7788 | SCOTT  | ANALYST   | 7566 | 1987-04-19 | 3000.00 |    NULL |     20 |
|  7839 | KING   | PRESIDENT | NULL | 1981-11-17 | 5000.00 |    NULL |     10 |
|  7844 | TURNER | SALESMAN  | 7698 | 1981-09-08 | 1500.00 |    0.00 |     30 |
|  7876 | ADAMS  | CLERK     | 7788 | 1987-05-23 | 1100.00 |    NULL |     20 |
|  7900 | JAMES  | CLERK     | 7698 | 1981-12-03 |  950.00 |    NULL |     30 |
|  7902 | FORD   | ANALYST   | 7566 | 1981-12-03 | 3000.00 |    NULL |     20 |
|  7934 | MILLER | CLERK     | 7782 | 1982-01-23 | 1300.00 |    NULL |     10 |
+-------+--------+-----------+------+------------+---------+---------+--------+
14 rows in set (0.00 sec)

mysql> create index emp_ename_index on emp(ename);
Query OK, 0 rows affected, 1 warning (0.02 sec)
Records: 0  Duplicates: 0  Warnings: 1

mysql> drop index emp_ename_index on emp;
Query OK, 0 rows affected (0.00 sec)
Records: 0  Duplicates: 0  Warnings: 0
```



查看一条sql语句是否使用了索引进行查询？

```bash
explain select * from emp where ename='king';
+----+-------------+-------+------------+------+----------------+----------------+---------+-------+------+----------+-------+
| id | select_type | table | partitions | type | possible_keys  | key            | key_len | ref   | rows | filtered | Extra |
+----+-------------+-------+------------+------+----------------+----------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | emp   | NULL       | ref  | emp_name_index | emp_name_index | 43      | const |    1 |   100.00 | NULL  |
+----+-------------+-------+------------+------+----------------+----------------+---------+-------+------+----------+-------+
1 row in set, 1 warning (0.00 sec)
```



#### 5.索引失效

（1）在模糊查询中，以%开始

```
mysql> explain select * from emp where ename like '%T';
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | emp   | NULL       | ALL  | NULL          | NULL | NULL    | NULL |   14 |    11.11 | Using where |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
1 row in set, 1 warning (0.00 sec)
```

应当尽量避免模糊查询用%开始，这是一种优化的手段。



（2）使用or

如果使用or，那么要求or两边的条件字段都要有索引才会使用索引，如果其中一个字段没有索引，那么另一个字段上的索引也会失效。

```bash
mysql> explain select * from emp where ename='king' or job='manager';
+----+-------------+-------+------------+------+----------------+------+---------+------+------+----------+-------------+
| id | select_type | table | partitions | type | possible_keys  | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+-------+------------+------+----------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | emp   | NULL       | ALL  | emp_name_index | NULL | NULL    | NULL |   14 |    16.43 | Using where |
+----+-------------+-------+------------+------+----------------+------+---------+------+------+----------+-------------+
1 row in set, 1 warning (0.00 sec)
```

少使用or语句，这也是一种优化手段。



（3）使用复合索引

使用复合索引的时候，没有使用左侧的列查找，索引失效。

创建复合索引：

```bash
mysql> create index emp_job_sal_index on emp(job, sal);
Query OK, 0 rows affected (0.02 sec)
Records: 0  Duplicates: 0  Warnings: 0

mysql> explain select * from emp where job='manager';
+----+-------------+-------+------------+------+-------------------+-------------------+---------+-------+------+----------+-------+
| id | select_type | table | partitions | type | possible_keys     | key               | key_len | ref   | rows | filtered | Extra |
+----+-------------+-------+------------+------+-------------------+-------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | emp   | NULL       | ref  | emp_job_sal_index | emp_job_sal_index | 39      | const |    3 |   100.00 | NULL  |
+----+-------------+-------+------------+------+-------------------+-------------------+---------+-------+------+----------+-------+
1 row in set, 1 warning (0.00 sec)

mysql> explain select * from emp where sal=800;
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | emp   | NULL       | ALL  | NULL          | NULL | NULL    | NULL |   14 |    10.00 | Using where |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
1 row in set, 1 warning (0.00 sec)
```



（4）在where当中索引参加了运算

```bash
mysql> create index emp_sal_index on emp(sal);
Query OK, 0 rows affected (0.01 sec)
Records: 0  Duplicates: 0  Warnings: 0

mysql> explain select * from emp where sal=800;
+----+-------------+-------+------------+------+---------------+---------------+---------+-------+------+----------+-------+
| id | select_type | table | partitions | type | possible_keys | key           | key_len | ref   | rows | filtered | Extra |
+----+-------------+-------+------------+------+---------------+---------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | emp   | NULL       | ref  | emp_sal_index | emp_sal_index | 9       | const |    1 |   100.00 | NULL  |
+----+-------------+-------+------------+------+---------------+---------------+---------+-------+------+----------+-------+
1 row in set, 1 warning (0.00 sec)

mysql> explain select * from emp where sal + 1 = 800;
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | emp   | NULL       | ALL  | NULL          | NULL | NULL    | NULL |   14 |   100.00 | Using where |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
1 row in set, 1 warning (0.00 sec)
```



（5）在where中索引列使用了函数

```bash
mysql> explain select * from emp where lower(ename)='smith';
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | emp   | NULL       | ALL  | NULL          | NULL | NULL    | NULL |   14 |   100.00 | Using where |
+----+-------------+-------+------------+------+---------------+------+---------+------+------+----------+-------------+
1 row in set, 1 warning (0.00 sec)
```



#### 6.索引分类

索引是各种数据库进行优化的重要手段，优化的时候有限考虑的因素就是索引。

索引分类：

（1）单一索引：一个字段上添加索引，一张表中往往有多个字段，所以可以创建多个“单值索引”；

（2）复合索引：多个字段添加索引，如创建(name, age)复合索引，先利用name进行索引查询，当name相同时，我们利用age再进行一次筛选。注意复合索引的字段并不是要都用完，当利用name索引找到结果之后，就不需要再使用age再次进行筛选了；

（3）主键索引：主键上添加索引

（4）唯一性索引：具有unique约束的字段上添加索引





### 六、视图

#### 1.定义

站在不同角度去看同一份数据



#### 2.创建删除视图

```
mysql> create view emp_view as select * from emp;
Query OK, 0 rows affected (0.01 sec)

mysql> drop view emp_view;
Query OK, 0 rows affected (0.00 sec)
```

as后面的只有DQL语句才能以view的形式创建。



#### 3.视图作用

（1）对视图对象的增删改查会导致原表被操作；

（2）假设有一条非常复杂的SQL语句，而这条语句需要在不同的位置上反复使用，每一次使用这个sql语句的时候都需要重新编写，可以把这条复杂的语句以视图对象的形式新建，可以大大简化开发；

（3）视图对象创建之后，可以对视图对象进行增删改查（又称CRUD）。



### 七、DBA命令

#### 1.创建用户

```sql
create user sunzheng identified by '99749544';
```

#### 2.数据导出导入

（1）数据导出

在mysql之外的命令行中将数据导出

```bash
(base) YihedeMBP:~ yihe$ mysqldump bjpowernode>/Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/bjpowernode.sql -u root -p
Enter password: 
```

（2）导出特定的表

```
(base) YihedeMBP:~ yihe$ mysqldump bjpowernode emp>/Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/emp.sql -u root -p
Enter password: 
```

（3）数据导入

在mysql数据库的服务器上，创建数据库，使用数据库之后即可导入。

```bash
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| bjpowernode        |
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
5 rows in set (0.01 sec)

mysql> drop database bjpowernode;
Query OK, 9 rows affected (0.04 sec)

mysql> create database bjpowernode;
Query OK, 1 row affected (0.00 sec)

mysql> use bjpowernode;
Database changed

mysql> source /Users/yihe/Documents/donkey/Alibaba/learning_note/SQL/bjpowernode.sql
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 4 rows affected (0.01 sec)
Records: 4  Duplicates: 0  Warnings: 0

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected, 2 warnings (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 14 rows affected (0.00 sec)
Records: 14  Duplicates: 0  Warnings: 0

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected, 2 warnings (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 14 rows affected (0.00 sec)
Records: 14  Duplicates: 0  Warnings: 0

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 5 rows affected (0.00 sec)
Records: 5  Duplicates: 0  Warnings: 0

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 2 rows affected (0.00 sec)
Records: 2  Duplicates: 0  Warnings: 0

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 9 rows affected (0.01 sec)
Records: 9  Duplicates: 0  Warnings: 0

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 4 rows affected (0.00 sec)
Records: 4  Duplicates: 0  Warnings: 0

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

mysql> show tables;
+-----------------------+
| Tables_in_bjpowernode |
+-----------------------+
| DEPT                  |
| emp                   |
| emp2                  |
| mytable               |
| SALGRADE              |
| t_class               |
| t_student             |
| t_user                |
| t_vip                 |
+-----------------------+
9 rows in set (0.00 sec)
```



### 八、数据库设计三范式

#### 1.定义

数据库表的设计依据。

#### 2.设计三范式

（1）任何一张表必须有主键，每一个字段原子性不可再分；

（2）在第一范式的基础上，要求所有非主键完全依赖主键，不要产生部分依赖；

（3）在第二范式的基础上，要求所有非主键直接依赖主键，不要产生传递依赖。

按照三范式进行，可以避免表中数据的冗余和空间的浪费。

#### 3.范式

（1）第一范式：

（2）第二范式：

多对多如何设计表：多对多，三张表，关系表两个外键！！！

（3）第三范式

一对多如何设计表：一对多，两张表，多的表加外键！！！



#### 4.总结数据库的设计

多对多如何设计表：多对多，三张表，关系表两个外键！！！

一对多如何设计表：一对多，两张表，多的表加外键！！！

一对一如何设计表：一对一，外键唯一！！！

三范式是理论上的，最终的目的都是为了满足客户的需求，有时候拿冗余换速度，在sql中，表与表之间连接的次数越多，效率越低。
