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
