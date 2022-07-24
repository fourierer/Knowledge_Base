drop table if exists t_student;
drop table if exists t_class;

create table t_class(
	classno int primary key,
	classname varchar(255)
);
create table t_student(
	no int primary key auto_increment,
	name varchar(255),
	cno int,
	foreign key(cno) references t_class(classno)
);

insert into t_class(classno, classname) values(100, '北京市大兴区亦庄镇第二中学高三1班');
insert into t_class(classno, classname) values(101, '北京市大兴区亦庄镇第二中学高三1班');

insert into t_student(name,cno) values('jack', 100);
insert into t_student(name,cno) values('lucy', 100);
insert into t_student(name,cno) values('lilei', 100);
insert into t_student(name,cno) values('hanmeimei', 100);
insert into t_student(name,cno) values('zhangsan', 101);
insert into t_student(name,cno) values('lisi', 101);
insert into t_student(name,cno) values('wangwu', 101);
insert into t_student(name,cno) values('zhaoliu', 101);

select * from t_student;
select * from t_class;