drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255),
    email varchar(255),
    unique(name,email) # name和email两个字段联合起来唯一
);
insert into t_vip(id, name, email) values(1,'zhangsan','zhangsan@123.com');
insert into t_vip(id, name, email) values(2,'zhangsan','lisi@456.com');
insert into t_vip(id, name, email) values(3,'zhangsan','lisi@456.com');

select * from t_vip;
