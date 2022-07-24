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
