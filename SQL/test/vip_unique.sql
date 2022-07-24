drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255) unique, # 唯一性约束，列级约束
    email varchar(255)
);
insert into t_vip(id, name, email) values(1,'zhangsan','zhangsan@123.com');
insert into t_vip(id, name, email) values(2,'lisi','lisi@123.com');

select * from t_vip;

insert into t_vip(id, name, email) values(3,'lisi','lisi@123.com');

insert into t_vip(id) values(4);
insert into t_vip(id) values(5);

select * from t_vip;
