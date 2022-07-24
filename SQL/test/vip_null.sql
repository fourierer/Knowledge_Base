drop table if exists t_vip;
create table t_vip(
    id int,
    name varchar(255) not null # 非空约束
);
insert into t_vip(id, name) values(1,'zhangsan');
insert into t_vip(id) value(2);