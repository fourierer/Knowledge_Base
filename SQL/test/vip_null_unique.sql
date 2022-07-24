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
