drop table if exists t_vip;
create table t_vip(
    -- 一个字段做主键，单一主键
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
