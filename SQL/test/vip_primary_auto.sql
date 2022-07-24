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
