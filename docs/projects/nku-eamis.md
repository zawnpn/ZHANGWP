---
title: NKU-EAMIS工具
date: 2017-07-01 20:01
tags:
 - python
 - crawler
 - NKU
categories: projects
---

### 简要介绍

一个针对NKU-EAMIS(NKU Education Affairs Management Information System, 南开大学教学管理信息系统)设计的命令行程序。目前具有查询成绩、课表、考试安排以及选课辅助的功能。有喜欢这种命令行风格的同学可以玩玩试试。

> 使用效果(查询课表和考试安排)如下图：
>
> ![nkueamis-demo](/images/projects/nku-eamis/demo.gif)

<!-- more -->

### 安装和使用

程序已经发布到PyPI上了，所以只需pip就可以方便地安装，还无需考虑依赖问题.

在shell中执行命令 `pip install nkueamis` 进行安装.

安装后在终端中直接执行 `nkueamis  参数` 即可使用.

如果不想通过pip安装在系统，可以在[zawnpn/nkueamis](https://github.com/zawnpn/NKU-EAMIS)中下载或者clone到本地，直接执行`python nkueamis.py 参数` 即可.(如果选用这种方式，需确保模块的正常依赖关系，否则会报错)

程序会保持更新，若要更新，运行 `pip install nkueamis --upgrade` 即可(必要时加上`--no-cache-dir`参数避免从本地缓存更新).

### TODO

 - <s>选课辅助</s>(2017/07/08更新: 已经加入选课功能，仅供测试，谨慎使用)

### 使用方法

    nkueamis -g <course_category> [-u <username> -p <password>]
    nkueamis -c [-s <semester>]
    nkueamis -c [-u <username> -p <password>]
    nkueamis -c -s <semester> -u <username> -p <password>
    nkueamis -e [-s <semester>]
    nkueamis -e [-u <username> -p <password>]
    nkueamis -e -s <semester> -u <username> -p <password>
    nkueamis --course-elect

### 参数说明

    course_category      要查询的课程类型(只能是A,B,C,D,E的组合，不区分大小写)
    semester             要查询的学期参数，必须是`[Year]-[Year]:[Semester]`的格式
    username             教务系统的用户名
    password             教务系统的密码

### 选项说明

    -g                   grade query
    -c                   course query
    -e                   exam query
    -s                   semester
    -u                   username
    -p                   password
    --course-elect       elect course
    -h, --help           guidance

### 用法举例

    nkueamis -c    查询课表
    nkueamis -g BCD    查询BCD类的课程成绩(需要按提示输入用户名和密码)
    nkueamis -g ABCDE -u your_username -p your_password    查询ABCDE类的课程成绩
    nkueamis -c -s 2016-2017:2 查询2016-2017学年第2学期的课程表
    nkueamis -e -u your_username -p your_password 查询当前系统默认学期的考试安排
    nkueamis -e -s 2016-2017:2 查询指定学期的考试安排

需要说明的是，查询成绩和课表时，选项`-u <username> -p <password>`不是必须的，一般来说更推荐不带这两项参数进行查询，因为按提示输入密码时密码是不可见的，更安全。而增加这种查询方式是考虑到Linux命令行程序的哲学，能更简洁就能做到的事情，就更简洁地搞定。一条命令查询的话，可以方便复用或者其他程序调用。

*注意：在选课系统中，伯苓班的课程分类只有四类(BC为一类)，在本程序的设计逻辑下，BC类被统一归为了C类，为确保程序逻辑正确，伯苓班的同学在查询成绩时，-g参数后不要带上B字符，否则可能无法成功分类。例如查询BCD成绩时只输入CD即可。*

### 更新

#### 2017/07/08

加入选课辅助功能，仅供测试，请谨慎使用

### 其他

感谢 @谢梓龙 同学在测试和开发上提供的帮助。

iOS用户可以通过Workflow在移动端使用NKU-EAMIS，具体情况请阅读 [「NKU-EAMIS for iOS(Workflow)」](http://www.oncemath.com/eamis-workflow.html)

此外，基于此项目还开发了微信小程序版的教务助手，具体情况见[NKU-EAMIS_MiniApp](https://github.com/zawnpn/NKU-EAMIS_MiniApp)

项目已开源，欢迎交流，互相学习。

Github地址:[zawnpn/nkueamis](https://github.com/zawnpn/NKU-EAMIS)

PyPI地址:[nkueamis](https://pypi.python.org/pypi/nkueamis)
