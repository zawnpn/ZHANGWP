## RSS？

RSS在Wiki中的定义如下

> RSS（简易信息聚合）是一种消息来源格式规范，用以聚合经常发布更新数据的网站，例如博客文章、新闻、音频或视频的网摘。RSS文件（或称做摘要、网络摘要、或频更新，提供到频道）包含全文或是节录的文字，再加上发布者所订阅之网摘数据和授权的元数据。

一些网站更新频率低，在讲究快速阅读的信息聚合时代，浏览零散的传统网页并不是最优的获取新闻的解决方案。RSS通过聚合信息使我们能更快速全面地获取新闻。

依托网页上的RSS源，可以借助多种多样的RSS阅读器，非常快速地浏览这些聚合后的新闻，更快速地获取信息。

## 介绍

南开数院官网并未提供官方RSS源，但网站结构简单，抓取信息非常方便，这里提供一个能够生成RSS订阅文件及推送邮件的python程序。

程序可定时抓取[南开数院](http://sms.nankai.edu.cn)的文章标题、日期及链接，生成xml文件便于订阅rss，同时推送邮件。

## RSS订阅

 - 学院新闻：[http://sms_rss.oncemath.com/sms-rss/xyxw.xml](http://sms_rss.oncemath.com/sms-rss/xyxw.xml)
 - 本科生教育：[http://sms_rss.oncemath.com/sms-rss/bksjy.xml](http://sms_rss.oncemath.com/sms-rss/bksjy.xml)
 - 研究生教育：[http://sms_rss.oncemath.com/sms-rss/yjsjy.xml](http://sms_rss.oncemath.com/sms-rss/yjsjy.xml)
 - 科研动态：[http://sms_rss.oncemath.com/sms-rss/kydt.xml](http://sms_rss.oncemath.com/sms-rss/kydt.xml)
 - 学生工作：[http://sms_rss.oncemath.com/sms-rss/xsgz.xml](http://sms_rss.oncemath.com/sms-rss/xsgz.xml)
 - 公共数学：[http://sms_rss.oncemath.com/sms-rss/ggsx.xml](http://sms_rss.oncemath.com/sms-rss/ggsx.xml)

## 邮件订阅

如有需要，可以发送相关信息至zwp@oncemath.com进行申请。

## 使用

 - 需要Python3环境，并按需安装模块
 - 更改`config/config.py`中的相关配置信息，同时在`config/receivers`中添加收件人信息(奇数行填写收件人的姓名等相关信息，偶数行填写对应的邮箱地址)
 - 通过命令`python3 main.py`即可运行，可配合crontab进行定时作业

## 注意

 - 使用crontab时，务必确保代码内的路径都是恰当的**绝对路径**，否则可能会报错
 - 在服务器上运行时，注意不要泄露`config/config.py`中的密码等相关信息
 - 注意不要关闭服务器的邮件端口，如22、465等
 - 发送邮件时若要使用SSL，请将`./function/mail.py`的`send`函数中`server = smtplib.SMTP(smtp_server, port)`修改为`server = smtplib.SMTP_SSL(smtp_server, port)`，并注意使用正确的SSL端口号

## GitHub地址

[NKU-SMS-RSS](https://github.com/zawnpn/NKU-SMS-RSS)

## 其他

感谢[@yqnku](https://www.quicy.cn)
