---
title: NKU-EAMIS_MiniApp(南开大学教务助手小程序)
date: 2018-02-01 23:47
tags:
 - javascript
 - MiniApp
 - NKU
categories: projects
---

### 背景

之前用Python写了nku-eamis工具，后来又移植到了服务端，同时根据服务端写了一个workflow脚本。不过在实际使用过程中，总感觉使用体验不是很好，于是利用寒假空闲时间，简单写了一个教务系统的微信小程序。

<!-- more -->

### 介绍

小程序是由两部分构成，一部分是前端展现在用户眼前的微信小程序，另一部分则是放在服务器的后端代码。之所以这样设计，是不得不去满足小程序的审核要求。

微信小程序的审核条件非常严格，首先小程序不能放外部链接，要与外部联系必须进行HTTPS请求，而且限制为只能设置一个通讯域名，同时条件十分严格：

> - 域名只支持 https (request、uploadFile、downloadFile) 和 wss (connectSocket) 协议；
> - 域名不能使用 IP 地址或 localhost
> - 域名必须经过 ICP 备案；
> - 出于安全考虑，api.weixin.qq.com 不能被配置为服务器域名，相关API也不能在小程序内调用。开发者应将 appsecret 保存到后台服务器中，通过服务器使用 appsecret 获取 accesstoken，并调用相关 API。
> - 对于每个接口，分别可以配置最多 20 个域名

此外，对域名的HTTPS证书也是限制重重：

> - HTTPS 证书必须有效。证书必须被系统信任，部署SSL证书的网站域名必须与证书颁发的域名一致，证书必须在有效期内;
> - iOS 不支持自签名证书;
> - iOS 下证书必须满足苹果 App Transport Security (ATS) 的要求;
> - TLS 必须支持 1.2 及以上版本。部分旧 Android 机型还未支持 TLS 1.2，请确保 HTTPS 服务器的 TLS 版本支持1.2及以下版本;
> - 部分 CA 可能不被操作系统信任，请开发者在选择证书时注意小程序和各系统的相关通告。

原本的考虑是，直接在本地小程序抓取教务系统的数据，然后再处理并返回到前端界面，然而仔细阅读小程序的要求后，发现南开大学的网址(nankai.edu.cn)并没有支持HTTPS证书（这里必须得吐槽一下了，现在已经是HTTPS的时代了，为什么还迟迟不支持？），所以要把抓取程序写在客户端是肯定没法过审的。

因此，才考虑分离前后端设计，将数据抓取部分放在了我的服务器上，同时为服务器购买了一个HTTPS书，小程序端简化为简单地作POST请求，服务器响应请求后作为中间端去抓取数据再返回给客户端，小程序拿到数据后只需将这些数据填入渲染好的布局即可。

### Demo

 - 登录页面

> <img width="320" src="/images/projects/eamis-miniapp/login.jpg"/>

 - 课表页面

> <img width="320" src="/images/projects/eamis-miniapp/table.jpg"/>

 - 此外，还有成绩查询、学分绩计算等功能

### 使用

 - 扫描小程序码：

> <img width="240" src="/images/projects/eamis-miniapp/minicode.jpg"/>

 - 或者：直接在微信小程序中搜索“南开教务助手”

### 项目地址

本项目已开源至GitHub，欢迎探讨交流。

小程序端:[zawnpn/NKU-EAMIS_MiniApp](https://github.com/zawnpn/NKU-EAMIS_MiniApp)

服务器端:[zawnpn/NKU-EAMIS_Server](https://github.com/zawnpn/NKU-EAMIS_Server)

