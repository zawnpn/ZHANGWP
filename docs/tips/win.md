# Windows

## Scoop

Scoop 是 Windows 下的一款好用的包管理软件，软件均安装在 `%USERPROFILE%\scoop` 目录，无需管理员权限，不污染环境。

### 安装

- 打开 Powershell（无需管理员权限）
- 允许本地脚本的执行：`set-executionpolicy remotesigned -scope currentuser`
- 执行：`iex (new-object net.webclient).downloadstring('https://get.scoop.sh')`

### 代理

执行以下命令：

```
scoop config proxy [username:password@]host:port
```

若要取消代理：

- 将代理设置为 `default`（清除代理）或者 `none`（使用系统代理）
- 或者执行命令 `scoop config rm proxy`（使用系统代理）

## Cmder

### 代理

在 `Settings-Startup-Environment` 中填写

```
set http_proxy=http://127.0.0.1:1080
set https_proxy=http://127.0.0.1:1080
```

若要取消代理，在命令前加 `#` 号。

### 关闭烦人的 bell 音效

在 `Settings-Features` 中取消勾选 `Suppress bells(beeps)` 。
