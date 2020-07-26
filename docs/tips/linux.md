# Linux

## Network

- DNS: `/etc/resolv.conf`
- hosts: `/etc/hosts`
- hostname: `/etc/hostname`

## System

### 查看进程

- `jobs -l | grep <PID>`
- `ps -ef | grep <PID>`
- `ls -la /proc/<PID>`

### 更改权限

```shell
chown <username> -R dir
chgrp <groupname> -R dir
```

### CRLF to LF

```shell
find . -type f -exec dos2unix {} \; # 将整个目录转换，需apt install dos2unix
```

## Server

### 添加用户

```shell
useradd -d /data/<username> -m -s /bin/bash <username>
passwd <username>
# usermod -G sudo <username>
```

### ssh-key

- 需要确保 `/etc/ssh/sshd_config` 中的 `StrictModes` 不为 `yes`
- 本机生成密钥/公钥：`ssh-keygen -t rsa`
- pub key 加入服务器 `~/.ssh/authorized_keys` 中即可（读写权限：`chmod 600 authorized_keys`，`chmod 700 -R ~/.ssh`）
- 也可使用 `ssh-copy-id -i ~/.ssh/id_rsa.pub user@host` ，更为简便

### Problem: ssh需要等待较长时间

可能是 `D-Bus` 与 `systemd`的问题。如果 `dbus` 意外重启了，就需要你去手动重启 `systemd-logind`.

确认一下ssh daemon log (e.g. `/var/log/auth.log`)，如果有下面这个报错，就说明是该问题

```shell
pam_systemd(sshd:session): Failed to create session: Connection timed out
```

此时需要重启 `systemd-logind` ：

```shell
systemctl restart systemd-logind
```