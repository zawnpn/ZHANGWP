# Linux

## Network

### ssh-key

- 需要确保 `/etc/ssh/sshd_config` 中的 `StrictModes` 不为 `yes`
- 本机生成密钥/公钥：`ssh-keygen -t rsa`
- pub key 加入服务器 `~/.ssh/authorized_keys` 中即可（需注意，可能需要确保相应的读写权限：`chmod 644 authorized_keys`，或者 600 权限）
- 也可使用 `ssh-copy-id -i ~/.ssh/id_rsa.pub user@host` ，更为简便

## System

### 非 root 用户安装程序

- make & make install（注意留下 logs 以便删除）
- 考虑使用 [linuxbrew](https://linuxbrew.sh/)

### 查看进程

- jobs -l | grep xxx
- ps -ef | grep xxx
