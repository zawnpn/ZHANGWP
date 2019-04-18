# Git

## 更新 Fork 下来的 Repo

### 检查远程状态

```bash
git remote -v
```

若未添加 upstream

```bash
git remote add upstream [fork 的原 repo 地址]
```

再次检查 remote 信息，确认是否正确。

### 同步 fork

假设是要同步 master 分支，步骤如下

```bash
git fetch upstream
git checkout master
git merge upstream/master
git push origin master
```

## 本地测试 Pull Request

### 获取 PR 远程内容

```bash
git remote add [PR发送人] [PR的对应 git 地址]
git fetch [PR发送人]
```
### 创建测试分支

```bash
git checkout -b pr_test
```

### Merge

```bash
git merge [PR发送人]/[相应分支]
```

### 删除测试分支

```bash
git branch -D pr_test
```

### 采纳 Pull Request

可以前去 Github 页面合并，也可继续在终端中手动合并：

```bash
git checkout [本地中需要合并的分支]
git merge [PR发送人]/[相应分支]
```

接下来即可 push ，最好 push 前再通过 diff 检查一遍

```bash
git diff origin/[需要合并的分支]
git push
```