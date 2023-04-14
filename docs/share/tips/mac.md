# macOS

## 批量替换

```shell
grep -rl -F "[before]" ./ | xargs perl -pi -e 's/[before]/[after]/g'
```

可使用正则；特殊符号要转义。

## 干净地压缩

压缩时排除掉macOS生成的一些文件。

```shell
alias zipx='zip -x "*.DS_Store" -x "__MACOSX" -r '
zipx <file>.zip <file>
```

## 清除 `__pycache__`

一条命令递归清除目录下所有的 `__pycache__` 及其他缓存文件。

```shell
alias pyclean='find . -name "*.py[c|o]" -o -name __pycache__ -exec rm -rf {} +'
pyclean
```