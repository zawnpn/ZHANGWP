# macOS

## 批量替换

```shell
grep -rl -F "[before]" ./ | xargs perl -pi -e 's/[before]/[after]/g'
```

可使用正则；特殊符号要转义。

