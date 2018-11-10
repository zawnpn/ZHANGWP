# Data Processing

## Pandas

### 取行/列

比较简单的一种（似乎性能不太好）：

`df.ix[a:b,m:n]`

### 从文件读表时，指定 index 列

`pd.read_csv('filename', index_col=[0])`

### 将某列变为 index

`df = df.set_index('column_name')`

### 根据 index/column 的名称做筛选

[Doc](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.filter.html)

```python
>>> df
one  two  three
mouse     1    2      3
rabbit    4    5      6
```

```python
>>> # select columns by name
>>> df.filter(items=['one', 'three'])
one  three
mouse     1      3
rabbit    4      6
```

```python
>>> # select columns by regular expression
>>> df.filter(regex='e$', axis=1)
one  three
mouse     1      3
rabbit    4      6
```

```python
>>> # select rows containing 'bbi'
>>> df.filter(like='bbi', axis=0)
one  two  three
rabbit    4    5      6
```

### json_normalize 将 json 变为表格

[Doc](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.io.json.json_normalize.html)

```python
>>> from pandas.io.json import json_normalize
>>> data = [{'id': 1, 'name': {'first': 'Coleen', 'last': 'Volk'}},
...         {'name': {'given': 'Mose', 'family': 'Regner'}},
...         {'id': 2, 'name': 'Faye Raker'}]
>>> json_normalize(data)
    id        name name.family name.first name.given name.last
0  1.0         NaN         NaN     Coleen        NaN      Volk
1  NaN         NaN      Regner        NaN       Mose       NaN
2  2.0  Faye Raker         NaN        NaN        NaN       NaN
```