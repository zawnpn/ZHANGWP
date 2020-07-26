# Python

## Tools

### Jupyter 中添加 env

 - 在主 env 下安装 ipykernel `conda install ipykernel`
 - 想要添加的 env 下安装 ipykernel `conda install -n 环境名称 ipykernel` (若是新建 env ，则 `conda create -n 环境名称 python=3.5 ipykernel`)
 - 进入想添加的 env `conda activate 环境名称`
 - 将 env 写入 kernel `python -m ipykernel install --user --name 环境名称 --display-name "在jupyter中显示的环境名称"`
 - 运行 `jupyter notebook` 即可

### Jupyter 服务器端配置

生成密码：

```python
from IPython.lib import passwd
passwd()
```

配置 `~/.jupyter/jupyter_notebook_config.py`：

```python
c.NotebookApp.password='sha1:xxxxxx'
c.NotebookApp.ip = '0.0.0.0' #所有绑定服务器的IP都能访问，若想只在特定ip访问，输入ip地址即可
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.notebook_dir = '<default_path>' #设置 Jupyter 根目录
```

### 安装 Matlab Engine for Python

```shell
cd /matlab_root/extern/engine/python
python setup.py install
## if not root user, run following ##
# python setup.py build -b <somedir> install
```