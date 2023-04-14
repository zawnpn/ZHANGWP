## 背景

一直挺爱玩Steam，之前和很多人一样，喜欢直接在上面购买，后来发现，利用好第三方网站与steam官方的价格差，其实可以享受到更低的折扣。

> ![example](imgs/steam-market-price-bot/example.png)

比如上图里这个饰品，我可以用11.9的第三方市场价买下，又能在Steam市场上以16.13的价格出售。虽然Steam会对市场交易收取大约15%的手续费，但仍能到手14.03左右，这样就相当于11.9/14.03=0.848的折扣为自己充值了一波余额，这样，以后再在Steam上买东西就可以看作是在享受折扣了（G胖微笑着说，是的，你赚了）

当然，上面这个饰品只是随便找的例子，实际上还有比例更为优秀的饰品，实际测试中经常出现比例低到0.6的饰品，然而这种比例不是那么好找的，于是便萌生了爬虫遍历抓取价格来进行比较的想法。

## 简要介绍

爬虫原理很简单，先对第三方网站进行request，拿到数据后先存起来，再针对每个item依次爬取Steam市场上的信息并进行比对，计算市场交易信息的均值、中位数、方差、极差。

同时，爬虫也具备邮件发送的功能，可以在服务端设置好参数，爬虫能够根据用户设置好的条件，将优秀的饰品信息推送给用户，即使出门在外也能随时通过手机收取比例信息。

## 使用

### Usage

    usage: main.py [-h] [--game GAME] [--sleep SLEEP] [--itemnum ITEMNUM]
               [--range RANGE RANGE] [--thresh THRESH] [--counts COUNTS]
               [--save] [--print] [--mail]
### Arguments

    optional arguments:
      -h, --help           show this help message and exit
      --game GAME          Choose a game
      --sleep SLEEP        Sleep time
      --itemnum ITEMNUM    Amount of items
      --range RANGE RANGE  Price range
      --thresh THRESH      Threshold of ratio
      --counts COUNTS      Transactions counts
      --save, -s           Save an output file
      --print, -p          Instant output
      --mail, -m           Send result by mail

### Example

    python3 main.py --game dota -m -s --days 8 --thresh 0.65

## Tips

1. 使用前只需更改`config/config.py`中的相关配置信息.
2. 可以考虑使用`crontab`来进行定时任务.

## Demo

> ![](imgs/steam-market-price-bot/mail.png" width="300" alt="mail.png")

## 项目地址

本项目已开源至GitHub，欢迎探讨交流。

[zawnpn/Steam-Bot_Market](https://github.com/zawnpn/Steam-Bot_Market)
