圈子文章 by 根生仔
### 装逼犯与树莓派续

目录  

装逼与渗透  
装逼与内核FUZZ  
装逼与智能语音

><font color = "yellow">注 本文适宜读者=渴望装逼者∩成功装逼者∩若无其事装逼者={曾经渴望若无其事装逼且成功的汉子}={圈子所有大佬们}</font>
>注 本文应当使用BGM https://music.163.com/#/song?id=472366887 歌单 https://music.163.com/#/playlist?id=642132193
>注 本文伴手读物 <font color = "gree">《绿帽子讲婚姻安全》</font>

---
---

#### 第三章 装逼与渗透

完成之前的部分后，找到LEADER求表扬  

我：您看看，已经可以执行命令了嘿。
LEADER:还行，如果我想渗透呢，一条一条敲命令？
我：无非是再加功能的事，您且捎待会儿
LEADER:小伙子不错，果然长得丑就写代码且还行。
我：（内心开始构造一个我和他媳妇女儿的激情故事但是表面还得笑嘻嘻）好的。

反过头来回溯需求，既然是渗透，那没有网的情况是不可行的（废话），那就需要一个4g路由做热点，一个树莓派就很吃力了，
而且没有vps的话也不安全，更加不合适操作了。

所以我们要梳理两个部分，一是渗透流程化问题 二是分别由什么部分完成。
代码其实是很简单的东西，毕竟我们现在研究开发的是demo，正在上线成为一个正式的产品是有难度的，
但是思路是最重要的，无论是渗透和做什么其他事情，理清了思路就会事半功倍。

所需|说明
:---------:|:----------:
华为4g无线路由B315|支持华为各位有什么意见么
vps|酌情选用
骚思路|各位猥琐且骚的玩意儿
u盘/更大的sd卡|我的16g卡不够用了嘿 看需求咯



梳理下整个渗透测试流程
`//貌似圈子的markdown不支持流程图，就用xmind pro来画画 by圈子小和尚Mannix--伟大的分享者 `
我上传到github吧，方便点 `https://github.com/lazybootsafe/MindMapbyself/`

一般渗透测试流程

* 明确目标
* 分析风险，获得授权
* 信息收集
* 漏洞探测（手动&自动）
* 漏洞验证
* 信息分析
* 利用漏洞，获取数据
* 信息整理
* 形成报告

但是在实战中一般不需要完全遵循这流程，我们简单划分两种情况（可能不对大佬请指教），在树莓派`/home/pi`new一个

```bash
mkdir pentest
cd pentest
mkdir WhiteHat
mkdir BlackHat
```
很明显就按照黑白帽来划分流程
- whitehat流程
   - 确定授权目标
   - 通用漏洞探测
   - 手工漏洞挖掘
   - 漏洞验证分析
   - 整理与报告
   - ...

- blackhat流程
   - 确定目标
   - 拓展目标
   - 目标信息收集
   - 根据收集情况来决定 社会工程学/渗透测试
   - ...

<font color ="red">回到情景中，团长是一个伟大的web安全工程师（白帽子），一日LEADER给到团长一个白帽项目，一个大学所有网站外部web安全测试。</font>  
在此情景中，会先确定如123.edu.cn，拓展成*.123.edu.cn（子域名） 然后对这些目标们简单做cms判断，做poc批量扫描，几千个poc先打打，说不定呢就有漏洞，不行再手工。在此前期工作就是子域名爆破-->whatcms-->poc批量打，手工部分是代替不了的操作，且不表。


<font color ="red">去到另一个情景中，团长是一个帅气的渗透测试工程师（黑帽子），一日LEADER给到团长一个JZ项目，某独组织，需求是人员监控。  </font>
在此情景中，这个信息收集的过程比较复杂，除了子域名外，在ip上的判断，比如端口，c段等，在人员判断上等等，运用很多额外的工具，比如shodan等确定目标关联性，最好还对人员进行分析，邮箱分析等等，比如目标为`vot.org`某之声，我们可以通过cms判断出是wordpress4.9.x ，poc打完后没结果，那可以从三个地方进行信息收集，人员，邮箱，ip。人员指的是管理者，邮箱是域名邮箱，ip可以whois，最好加入长期监控列表中，来进行长期跟踪，比如下次wordpress出新的exp或者自己审计出0day的话，可以快速实现。

**监控信息.txt**
```
目标:某独组织
分目标：
1. vot.org
2. xxx.org
3. ....org
子域名：
1. a.vot.org
2. b.vot.org
3. ***.vot.org ...

邮箱：
info@vot.org
editor@vot.org
officemanager@vot.org
norsangn@vot.org
tpeldon@vot.org
mail@vot.org  

...
... ...
... ... ...

```  
>“渗透测试的本质是信息收集” ---候某大佬如是说

一位大佬在defcon上说。一直觉得很对，信息收集才是一切的出发点，
>无论<font color = "black">黑</font>帽<font color = "white">白</font>帽，带上<font color = "gree">绿</font>帽就是好黑客。 --某F菜鸡如是说

我在圈子说的。。然后就被打了...

总结上述，我们能够使用机器代劳的有 **信息收集监控** **漏洞批量** ，将黑白帽当做分组标签。

- 信息收集
   - 子域名爆破
   - c段信息收集
   - 端口
   - CMS判断

- 漏洞批量payload
   - 死命丢payload来打
   - pocscan改版
   - whatcms改版

- 目标监控
   - 有啥可说的，监控呗。
   - CVE监控
   - 目标站点监控

**怎么实现**

选用一些能够使用且能够完成我们的目标的开源或者免费使用的工具(本地资源/工具)，
毕竟使用树莓派的不是傻逼就是穷逼，再加上一些在戏资源，我觉得还是ok的。

- 1 扫描工具 nmap+masscan 这是首选，但是怎么搭配，能够达到我们为最佳的扫描效果呢，为父已经想好了，且看着。
- 2 子域名爆破 修改版subdomain.py 放心代码都会给上。
- 3 c段暂时没想好
- 4 cms和payload批量 修改版xx.py 有bugscan 插件和 pocsuit等插件。加上一个whatcms插件。
- 5 监控 一个cve监控+某修改版工具源码，放心 源码均给出。

统一声明喂给扫描工具的都是域名domain.txt
结果统一展示为result.txt

上一文我们将了subproccess是没办法做到占用进程的，那我们结果给到某目录的result.txt 到时候，使用cat 命令展示，或者我再想想，但是我前端能力≈屎....

1 ** 扫描工具**

首先是nmap+masscan，我们也是写一个python3程序，我们看下流程图，然后我会给出代码，直接单独用即可，也可以结合使用。如有版权信息，自行删除换成自己的即可。
图

2 ** 子域名爆破**
然后是子域名工具，我也会提供代码，我改进的部分是
- **DNS服务器增加**
- **字典数量增加**
- **线程数 稳定性**
- **中文化**

版权是原作者，随意改不干我的事吧，我也改成自己的了，毕竟我们课题是啥...//装逼犯XX的。

3 ** c段工具**

<font color="red">我们研究下这个</font>
c段尽管有很多工具，但是没有好的拓展办法目前，还是手工判断为准，比如说某国/内政部，ip是10.10.10.x 拓展到10.10.x.x ，但是10.10.9.x 和10.10.11.x； 9的段是比如某国/外交部，这就是相关性的站点，可拓展；11的段是某国某些企业，那就不具有相关性，那就不应该拓展，如果是这样的，还得做c段的反查域名，这是人工手动的办法，再这样想下去流程应该是

```
域名查ip拓展成c段范围
         |
         ↓
c段反查域名做列表
         |
         ↓
提取域名站点title/content等特征
         |
         ↓
与初始域名的keywords做相关性/相似性分析
    |         |
    |         |
    是        否
  拓展加入     关闭

```
感觉是这个思路，但是一直没找到好的办法写，有两个难点：
第一是怎么做反查，api里面的话 shodan/微步 这块都不错（问题是收费），就阻止我继续写代码了。

第二是使用相关性或者相似性分析，我也讲下自己的思路，哪位大佬觉得说的还像是个人话，可以交流下，也可以帮助圈子的孩子健康快乐成长。我这解释和分析下两种办法的判断。

##### 第一种** 使用相关性系数来判断**
我们先定义两组数据，下列不知道markdown怎么写公式，我就word写截图吧。
```
设下列集合为 人取特征--集合

A{1,2,3}
B{4,5,6}

  a是原域名特征
  b是拓展域名特征
```
那我们需要定义的数据的期望
连续随机变量
离散随机变量
图

再定义数据的离散度 方差标准差
也是对于连续/离散随机变量
图

代码实现
```python
# 计算每一项数据与均值的差
def de_mean(x):
  x_bar = mean(x)
  return [x_i - x_bar for x_i in x]
# 辅助计算函数 dot product 、sum_of_squares
def dot(v, w):
  return sum(v_i * w_i for v_i, w_i in zip(v, w))
def sum_of_squares(v):
  return dot(v, v)
# 方差
def variance(x):
  n = len(x)
  deviations = de_mean(x)
  return sum_of_squares(deviations) / (n - 1)
# 标准差
import math
def standard_deviation(x):
  return math.sqrt(variance(x))

```
再写一个将两个变量打印出来即可

定义好了基本概念，协方差与相关系数
接下来，要计算两组数据的相关性。
一般采用相关系数来描述两组数据的相关性，而相关系数则是由协方差除以两个变量的标准差而得，
相关系数的取值会在 [-1, 1] 之间，-1 表示完全负相关，1 表示完全相关，中间值...
再定义两个公式
图

代码实现
```python
# 协方差
def covariance(x, y):
  n = len(x)
  return dot(de_mean(x), de_mean(y)) / (n -1)
# 相关系数
def correlation(x, y):
  stdev_x = standard_deviation(x)
  stdev_y = standard_deviation(y)
  if stdev_x > 0 and stdev_y > 0:
    return covariance(x, y) / stdev_x / stdev_y
  else:
    return 0
```

上述是用python代码实现的，其实也能用numpy封装好的函数，或者使用 pandas库计算协方差、相关系数

然后用相关性系数来判断这个ip对应的域名是否加入新target中。但是怎么取值，怎么具体实现，我也没写出来。


##### 第二种** 使用相似性分析来判断**

这种办法我想的是文本相似性分析，不使用人工手动赋值特征词，首先是使用同一种爬虫对两个站点的index或者首页进行爬取，我是觉得应该转成转成文字集合I{abc-xyz} 网站全文定义

```
设下列集合为 机取特征--集合

A{1,2,3}
B{4,5,6}

  a是原域名特征
  b是拓展域名特征
  比如说 123 456 这是我写进集合里面的，就是手工赋值的，比如A{中华民国内政部,蔡英文,基础建设}，这就是渗透人员对站点的一种定义就算是人取特征
  由爬虫和代码提取出来的，那就是机取特征。
```
** 特征怎么取**
定义几个概念
TF值 ：词频
IDF值 ：逆文档频率
TF-IDF值 ：二次权重
corpus 余弦相似值 向量.

对于提取一篇文章的关键特征，如果某个词很重要能作为一种特征（keywords）它应该在这篇文章中多次出现。所以需要进行”词频”（Term Frequency，缩写为TF）统计。

现次数最多的词是—-“的”、”是”、”在”—-这一类最常用的词。它们叫做”停用词”（stop words），表示对找到结果毫无帮助、必须过滤掉的词，所以要先行对全文进行处理。

而对于我们需要的关键特征，例如对于我之前分析过的伟大文学家莫言的作品--《丰乳肥臀》，出现最多的前十个词中包括如：上官，女人，一个，地说，母亲。显然对于关键特征：一个，地说，对反应文章的特性并没有其余几个好，这时，就是在词频的基础上，要对每个词分配一个”重要性”权重。最常见的词（”的”、”是”、”在”）给予最小的权重，较常见的词（”一个”）给予较小的权重，较少见的词（”上官”、”女人”）给予较大的权重。也就是对一次提取的特征keywords中再再进行二次权重划分。

这个权重叫做”逆文档频率”（Inverse Document Frequency，缩写为IDF），它的大小与一个词的常见程度成反比。知道了”词频”（TF）和”逆文档频率”（IDF）以后，将这两个值相乘，就得到了一个词的TF-IDF值。某个词对文章的重要性越高，它的TF-IDF值就越大。

corpus 语料库 ⒈语料库中存放的是在语言的实际使用中真实出现过的语言材料，因此例句库通常不应算作语料库；⒉语料库是承载语言知识的基础资源，但并不等于语言知识；⒊真实语料需要经过加工（分析和处理），才能成为有用的资源。
> https://baike.baidu.com/item/%E8%AF%AD%E6%96%99%E5%BA%93/11029908?fr=aladdin

为了省事自行看神奇又错误的百度百科

cosine similiarity 余弦相似值 对于要要计算相似度的两个句子，步骤：分词-计算词频-将词频写成向量形式-计算向量相似程度（向量夹角),这玩意要记得看余弦定理就知道咋求，我就不多说了 设一个a向量 b向量 夹角为θ 读音//'θi:tə//
>https://en.wikipedia.org/wiki/Cosine_similarity

两条线段之间形成一个夹角，如果夹角为0度，意味着方向相同、线段重合；如果夹角为90度，意味着形成直角，方向完全不相似；如果夹角为180度，意味着方向正好相反。因此，我们可以通过夹角的大小，来判断向量的相似程度。夹角越小，就代表越相似。余弦值越接近1，就表明夹角越接近0度，也就是两个向量越相似

图

** 计算方法**
（1）使用TF-IDF算法，找出的关键词；

（2）每个域名对应关键词集合中各取出若干个关键词（比如20个），合并成一个集合，计算每个对于这个集合中的词的词频（为了避免长度的差异，可以使用相对词频）；

（3）生成两篇文章各自的词频向量；

（4）计算两个向量的余弦相似度，值越大就表示越相似。

综上所述，我们使用一个python库完成，那就是Gensim，下面是介绍:

>Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。它支持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法，支持流式训练，并提供了诸如相似度计算，信息检索等一些常用任务的API接口。
gensim 以“文集”——文本文档的集合——作为输入，并生成一个“向量”来表征该文集的文本内容，从而实现语义挖掘。

这里也合适我们的情景，我设想下代码:
cduan-gensim.py
```python
#！／usr/bin/env python
#-*- coding:utf-8 -*-
#author:finger
#简单说下 jieba是中文分词一个玩意，玩几次就熟悉了
import jieba
from gensim import corpora,models,similarities
from collections import defaultdict
doc1 = '/home/pi/zhuangbi/pentest/blackhat/cduan/keyword1.txt'
doc2 = '/home/pi/zhuangbi/pentest/blackhat/cduan/keyword2.txt'
d1 = open(doc1).read()
d2 = open(doc2).read()
data1 = jieba.cut(d1)
data2 = jieba.cut(d2)
list1 = []
list2 = []
list = []
for i in data1:
    list1.append(i)
for i in data2:
    list2.append(i)
list = [list1,list2]
frequency = defaultdict(int) #如果键不存在则返回N/A,而不是报错,获取分词后词的个数 防止中途出错
for i in list:
    for j in i:
        frequency[j] +=1
#创建关键词 --集合-- 词典
dictionary = corpora.Dictionary(list)
#词典 关键词集合 保存到本地
dictionary.save('/home/pi/zhuangbi/pentest/blackhat/cduan/keyword1.txt')
doc3 = '/home/pi/zhuangbi/pentest/blackhat/cduan/doc3.txt'
d3 = open(doc3).read()
data3 = jieba.cut(d3)
data31 = []
for i in data3:
    data31.append(i)
new_doc = data31
#稀疏向量.dictionary.doc2bow(doc)是把文档doc变成一个稀疏向量，[(0, 1), (1, 1)]，表明id为0,1的词汇出现了1次，至于其他词汇，没有出现。
new_vec = dictionary.doc2bow(new_doc)
#获取corpus 语料库
corpus = [dictionary.doc2bow(i) for i in list]
tfidf = models.TfidfModel(corpus)
#特征数 NUM
featureNUM = len(dictionary.token2id.keys())
#通过TfIdf对整个语料库进行转换并将其编入索引，以准备相似性查询
index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featureNUM)
#计算向量相似度
sim = index[tfidf[new_vec]]
print(sim)

```

图

本来不想贴代码，因为很多代码没有完善，还在测试中，但是截图中cat代码文件，不支持中文。。。

附录 gensim教程
> https://radimrehurek.com/gensim/tutorial.html

4 ** 批量**

改版于pocscan +whatcms代码 之后会贴出代码，就是改成终端api模式，代码所属权 原作者。

还没完成 等待上传


5 ** 监控**

参考https://xz.aliyun.com/t/1694/

CVE监控 我的设计方案
实现：cve监控 监控漏洞，爬取、分析、存储并第一时间通知到我。
* 来源
  * CVE(Common Vulnerabilities and Exposures)

     * https://nvd.nist.gov/vuln/data-feeds
  * exploit-db

     * https://www.exploit-db.com
  * other
    * https://cwiki.apache.org/confluence/display/WW/Security+Bulletins
    * https://pivotal.io/security/

* 漏洞处理
    * 爬取
    * 分析
    * 存储

* 通知
    * 邮件通知
    * 微信订阅

代码主要是BeautifulSoup解析网页和mongodb数据库的使用，smtplib用于现在的邮件通知，
微信这块暂时没写进去，我想的是之前用的itchat来完成，目前还没测试代码，所以先贴出jk的代码，之后代码测试完成我会给出地址。

/home/pi/zhuangbi/pentest/whitehat/cvejiankong/cvejk.py如下：

```python
#! /usr/bin/env python
#-*- coding:utf-8 -*-


import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import *
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import sys


reload(sys)
sys.setdefaultencoding('utf8')

class CVEInfo:
    def __init__(self,url, cveid, keyword, description, company, createdate):
        self.url = url
        self.cveid = cveid
        self.keyword = keyword
        self.description = description
        self.company = company
        self.createdate = createdate

    def show(self):
        return '<p><b>漏洞编号：</b><a href="'+self.url+'">'+self.cveid+'</a></p><b>相关厂商：</b>'\
            +self.company +'<br><b>披露日期：</b>'\
            +self.createdate+'<br><b>关键字：</b><font size="3" color="red">'\
            +self.keyword+'</font><br><b>漏洞描述：</b>'\
            +self.description + '<br><br><hr/>'

    def add(self):
        data = {
            'cveid': self.cveid,
            'keyword': self.keyword,
            'description': self.description,
            'company': self.company,
            'createdate': datetime.strptime(self.createdate, "%Y%m%d"),
            'addDate': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),

        }
        return data
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

def getMiddleStr(content, startStr, endStr): # 获取文本中间内容
    startIndex = content.index(startStr)
    if startIndex >= 0:
        startIndex += len(startStr)
        endIndex = content.index(endStr)
    return content[startIndex:endIndex]


def getCVES():  # 获取最新到CVE列表
    urls = []
    try:
        url = 'https://cassandra.cerias.purdue.edu/CVE_changes/today.html'
        res = requests.get(url, headers=headers, timeout=60)
        CVEList_html = getMiddleStr(res.text, 'New entries:', 'Graduations')
        soup = BeautifulSoup(CVEList_html, 'html.parser')
        for a in soup.find_all('a'):
            urls.append(a['href'])
        return urls
    except Exception as e:
        print(e)


def getCVEDetail(url):  # 获取CVE详情
    keywords = ['WordPress','Struts','Jboss','Remote Code Execution Vulnerability'] #关注的关键字
    try:
        res = requests.get(url, headers=headers, timeout=60)
        soup = BeautifulSoup(res.text, 'html.parser')
        cveId = soup.find(nowrap='nowrap').find('h2').string
        table = soup.find(id='GeneratedTable').find('table')
        description = table.find_all('tr')[3].find('td').string
        keyword = None
        for k in keywords:
            if k in description:
                keyword = k
                break
        company = table.find_all('tr')[8].find('td').string
        createdate = table.find_all('tr')[10].find('td').string
        cveInfo = CVEInfo(url, cveId, keyword, description, company, createdate)
        if keyword is None:
            return None
        else:
            return cveInfo

    except Exception as e:
        print(e)

def addData(data):#安装mongodb 新建一个数据库mycvedb 用户名密码自己设置就行
    DBNAME = 'mycvedb'
    DBUSERNAME = 'test'
    DBPASSWORD = 'secquan'
    DB = '127.0.0.1'
    PORT = 65521
    db_conn = MongoClient(DB, PORT)
    na_db = getattr(db_conn, DBNAME)
    na_db.authenticate(DBUSERNAME, DBPASSWORD)
    c = na_db.cvedatas
    c.update({"cveid": data['cveid']}, {'$set': data}, True)

def sendEmail(mail_msg):  # 发送邮件
    sender = '发件人@163.com' # 发件人
    password = 'password' # 发件人密码
    receiver = '收件人r@163.com' # 收件人
    message = MIMEText(mail_msg, 'html', 'utf-8') #以html发送
    message['From'] = sender
    message['To'] = receiver
    subject = '最新CVE列表'
    message['Subject'] = Header(subject, 'utf-8')
    try:
        smtpObj = smtplib.SMTP('smtp.163.com')
        smtpObj.login(sender, password)
        smtpObj.sendmail(sender, receiver, message.as_string())
        print('邮件发送成功')#终端打印信息
    except smtplib.SMTPException:
        print('Error: 无法发送邮件')
def main():
    nowTime = '当前时间：<font size="3" color="red">' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '</font><br>'
    urls = getCVES()
    msg = ''
    if(len(urls)==0):#这其实只是个异常，怎么会没有cve 每天都有，响应尼玛呢
        msg = nowTime + '<p>无cve</p>'
    else:
        msg_header = '<p>今日CVE一共<font size="3" color="red">' + str(len(urls))+'</font>个。'
        i = 0
        for url in urls:
            cveInfo = getCVEDetail(url)
            if cveInfo is not None:
                i = i + 1
                data = cveInfo.add()
                addData(data)
                msg = msg + cveInfo.show()
        if i == 0:
            msg = nowTime + msg_header +  '根据设置的关键字，未匹配到关注的CVE信息。</p>'
        else:
            msg_key_header = '</p>根据设置的关键字，关注的CVE信息一共<font size="3" color="red">' + str(i)+'</font>个。具体如下：<br><br>'#内容可随意自己改
            msg = nowTime + msg_header + msg_key_header + msg
    sendEmail(msg)
if __name__ == '__main__':
    main()
```


邮件效果图

下一步的监控是目标站点的监控，这里用的是专门优化的一个工具完成。

改版比较大，而且涉及到版权问题了（在gmail商量过人家不允许汉化和改版），
待我删尽了版权再上传代码压缩包，仅限内部使用，不然XX了就搞笑了。
先行截图吧 cache是监控缓存，db是监控内容，我这还有twXX部门的监控资料呢，我就不删了嘿嘿~~


### 实践与情景

** 实践**

回到上次我们的remote_command.py中，我们分类下命令

命令|功能|对应文件|参数示例|结果文件|其他|
:-----:|:-----:|:-----|:---:|:---:|:---|
pentest|渗透帮助菜单|remote_command.py|无|无|complete
nm|nmap+masscan扫描ip端口|startnm.py|nm '192.168.0.1'|result1.txt|complete
sd|修改版subdomain爆破子域名|subdomain.py|sd 'evilxyz.xyz'|result2.txt|complete
tz|c段拓展目标范围|cduantuozhan.py|tz '192.168.1.1'|newtarget.txt|thinking
sb|识别cms类型|whatcms.py|sb 'result1.txt'/sb 'evilxyz.xyz'|resultcms.txt|coding
pl|批量打payload|pl.py|pl 'result1.txt'|resultshell.txt|coding
jk1|现在执行cve今日更新情况邮箱回执|cvejk.py|jk1|微信端显示“邮件发送成功”|complete
jk2|加入sf网站监控|sf.py|jk2 'evilxyz.xyz'|无，需要进web页面查看|conding
ct|查看各种结果文件|bash cat ** *.txt ** | 需要什么result 就ct什么|无|coding

相应的remote_command.py 修改后放入pentest目录：

```python
#!/usr/bin/env python
#coding=utf-8
#根据需要删除增加代码，此处只是基本代码，但是也能完成功能

import os
import sys
import itchat
import time

if os.name == 'posix' and sys.version_info[0] < 3:
    import subprocess32 as subprocess
else:
    import subprocess

pentesthelp = '''\
WeChat remote:
* this is helpful menu to update for pentest
* in midnight add pre-pentest'{value for value in variable} is bitch
* input:help(show this message)\
'''

@itchat.msg_register('Text')
def remote(msg):

    #if not msg['FromUserName'] == msg['ToUserName']: return  
  # comment this line if you can't send message to yourself
    if msg['Text'] in ['pentest','help', u'帮助']:
        return pentesthelp
    else:
        commands = msg.get('Content', '')
        args = commands.split()

        # Custom aliases ,Can add commands by self
        if args[0] == 'nm':
            args[0] = '/home/pi/zhuangbi/pentest/whitehat/nmaptomasscan/startnm.py'

        if args[0] == 'sd':
            args[0] = '/home/pi/zhuangbi/pentest/whitehat/subdomain/subdomain.py'

        if args[0] == 'tz':
            args[0] = '/home/pi/zhuangbi/pentest/blackhat/cduan/cduantuozhan.py'

        if args[0] == 'sb':
            args[0] = '/home/pi/zhuangbi/pentest/whitehat/whatcms/whatcms.py'

        if args[0] == 'pl':
            args[0] = '/home/pi/zhuangbi/pentest/whitehat/payload/pl.py'

        if args[0] == 'jk1':
            args[0] = '/home/pi/zhuangbi/pentest/whitehat/cvejiankong/cvejk.py'

        if args[0] == 'jk2':
            args[0] = '/home/pi/zhuangbi/pentest/whitehat/monitoring/sf.py'

        if args[0] == 'ct':
            args[0] = '/home/pi/zhuangbi/pentest/whitehat/result/cat.sh'          

        try:
            proc =  subprocess.Popen(args,
                                    # shell=True,
                                    # executable='/bin/bash',
                                    # stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
            return proc.communicate()[0].strip()
        except OSError as e:
            return u'Commands/Files not found'

itchat.auto_login(enableCmdQR = True, hotReload = True)
itchat.run()
```

** 情景**

假定目标“secquan.org/192.168.10.1”

** 某一<font color="yellow">日</font>红<font color="yellow">日</font>落山，团长买完<font color="yellow">日</font>用品后写<font color="yellow">日</font>志，往<font color="yellow">日</font>知己多<font color="yellow">日</font>不见对<font color="yellow">日</font>语都有点生疏了。随即约三五知己寻欢饮酒作乐（某洗浴中心），但是天公不作美，突然安排了渗透任务，要求明<font color="yellow">日</font>出报告，难不成半夜回去做信息收集？<font color="yellow">日</font>夜<font color="yellow">日</font>？也太难受了。**

团长微微笑，对下面说“继续”，旋即掏出手机打开微信向着某finger账号开始发送

团长输入内容|后台完成操作|微信端显示内容
:---|:---|:---|
nm '192.168.10.1'|扫描ip端口|无
sd 'secquan.org'|爆破子域名|无
tz '192.168.10.1'|拓展目标站点|无
sb 'secquan.org'|识别cms类型|无
jk2 'secquan.org'|加入监控并收集人员信息|add successful

团长的能力你懂得
 ~~45秒~~
 ~~10分钟~~
 ~~30分钟~~
 ~~2个小时~~
 过了三个多小时后，鏖战稍歇

 团长再掏出手机，打开微信

 团长输入内容|后台完成操作|微信端显示内容
 :---|:---|:---|
 ct|ls查看result目录|执行ls result1.txt result2.txt newtarget.txt resultcms.txt
 ct 'resultcms.txt'|cat 该文件|456
 pl 'secquan.org'|批量打payload|无

//456是我创建进去的，为了有响应效果，secquan.org扫不出来..

发现whatcms已经完事了，那就查看结果，即resultcms.txt文件。旋即发现这是个 框架
正好团长手头上有几个0day是   框架的，正好也在pyaload库中，打 -->getshell
getshell比什么报告都有用是吧~

ok 穿上裤子写报告咯~~

一些截图

ct有点bug，因为我前面执行都是用>>result.txt 这个写法的，也就是说，后台执行就会直接创建这个result.txt文件，应该改为ls -al 看大小，貌似也不对，应该追踪pid，如果结束了再创建？一时之间 想不明白。

---

#### 第四章 装逼与内核FUZZ

完成了渗透模块(其实还在debug...)，寻找LEADER求抱腿

我：您看看，渗透流程化，秒杀  
LEADER: 嗯也还行，如果我想内核漏洞挖掘呢？时刻监视fuzzing的情况。  
我：无非是再加功能的事。您且稍待会儿  
LEADER:小伙子长得寒掺真是应了那句话，越丑越会写代码，抓紧时间搞，明天差不多了吧。
我：（我X你M）您请好吧您（笑的和~~假装~~高潮的小姐一样的赔笑脸）





先行说明：此处操作使我损失多台设备~需谨慎 树莓派3b armv7多处不兼容。










---


#### 不存在的第五章 装逼与智能语音


我：您看看，fuzz模块也睾丸了啊
LEADER:嗯还行，一开始说的语音呢，我想要口述命令，不想打命令 nm '192.168.0.1'
我：（mmp）就算能识别出是nm 192.168.0.1 但是没有单双引号是会在shell执行出错的，语音挺难弄的
LEADER:你再想想办法
我：（cnmb）在用深度学习训练一个渗透专用的语料，声明 .（点）为dot 双引号 单引号再代码中加个功能 除了前置命令（nm ct等）外其他都转成引号内内容。这样说不定可以。
LEADER:对了，能不能做成智能家居，控制我家空调摄像头
我：（wsnd nmsl）???（迷惑脸）
LEADER：顺便再实现语音闲聊，最好能离线唤醒比较省电，还有和云计算大数据结合起来我们也好申请经费。怎么样 很简单吧
我：... ...
LEADER:怎么了丑逼，你拿砖头干啥子

XX地看守所
我：不做人真好



终极模式：





---

“去你的吧” 两人鞠躬下台，我们下次德云社封箱再见。
后续将代码整体打包 至github 地址和密码见后续。
终末:希望各位dalao在冬至节都能成双成对，饺子就酒越吃越有。
