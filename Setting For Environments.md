**该readme文件中介绍在服务器（linux系统）上安装各种环境，包括tensorflow,pytorch,无root权限下升级gcc版本等，具体内容不断更新！**



#### 1.安装anaconda

一般服务器上的个人用户基本上没有sudo权限，所以建议在自己的账户里安装一个anaconda，然后可以配置不同的环境来使用。

首先进入anaconda官网下载anaconda，之后安装（安装在自己个人账户下），如：

```shell
bash Anaconda3-5.0.0-Linux-x86_64.sh
```

此时按照步骤提示（可以阅读，观察是否为自己需要的）一步步安装，注意最后会询问你是否将a naconda添加到配置文件中，这一步选择yes；如果没有添加，自己一定要添加环境变量到.bashrc中，.bashrc文件非常重要。

关于.bashrc文件(https://blog.csdn.net/eleanoryss/article/details/70207767)

1.切换anaconda的python和建立的虚拟环境中的python：需要在.bashrc文件添加环境变量。export PATH=/home/sz/anaconda3/bin:$PATH，这是anaconda自带的python；export PATH="/home/sz/anaconda3/envs/pytorch/bin:$PATH"，这是建立的虚拟环境中的python。在激活相应的虚拟环境后，可以输入python查看是上面版本，是否是相应虚拟环境的python

2.关于.bashrc文件：.bashrc文件一般隐藏在/home/sz文件夹下面，每次一打开xshell，就会重新运行一下该文件。如果打开端口的时候不是base环境，可以source .bashrc进入base环境。(在后面4会说明原因)

3.查看.bashrc文件中环境变量：echo $PATH

4.每次修改.bashrc后，使用source \~/.bashrc就可以立刻加载修改后的设置，使之生效。 一般会在.bash_profile文件中显式调用.bashrc。登陆linux启动bash时首先会去读取~/.bash_profile文件，这样\~/.bashrc也就得到执行了，个性化设置也就生效了(https://www.jianshu.com/p/a5e9a5d18335)。

我曾经不小心把.profile文件删除了(我也不知道我怎么删除的)，所以每次重新启动端口时，都不执行.bashrc文件，都不会出现base的环境。然后看了别人的.profile文件的内容，一模一样地copy了一份存进/home/sz中就可以了。不执行.bashrc文件就没办法使用conda命令，只能source .bashrc才能使用conda命令。

编辑.bashrc文件：

```shell
vim ~/.bashrc
```

我的.bashrc文件内容如下：

```shell
__conda_setup="$('/home/sz/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/sz/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/sz/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/sz/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
```

由于端口一打开就要执行.bashrc文件，所以.bashrc中的环境变量一定要准确。正常安装好anaconda之后，一打开端口，就会进入base环境，可以接着安装其他需要的虚拟环境。



#### 2.使用anaconda建立虚拟环境安装tensorflow或者pytorch

pytorch和tensorflow的成功安装和使用都需要有一系列硬件和软件支持和兼容，主要有显卡算力，服务器的驱动版本，cuda版本等。

2.1.建立虚拟环境

```shell
conda create -n pytorch13_cuda90 pip python=3.6
```

然后激活该环境，在该环境中继续安装所需的库：

```shell
conda activate pytorch13_cuda90
```



2.2.理解兼容关系

（1）cuda版本需要和服务器显卡驱动版本兼容（nvidia-smi查看，右上角CUDA VERSION为当前驱动可以支持的最大cuda版本），兼容关系如下：

![cuda-driver_version](/Users/momo/Documents/video/cuda-driver_version.png)

（2）cuda版本需要和显卡算力（低级的显卡无法使用cuda加速）匹配

可以通过网址(https://developer.nvidia.com/cuda-gpus) 查看显卡算力，这一点我没有太多经验，由于在自己的服务器上试过Tesla K80和Tesla P40，这两种显卡用cuda9.0是可以加速的，超过这个cuda版本是没办法使用显卡加速的。

（3）cuda和pytorch、tensorflow版本要兼容，这一点可以在官网查看

pytorch官网（https://pytorch.org/get-started/previous-versions/）

tensorflow官网查看（https://tensorflow.google.cn/install/gpu?hl=zh-cn#install_cuda_with_apt）



2.3.安装pytorch

首先激活pytorch相关的环境，进入pytorch官网，可以看到pytorch版本和cuda的匹配关系，安装匹配的pytorch和cuda，如：

```shell
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```

安装完之后要验证pytorch是否可以检测到cuda，以及cuda是否可以检测到显卡加速：

注意不检测，一般跑代码时都会报错：RuntimeError: CUDA error: no kernel image is available for execution on the device

说明cuda无法使用，用下面两段代码验证：

```python
import torch
print(torch.cuda.is_available())
```

如果输出True，说明CUDA安装正确并且能被Pytorch检测到，并没有说明是否正常使用，要想看Pytorch能不能调用cuda加速，还需要简单地测试：

```python
import torch
a = torch.Tensor(5,3)
a = a.cuda()
print(a)
```

或者：

```python
import torch
a = torch.Tensor(5,3)
device = torch.device('cuda:0')
a = a.to(device)
print(a)
```

此时一般会报同样的错误，原因在于显卡算力和CUDA不匹配。要么更换显卡提高显卡算力，要么降低CUDA的版本。但是由于一些比较新的库（如detectron2）对pytorch版本要求很高，此时CUDA版本没办法降低，可以建议使用高算力的显卡。



在安装pytorch这一步经常会出现等待时间较长的现象，此时可以添加国内镜像来安装：

```shell
# 配置国内源，方便安装Numpy,Matplotlib等
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# 配置国内源，安装PyTorch用
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# 显示源地址
conda config --set show_channel_urls yes
```

如果想加速pytorch的下载，直接使用第三条命令即可，然后再输入安装指令（使用国内源一定要去掉-c pytorch这个参数）：

如果pytorch官方指令为：

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

则去掉-c pytorch来使用国内源：

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1
```

换回默认源：

```shell
conda config --remove-key channels # 会删除.condarc文件
```

在执行conda config命令的时候，会在当前用户目录下创建.condarc文件，可以查看更换源前后该文件内容的变化。



2.4.安装tensorflow

首先激活tensorflow相关的环境，进入tensorflow官网，可以看到tensorflow版本和cuda的匹配关系，安装匹配的tensorflow和cuda，如：

```shell
conda install cudatoolkit==9.0 tensorflow-gpu==1.10.0 # 同时指定cuda版本和tensorflow-gpu版本
conda install tensorflow-gpu==1.15 # tensorflow1.x系列最终版本
conda install tensorflow-gpu=2.0.0
```

这里最好使用conda安装，conda可以安装好各种必备的工具包，如cudnn和cudatoolkit，对于tensorfow 2.0.0系列，要求linux上驱动要大于410.x,cuda版本只能是10，cudnn版本要大于7.4.



安装好了之后可以验证tensorflow是否可以正常使用和是否可以使用:

```python
#验证tensorflow是否可以正常使用
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

```python
#验证是否可以使用gpu加速
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

如果可以使用gpu加速，则会输出gpu的信息：

2020-03-14 16:07:22.167039: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0, 1
2020-03-14 16:07:22.167157: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-14 16:07:22.167179: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 1
2020-03-14 16:07:22.167193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N Y
2020-03-14 16:07:22.167206: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 1:   Y N
2020-03-14 16:07:22.167603: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10756 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:06:00.0, compute capability: 3.7)
2020-03-14 16:07:22.167756: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10756 MB memory) -> physical GPU (device: 1, name: Tesla K80, pci bus id: 0000:07:00.0, compute capability: 3.7)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K80, pci bus id: 0000:06:00.0, compute capability: 3.7
/job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla K80, pci bus id: 0000:07:00.0, compute capability: 3.7
2020-03-14 16:07:22.167928: I tensorflow/core/common_runtime/direct_session.cc:288] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K80, pci bus id: 0000:06:00.0, compute capability: 3.7
/job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla K80, pci bus id: 0000:07:00.0, compute capability: 3.7



对于tensorflow2.0版本的验证，很多模块需要加上tf.compat.v1.xxxx：

```python
#验证tensorflow是否可以正常使用
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
print(hello)
```

```python
#验证是否可以使用gpu加速
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
```



#### 3.更改anaconda虚拟环境的名字

```shell
conda create --name new_name --clone old_name # 先克隆旧环境，同时更改名字
conda remove --name old_name --all # 再删除旧环境
```



#### 4.anaconda移植环境

当在一台机器上安装好环境之后，如果想在另一台机器上安装一模一样的环境，可以使用环境移植。（3中环境克隆只能在相同机器上进行）

```shell
conda env export > environment.yml # 导出环境为yml文件
conda env create -f environment.yml # 根据yml文件创建新的环境（将yml文件放到另一台机器上即可）
```



#### 5.虚拟环境python版本问题

正常情况下，虚拟环境中的python版本在创建该虚拟环境时就已经指定了，并且默认anaconda3中可以创建各种python3的虚拟环境，anaconda2中可以创建各种python2的虚拟环境。

曾经遇到这样的问题，在anaconda3中，创建了python2和python3的虚拟环境，使用python2对应的虚拟环境一段时间后，再次激活python3对应的虚拟环境，此时python版本显示为2.x的版本（即python2虚拟环境的python版本，一般服务器很少出现这种情况，原因未知）。此时要想使用python3，直接使用指令python3即可启用python3.x的版本，如：

```shell
python3 xx.py
```



#### 6.无root权限下升级gcc版本

github上很多高手会专门写用于特定用处的python库，安装这些库一般都需要较高版本的gcc，而一般用户又没有权限直接升级，这里介绍无root权限的情况下升级gcc。

在Linux下，如果有root权限的话，使用sudo apt install 就可以很方便的安装软件，而且同时也会帮你把一些依赖文件也给编译安装好。但是如果不是用的自己的机器，一般情况下是没有root 权限的。所以就需要自己动手下载tar文件，解压安装。在安装中遇到的最大的问题是依赖的问题。
i)首先下载gcc压缩包并解压：

在网址https://ftp.gnu.org/gnu/gcc 找到需要下载的版本，这里选择gcc-5.5.0.tar.gz(不要下载太新的版本，可能会不支持，满足要求即可)，上传到服务器(我自己的服务器路径为/home/sunzheng/gcc-5.5.0.tar.gz);

解压：

```shell
tar zxvf gcc-5.5.0.tar.gz
```

解压之后出现文件夹gcc-5.5.0；

进入该文件夹（后续操作都在该解压缩文件夹中进行）；

```shell
cd gcc-5.5.0
```

ii)下载gcc，和gcc依赖的包到文件夹gcc-5.5.0中

```shell
./contrib/download_prerequisites
```

如果运行过程中出现错误，可以依次运行文件中每个命令，来安装或者解压gcc所依赖的包；

iii)编译gcc

在gcc解压缩根目录下(gcc-5.5.0下)新建一个文件夹，然后进入该文件夹配置编译安装：

```shell
mkdir gcc-build
cd gcc-build
../configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=/path/to/install --enable-threads=posix
make -j64    # 多线程编译，否则很慢很慢很慢，能多开就多开几个线程
make install
```

`path/to/install`就是要安装GCC的目录，比如我的服务器上就是/home/sunzheng/GCC-5.5.0，一定要是有安装权限的目录，所以第二条指令就是../configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=/home/sunzheng/GCC-5.5.0 --enable-threads=posix

iv)为当前用户配置系统环境变量

打开～/.bashrc文件：

```shell
vim ~/.bashrc
```

在末尾加入：

```shell
export PATH=/path/to/install/bin:/path/to/install/lib64:$PATH
export LD_LIBRARY_PATH=/path/to/install/lib/:$LD_LIBRARY_PATH
```

在我的服务器上就是：

```shell
export PATH=/home/sunzheng/GCC-5.5.0/bin:/home/sunzheng/GCC-5.5.0/lib64:$PATH
export LD_LIBRARY_PATH=/home/sunzheng/GCC-5.5.0/lib/:$LD_LIBRARY_PATH
```

一定要确保安装路径在`$LD_LIBRARY_PATH`和`$PATH`之前，这样安装的程序才能取代之前系统默认的程序。同样地，也可以安装别的软件到自己的目录下并采用以上方式指定默认程序。

更新bashrc文件：

```shell
source ~/.bashrc
```

或者重启shell.

v)输入gcc -v检查版本

至此gcc升级完成。

![gcc](/Users/momo/Documents/video/gcc.jpeg)



#### 7.管理员管理服务器

​    一般一台服务器会有多个用户使用，每个用户都有自己的一个账号，但都没有sudo权限，只有管理员才有权限管理账号。

（1）管理员添加账号

```shell
sudo useradd -m usename # 添加用户名
sudo passwd usename # 设置用户名的密码，然后连续两次输入密码即可
```



（2）sudo apt install xxx报错

比如：

升级了 0 个软件包，新安装了 2 个软件包，要卸载 0 个软件包，有 285 个软件包未被升级。

需要下载 252 kB 的归档。
解压缩后会消耗 1,461 kB 的额外空间。
您希望继续执行吗？ [Y/n] y
错误:1 http://cn.archive.ubuntu.com/ubuntu xenial/main amd64 libtinfo-dev amd64 6.0+20160213-1ubuntu1
  暂时不能解析域名“cn.archive.ubuntu.com”
错误:2 http://cn.archive.ubuntu.com/ubuntu xenial/main amd64 libncurses5-dev amd64 6.0+20160213-1ubuntu1
  暂时不能解析域名“cn.archive.ubuntu.com”
E: 下载 http://cn.archive.ubuntu.com/ubuntu/pool/main/n/ncurses/libtinfo-dev_6.0+20160213-1ubuntu1_amd64.deb  暂时不能解析域名“cn.archive.ubuntu.com” 失败
E: 下载 http://cn.archive.ubuntu.com/ubuntu/pool/main/n/ncurses/libncurses5-dev_6.0+20160213-1ubuntu1_amd64.deb  暂时不能解析域名“cn.archive.ubuntu.com” 失败

E: 有几个软件包无法下载，要不运行 apt-get update 或者加上 --fix-missing 的选项再试试？

报错原因提示域名解析失败，那么就需要加一个万能的域名：

1）编辑文件 

```shell
sudo vim /etc/resolv.conf
```

2）在最后加上 nameserver 8.8.8.8

3）保存退出

（3）安装各种终端工具

```shell
sudo apt install git
sudo apt install zsh
sudo apt install tmux
```

（4）管理员安装好这些之后，可以在个人用户下安装oh-my-zsh，用于个性化设置

直接输入指令：

```
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
```

或者：

```
git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
chsh -s /bin/zsh
```

安装好zsh之后可以更改zsh终端的主题风格：

```shell
vim ~/.zshrc #编辑配置文件
ZSH_THEME="robbyrussell" #在脚本中将这一句替换成相应的风格，如ZSH_THEME="agnoster"，ZSH_THEME="ys"
source ~/.zshrc #激活配置文件
```



#### 8.配置本地VS Code及pycharm连接服务器

1.配置VS Code远程连接服务器

​    VS Code是一款功能强大的编辑器，非常的轻巧，远程连上服务器之后，更加的强大。下面介绍连接的方法，网上的教程乱七八糟，大多都是相互抄的，并且失败的很多，这里给出自身配置的过程，一次性成功。

（1）安装拓展Remote Development

![vscode1](/Users/yihe/Documents/sunzheng/interview/vscode1.png)

这个拓展安装好了之后，其余的必须模块都安装了，可以省很多事情。

（2）配置ssh

点击界面最左下角的绿色按钮，出现配置信息：

![vscode2](/Users/yihe/Documents/sunzheng/interview/vscode2.png)

选择第一个Remote-SSH:

![vscode3](/Users/yihe/Documents/sunzheng/interview/vscode3.png)

由于我已经配置好了一个sunzheng，直接连接即可，如果第一次配置，或者连接其他服务器，可以点Add New SSH Host.

![vscode4](/Users/yihe/Documents/sunzheng/interview/vscode4.png)

输入连接信息，如ssh sunzheng@172.18.32.216

![vscode5](/Users/yihe/Documents/sunzheng/interview/vscode5.png)

接下来选择配置文件的位置，我默认选择的第一个。

![vscode6](/Users/yihe/Documents/sunzheng/interview/vscode6.png)

在里面输入需要连接的服务器的相关信息，保存。我这里面配置了两台服务器的信息，配置好了之后，再次点击绿色按钮即可选择相应的服务器，输入密码即可连接，连接好之后界面如下：

![vscode7](/Users/yihe/Documents/sunzheng/interview/vscode7.png)

选择Open folder打开/home/sunzheng/等文件夹，可以自行选择打开文件主目录，如下：

![vscode8](/Users/yihe/Documents/sunzheng/interview/vscode8.png)

此时可以发现并没有和运行环境相关的设置，还需要安装anaconda和python两个拓展：

![vscode9](/Users/yihe/Documents/sunzheng/interview/vscode9.png)

![vscode10](/Users/yihe/Documents/sunzheng/interview/vscode10.png)

安装好了需要重启一下vscode，因为python这个拓展需要Reload Required。重启之后的界面如下，可以发现界面的最下方已经有了可以选择的环境：

![vscode11](/Users/yihe/Documents/sunzheng/interview/vscode11.png)

界面下半部分是服务器的终端，可以进行命令的执行，上半部分是代码编辑，还有文件索引目录，相比较于直接在服务器上使用vim编写方便多了。

windows上配置问题汇总：

（1）windows上配置vscode也是一样的操作流程，不过在windows要注意一个问题，配置文件的路径不能有中文，我曾经的路径为C:\Users\中文\\.ssh\config，每次连接的时候报错：could not establish connection to "xxx". Connecting was canceled. 在网上找了很多，大部分都提示要指明远程连接服务器的平台，这个确定没问题之后要注意下配置文件的路径有没有中文。如果有中文，需要更改配置文件的路径。

解决方法：

把这个配置文件复制了一份到别的英文路径下，重新指定了配置文件的路径(选择Setting specify a custom configuration file)，即可连接成功。

（2）报错"An SSH installation couldn't be found"，这种错误一般出现在win7系统中，因为win10自带了一个SSH客户端--Git，如果没有SSH客户端就会报错。

解决方法：

windows上安装Git，记住安装路径，然后在vscode的settings.json文件中添加：

```
"remote.SSH.path":"F:\\Git\\usr\\bin\\ssh.exe",
"remote.SSh.remotePlatform":{"sunzheng":"linux"}
```

重启vscode即可。



#### 9.服务器更新显卡驱动

一般来说，服务器供应商在安装服务器时会帮你安装好显卡驱动。由于linux一般都是自动更新下载内核，导致在服务器重启之后，内核更新了，但是显卡驱动还是老版本，这时输入nvidia-smi就会报错

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the lastest NVIDIA driver is installed and running.
```

表示驱动无法使用，这时就需要安装新的驱动，以保证和新的内核匹配。

直接在官网( https://www.nvidia.cn/geforce/drivers )下载对应显卡的最新驱动，如下图所示：

/source_image/nvidia_driver1

/source_image/nvidia_driver2

一般选择最新的即可(驱动更新都是跟着内核走的)。

将下载好的驱动上传至服务器上，要注意下文件的权限，要保重该驱动是可以运行的，输入ls -la xxx或者ls -ls观察所有文件的权限，如下图所示：

/source_image/nvidia_driver3

链接( https://www.cnblogs.com/lixiaolun/p/5391803.html )中介绍了权限字母的意义：

以-rwxr-xr-x为例，共十个字符，第一个字符-表示文件，如果是d则表示文件目录；第2位到第4位rwx表示所有用户(即sunzheng)的权限，可读(r)，可写(w)，可执行(x)；第5位到第7位r-x表示所有组(即sunzheng)的权限，可读，可执行；第8位到第10位表示其他人的权限。

如果想改变权限，可以使用如下命令：

```shell
chmod o+w NVIDIA-Linux-x86_64-440.100.run # 表示给其他人添加可写权限
chmod a+x NVIDIA-Linux-x86_64-440.100.run # 表示给所有人添加执行权限
chmod 777 NVIDIA-Linux-x86_64-440.100.run # 表示对所有人开放所有权限
chmod 777 * # 对当前目录中的所有文件对所有人开放所有权限
```

```
u:所有者(user)
g:所有者所在的组群(group)
o:其他人，但不是u和g(other)
a:所有的人，包括u,g,o
r:可读(read)
w:可写(write)
x:可执行
其中rwx也可以用数字来表示，具体的可以看链接
```



将文件改为可执行之后，即可进行安装：

```
sudo chmod a+x NVIDIA-Linux-x86_64-450.57.run
sudo ./NVIDIA-Linux-x86_64-450.57.run -no-x-check -no-nouveau-check -no-opengl-files
```

注：如果以后都不想再更新显卡驱动的话，可以直接关闭linux内核自动更新：

```
zkti@zkti:~$ sudo apt-mark hold linux-image-generic linux-headers-generic
[sudo] password for zkti:
linux-image-generic set on hold.
linux-headers-generic set on hold.
```

重新开启：

```
sudo apt-mark unhold linux-image-generic linux-headers-generic
```



#### 10.linux运行c++文件以及一点点编译原理

##### 10.1 linux添加环境变量运行cpp编译文件a.out

（1）在windows中，安装了g++（专门编译c++）或者gcc（可以用于编译c++，forthon等语言，是一个比较全面的编译工具）之后，在cpp文件所在目录下命令行输入：

```
g++ test.cpp
```

即可生成a.exe文件，再输入：

```
a.exe
```

既可以执行该文件，输出文件内容。



（2）在linux中，在cpp所在目录下命令行输入：

```
g++ test.cpp
```

会生成a.out文件（与windows系统的差异），再输入：

```
./a.out
```

既可以执行该文件，输出文件内容。（如果输入a.out，则一般无法识别，除非该目录被添加进环境变量中）。

将cpp所在目录添加进环境变量：

在安装anaconda时会在.bashrc或者.zshrc中添加conda的环境变量，这里再介绍一个添加任意一个路径的环境变量的方法。

执行指令：

```
echo $PATH
```

会输出当前.profile中的环境变量，如：

```
/home/sunzheng/anaconda3/envs/pytorch12_cuda10/bin:/home/sunzheng/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
```

这是linux系统下永久存在的环境变量（除非从.profile中删除）。
执行指令：

```
export PATH="/home/sunzheng/CPP"
```


会在当前端口中**临时**加入环境变量（重开一个端口会消除这个环境变量），此时在/home/sunzheng/CPP中命令行执行a.out可以直接输出结果；如果没有添加该文件下的环境变量，只能输入./a.out才可以输出结果。

##### 10.2编译过程

指令：

```
g++ test.cpp
```

是在编译源码生成可执行文件，实际上这一个指令分为两步：编译（编译为目标文件，主要检查语法错误）和联接（将目标文件联接为可执行文件）

具体内容通过下面一个函数声明的例子来解释这两个过程：

add.cpp

```c++
#include<iostream>

using namespace std;

int Sum(int a, int b);

int main()
{
    int sum = Sum(3,4);
    cout<<sum<<endl;
    return 0;
}
```

这段代码给出了add函数的声明，但没有给出函数的定义，所以直接进行编译g++ add.cpp会报错:

```
/tmp/ccw4ccHo.o: In function `main':
add.cpp:(.text+0x13): undefined reference to `Sum(int, int)'
collect2: error: ld returned 1 exit status
```

将编译生成可执行文件的过程拆分为两步：

（1）编译（即检查源码是否有语法错误）

```
g++ -c add.cpp
```

这一步可以正常执行，因为add.cpp中虽然缺少了函数定义，但是没有语法错误，可以成功生成目标文件add.o；

（2）联接（将目标文件联接为可执行文件）

```
g++ add.o
```

此时发生报错：

```
add.o: In function `main':
add.cpp:(.text+0x13): undefined reference to `Sum(int, int)'
collect2: error: ld returned 1 exit status
```

即缺少add函数的定义。

解决方法一是在add.cpp源码中添加函数定义，二是可以再写一个源文件add1.cpp:

```
int Sum(int a, int b)
{
    return a+b;
}
```

执行：

```
g++ -c add1.cpp
```

生成add1.o文件。

再执行：

```
g++ add.o add1.o
```

可生成a.out可执行文件。



#### 11.vscode自带的git以及gitlens插件

![vscode_git](G:\Documents\sunzheng\interview\vscode_git.png)

导航栏的第三个标志就是vscode自带的git工具，但只可以看到更改的变化。例如，在master里面进git操作的时候，比如git commit，该文件夹里面会生成一个.git文件夹，记录git的操作。然后当前vscode打开了master目录下的文件，此时如果修改master中的文件，则这个工具会自动将当前修改的版本和上一次git commit的版本进行对比，如下：

![vscode_git_change](G:\Documents\sunzheng\interview\vscode_git_change.png)



安装gitlens插件之后就可以发现在上图git工具下的导航栏中，还有COMMITS，FILE HISTORY，BRANCHES等功能。

