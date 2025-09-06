---
title: Hello World
---
Welcome to [Hexo](https://hexo.io/)! This is your very first post. Check [documentation](https://hexo.io/docs/) for more info. If you get any problems when using Hexo, you can find the answer in [troubleshooting](https://hexo.io/docs/troubleshooting.html) or you can ask me on [GitHub](https://github.com/hexojs/hexo/issues).

## Quick Start

### Create a new post

``` bash
$ hexo new "My New Post"
```

More info: [Writing](https://hexo.io/docs/writing.html)

### Run server

``` bash
$ hexo server
```

More info: [Server](https://hexo.io/docs/server.html)

### Generate static files

``` bash
$ hexo generate
```

More info: [Generating](https://hexo.io/docs/generating.html)

### Deploy to remote sites

``` bash
$ hexo deploy
```

More info: [Deployment](https://hexo.io/docs/one-command-deployment.html)

## Additional Information

This blog is deployed in this [Github repo](https://github.com/zrkc/zrkc.github.io/tree/main), with source code stored in the branch [hexo](https://github.com/zrkc/zrkc.github.io/tree/hexo).

To maintain the blog in a new device, clone the branch [hexo](https://github.com/zrkc/zrkc.github.io/tree/hexo) and run hexo:

```
git clone -b hexo https://github.com/zrkc/zrkc.github.io.git
cd .\zrkc.github.io\
npm install -g hexo-cli
npm install
npm install hexo-deployer-git
```

(And delete `.deploy_git` if downloaded from remote, this is only used local. I have added it to `.gitignore`)

To syn from local to remote (hexo branch, not main!):

```
git add .
git commit -m "some comments"
git push origin hexo
```

To syn from remote to local:

```
git pull
```

[reference](https://cloud.tencent.com/developer/article/1046404)