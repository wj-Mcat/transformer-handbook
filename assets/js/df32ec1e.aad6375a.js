"use strict";(self.webpackChunktransformer_blog=self.webpackChunktransformer_blog||[]).push([[2519],{3057:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>p,contentTitle:()=>s,default:()=>u,frontMatter:()=>a,metadata:()=>r,toc:()=>l});var t=i(7462),d=(i(7294),i(3905)),o=i(1839);const a={sidebar_label:"Input",sidebar_position:1,description:"\u6a21\u578b\u7684\u8f93\u5165"},s="\u6a21\u578b\u7684\u8f93\u5165",r={unversionedId:"basic/input",id:"basic/input",title:"\u6a21\u578b\u7684\u8f93\u5165",description:"\u6a21\u578b\u7684\u8f93\u5165",source:"@site/docs/basic/input.md",sourceDirName:"basic",slug:"/basic/input",permalink:"/transformer/docs/basic/input",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/basic/input.md",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_label:"Input",sidebar_position:1,description:"\u6a21\u578b\u7684\u8f93\u5165"},sidebar:"tutorialSidebar",previous:{title:"\u6a21\u578b\u8bb2\u89e3",permalink:"/transformer/docs/category/\u6a21\u578b\u8bb2\u89e3"},next:{title:"decoder",permalink:"/transformer/docs/basic/decoder"}},p={},l=[{value:"\u8f93\u5165\u8ba1\u7b97\u903b\u8f91",id:"\u8f93\u5165\u8ba1\u7b97\u903b\u8f91",level:2},{value:"Token Embedding",id:"token-embedding",level:2},{value:"Position Embedding",id:"position-embedding",level:2},{value:"\u53c2\u8003\u94fe\u63a5",id:"\u53c2\u8003\u94fe\u63a5",level:2}],m={toc:l};function u(e){let{components:n,...a}=e;return(0,d.kt)("wrapper",(0,t.Z)({},m,a,{components:n,mdxType:"MDXLayout"}),(0,d.kt)("h1",{id:"\u6a21\u578b\u7684\u8f93\u5165"},"\u6a21\u578b\u7684\u8f93\u5165"),(0,d.kt)("p",null,"Transformer\u4e2d\u7684\u8f93\u5165\u5206\u4e3aEncoder\u548cDecoder\u7684\u8f93\u5165\uff0c\u5176\u539f\u7406\u548c\u8ba1\u7b97\u8fc7\u7a0b\u4e00\u81f4\uff0c\u4e14\u4e24\u8005\u7684Embedding\u6a21\u5757\u53ef\u5171\u4eab\u53c2\u6570\uff08\u5f53\u7136\u4e5f\u662f\u53ef\u4ee5\u4e0d\u5171\u4eab\uff09\u3002"),(0,d.kt)("p",null,"\u6a21\u578b\u7684\u8f93\u5165\u4e3b\u8981\u5206\u4e3a\u4e24\u4e2a\u90e8\u5206\uff1atoken embedding\u548cposition embedding\u3002\u524d\u8005\u9488\u5bf9\u4e8etoken\u5b58\u50a8\u5176\u76f8\u5173\u8bed\u4e49\u5411\u91cf\uff0c\u540e\u8005\u5bf9\u4f4d\u7f6e\u8fdb\u884c\u7f16\u7801\uff0c\u8ba1\u7b97\u793a\u4f8b\u5982\u4e0b\u6240\u793a\uff1a"),(0,d.kt)("p",null,(0,d.kt)("img",{src:i(2747).Z,width:"1402",height:"498"})),(0,d.kt)("h2",{id:"\u8f93\u5165\u8ba1\u7b97\u903b\u8f91"},"\u8f93\u5165\u8ba1\u7b97\u903b\u8f91"),(0,d.kt)("p",null,"\u4ee5\u4e0b\u6d41\u7a0b\u56fe\u6a21\u62df\u4e86\u4ee3\u7801\u8fd0\u7b97\u8fc7\u7a0b\u4e2d\u7684\u76f8\u5173\u903b\u8f91\uff1a"),(0,d.kt)(o.G,{chart:"graph LR;\n    sentence--\x3e|tokenizer|input_ids;\n    sentence--\x3e|tokenizer|position_ids;\n\n    subgraph token\n    input_ids--\x3e|lookup|token_embedding;\n    end\n\n    subgraph position\n    position_ids--\x3e|lookup|position_embedding;\n    end\n\n    subgraph hidden-states \n    token_embedding--\x3eembedding;\n    position_embedding--\x3eembedding;\n    end",mdxType:"Mermaid"}),(0,d.kt)("p",null,"\u5176\u4e2d",(0,d.kt)("inlineCode",{parentName:"p"},"sentence"),"\u4e3a\u539f\u59cb\u6587\u672c\uff0c\u901a\u8fc7",(0,d.kt)("inlineCode",{parentName:"p"},"tokenizer"),"\u4e4b\u540e\u5373\u53ef\u5f97\u5230",(0,d.kt)("inlineCode",{parentName:"p"},"input_ids"),"\u548c",(0,d.kt)("inlineCode",{parentName:"p"},"position_ids"),"\u4e24\u4e2a\u6570\u636e\uff0c\u4f8b\u5982:"),(0,d.kt)("pre",null,(0,d.kt)("code",{parentName:"pre",className:"language-py"},"input_ids = [465, 263, 2163, 28736]\nposition_ids = [0, 1, 2, 3]\n")),(0,d.kt)("p",null,(0,d.kt)("inlineCode",{parentName:"p"},"input_ids"),"\u4f1a\u5728",(0,d.kt)("inlineCode",{parentName:"p"},"WordEmbedding"),"\u5f53\u4e2d\u6839\u636e\u7d22\u5f15\u68c0\u7d22\u51fa\u5bf9\u5e94",(0,d.kt)("inlineCode",{parentName:"p"},"token"),"\u7684\u8bed\u4e49\u5411\u91cf\uff0c",(0,d.kt)("inlineCode",{parentName:"p"},"position_ids"),"\u4f1a\u5728",(0,d.kt)("inlineCode",{parentName:"p"},"PositionEmbedding"),"\u5f53\u4e2d\u6839\u636e\u7d22\u5f15\u68c0\u7d22\u51fa\u5bf9\u5e94",(0,d.kt)("inlineCode",{parentName:"p"},"position"),"\u7684\u4f4d\u7f6e\u5411\u91cf\u3002\u8fd9\u4e24\u8005\u90fd\u662f\u67e5\u8868\u8fc7\u7a0b\uff0c\u76f8\u5bf9\u6bd4\u8f83\u7b80\u5355\u3002"),(0,d.kt)("h2",{id:"token-embedding"},"Token Embedding"),(0,d.kt)("p",null,(0,d.kt)("inlineCode",{parentName:"p"},"Token Embedding"),"\u4e5f\u53ef\u79f0\u4e3a",(0,d.kt)("inlineCode",{parentName:"p"},"Word Embedding"),"\uff0c\u4e5f\u6709\u76f8\u5173\u7684\u53d1\u5c55\u5386\u7a0b\uff0c\u4ece\u6700\u65e9\u7684\u57fa\u4e8e\u8bcd\u9891\u7387\u7edf\u8ba1\u7684\u6709Count Vector\u3001\u5173\u952e\u8bcd\u7279\u5f81\u65b9\u6cd5 TF-IDF Vector\u4ee5\u53ca\u8bcd\u5171\u73b0\u77e9\u9635Co-Occurence Vector\u7b49\u4e0d\u540c\u7c7b\u522b\u7684\u65b9\u6cd5\u3002"),(0,d.kt)("h2",{id:"position-embedding"},"Position Embedding"),(0,d.kt)("h2",{id:"\u53c2\u8003\u94fe\u63a5"},"\u53c2\u8003\u94fe\u63a5"),(0,d.kt)("ul",null,(0,d.kt)("li",{parentName:"ul"},(0,d.kt)("a",{parentName:"li",href:"https://zhuanlan.zhihu.com/p/385146997"},"Word Embedding\u7684\u53d1\u5c55\u548c\u539f\u7406\u7b80\u4ecb"))))}u.isMDXComponent=!0},2747:(e,n,i)=>{i.d(n,{Z:()=>t});const t=i.p+"assets/images/embedding-3b1fac974c3f6dfad4633725d8ca4e63.png"}}]);