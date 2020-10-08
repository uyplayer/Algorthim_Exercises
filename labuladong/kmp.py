# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time : 2020/9/26 17:25 
# @Author : UYPLAYER
# @File : kmp.py 
# @Software: PyCharm
Change Activity:2020/9/26
-------------------------------------------------
"""

'''
原理 http://www.ruanyifeng.com/blog/2013/05/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm.html
'''
def kmp_match(s, p):
    m = len(s)
    n = len(p)
    cur = 0  # 起始指针cur
    table = partial_table(p)
    while cur <= m - n:     #只去匹配前m-n个
        for i in range(n):
            if s[i + cur] != p[i]:
                cur += max(i - table[i - 1], 1)  # 有了部分匹配表,我们不只是单纯的1位1位往右移,可以一次移动多位
                break
        else:           #for 循环中，如果没有从任何一个 break 中退出，则会执行和 for 对应的 else
                        #只要从 break 中退出了，则 else 部分不执行。
            return True
    return False


# 部分匹配表
def partial_table(p):
    '''''partial_table("ABCDABD") -> [0, 0, 0, 0, 1, 2, 0]'''
    prefix = set()
    postfix = set()
    ret = [0]
    for i in range(1, len(p)):
        prefix.add(p[:i])
        postfix = {p[j:i + 1] for j in range(1, i + 1)}
        ret.append(len((prefix & postfix or {''}).pop()))
    return ret


# print(partial_table("ABCDABD"))
#
# print(kmp_match("BBC ABCDAB ABCDABCDABDE", "ABCDABD"))


#KMP算法
#首先计算next数组，即我们需要怎么去移位
#接着我们就是用暴力解法求解即可
#next是用递归来实现的
#这里是用回溯进行计算的
def calNext(str2):
    i=0
    next=[-1]
    j=-1
    while(i<len(str2)-1):
        if(j==-1 or str2[i]==str2[j]):#首次分析可忽略
            i+=1
            j+=1
            next.append(j)
        else:
            j=next[j]#会重新进入上面那个循环
    return next
print(calNext('abcabx'))#-1,0,0,0,1,2
def KMP(s1,s2,pos=0):#从那个位置开始比较
    next=calNext(s2)
    i=pos
    j=0
    while(i<len(s1) and j<len(s2)):
        if(j==-1 or s1[i]==s2[j]):
            i+=1
            j+=1
        else:
            j=next[j]
    if(j>=len(s2)):
        return i -len(s2)#说明匹配到最后了
    else:
        return 0
s1 = "acabaabaabcacaabc"
s2 = "abaabcac"
print(KMP(s1,s2))