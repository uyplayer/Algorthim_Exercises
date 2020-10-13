# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time : 2020/9/26 18:37 
# @Author : UYPLAYER
# @File : kmp1.py 
# @Software: PyCharm
Change Activity:2020/9/26
-------------------------------------------------
"""

# def partial_table(s):
#     prefex = set()
#     postfix = set()
#     res = [0]
#     for i in range(1,len(s)):
#         prefex.add(s[:i])
#         postfix = {s[j:i + 1] for j in range(1, i + 1)}
#         print("postfix:",postfix)
#         print("postfix:",postfix)
#         print(postfix & postfix)
#         res.append(len((postfix & postfix or {''}).pop()))
#     return res
#
# def kmp(s, p):
#     m = len(s)
#     n = len(p)
#     cur = 0
#     table = partial_table(s)
#     while cur <= m - n:
#         for i in range(n):
#             if s[i+cur] == p[i]:
#                 cur += max(i - table[i - 1], 1)
#                 break
#             else:
#                 return True
#         return False
#
# print(partial_table('ABCDABD'))
# print(kmp("BBC ABCDAB ABCDABCDABDE", "ABCDABD"))


def nextTable(s):
    i = 0
    next = [-1]
    j = -1
    while i < len(s)-1 :
        if j == -1 or s[i] == s[j]:
            i += 1
            j += 1
            next.append(j)
        else:
            j = next[i]
    return  next

def kmp(p,s,id):
    tanle = nextTable(s)
    i = id
    j = 0
    while (i<len(p) and j<len(s)):
        if (j == -1 or p[i] == s[j]):
            i += 1
            j += 1
        else:
            j = tanle[j]
        if (j >= len(p)):
            return i - len(p)
        else:
            return 0
p = "ADOBECODEBANC"
s = "BANC"
print(kmp(p,s,0))




