'''
https://labuladong.gitbook.io/algo/di-ling-zhang-bi-du-xi-lie/hua-dong-chuang-kou-ji-qiao-jin-jie
滑动窗口
'''
def minWindow(s,t):
    res = ""
    if not s:
        return ''
    if len(t) == 0:
        return res
    left = 0
    right = 0
    valid = 0
    need = {}
    window = {}
    max_len = max(len(s), len(t))
    for i in t:
        need[i] = need.get(i, 0) + 1
        window[i] = 0
    while right <= len(s):
        cur = s[right]
        right += 1
        if cur in need:
            window[cur] += 1
            if need[cur] == window[cur]:
                valid += 1
        while valid == len(need):
            res = s[left:right] if max_len >= len(s[left:right]) else res
            max_len = len(res)
            d = s[left]
            left += 1
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    return res






