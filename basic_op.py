# 数据结构的基本操作
# https://labuladong.gitbook.io/algo/di-ling-zhang-bi-du-xi-lie/xue-xi-shu-ju-jie-gou-he-suan-fa-de-gao-xiao-fang-fa#er-shu-ju-jie-gou-de-ji-ben-cao-zuo

'''数组'''
a = [1,2,3,4,5]
for i in a:
    print(i)
for i in range(len(a)):
    print(a[i])

'''链表遍历框架'''

class ListNode:
    def __init__(self,x):
        self.value = x
        self.node  = None 
# 访问我们的链表        
def traverse(head):
    while head:
        p = ListNode(0) 
        p = head
        if p:
            p = p.next
        else:
            return 
        print(p.value)


'''二叉树遍历框架'''
class TreeNode:
    def __init__(self,x):
        self.value = x
        self.right = None
        self.left = None
#先序遍历   
def pre(root):
    if root == None :
        return None 
    print(root.value)
    pre(root.left)
    pre(root.right)
##中序遍历
def premiddle(root):
    if root == None :
        return None 
    premiddle(root.left)
    print(root.value)
    premiddle(root.right)

#后序遍历      
def back(root):
    if root == None :
        return None
    back(root.left)
    back(root.rught)
    print(root.value)    
        







