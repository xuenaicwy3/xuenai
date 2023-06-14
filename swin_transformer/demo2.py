# 题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？、
# a = ['1', '2', '3', '4']
#
# b = []
# for i in a:
#     print(i)
#     for j in [x for x in a if x != i]:
#         print(j)
#         for m in [x for x in a if x != i and x != j]:
#             print(m)
#             b.append(int(i + j + m))
# print(b)
# print('互不相同且无重复数字的三位数总计 %s 个' % len(b))

# nums = [1,2,2,3,4,5,1,3]
# # for sub in range(1, 8):
# #     print(sub)
# #print(len(nums))
# m = 6
# items = []
# for index in range(len(nums)):
#     #print(index)
#     for sub in range(index+1, len(nums)):
#         if nums[index] + nums[sub] == m:
#             if index not in items and sub not in items:
#                 items.append(index)
#                 items.append(sub)
#
# print(items)

# l = [1, 7, 5, 6,2,8,3, 9,4]
# n = len(l)
# print(n)
# for m in range(n-1):
#     print(m)
#     flag = True
#     for i in range(n-m-1):
#         if l[i] > l[i+1]:
#             l[i], l[i+1] = l[i+1], l[i]
#             flag = False
#     if flag:
#         break
# print(l)

# def str_reverse():
#     str1 = "hell0 xiao mi"
#     # 将字符串以空格为分割点，分割为列表
#     str_list = str1.split(' ')
#     # 将列表反转
#     str_list.reverse()
#     # 将反转的列表重新拼接成字符串
#     res = " ".join(str_list)
#     print(res)
#
# str_reverse()

# l = [1,2,3,'a','b','c',1,2,'a','b',3,'c','d','a','b',1]
# set1 = set(l)
# result = [(item, l.count(item)) for item in set1]
# result.sort(key=lambda x:x[1], reverse=True)
# print(result)

# list1 = [1,2,3,4,5,6,7,8]
# set1 = set(list1)
# print(set1)
# target = 6
# result = []
# for a in list1:
#     #print(a)
#     b = target - a
#     if a < b < target and b in set1:
#         result.append((list1.index(a), list1.index(b)))
#
# print(result)

# res = [i for i in map(lambda x: x**2,[1,2,3,4,5]) if i > 10]
# print(res)

# s = 'ajldjlafdljfddd'
# print(''.join(sorted(list(set(s)))))

# count = 0
# for num in range(100, 10001):
#     num_str = str(num)
#     #print(num_str)
#     sum = 0
#     for item in list(num_str):
#       # print(item)
#         sum += int(item)
#       # print(sum)
#     #print(sum)
#     if sum % 15 == 0:
#         if count and count % 10 == 0:
#             print()
#         print(num, end=" ")
#         count += 1
# #
# for i in range(2000, 2501):
#     if i % 4 == 0 and i % 100 !=0 or i % 400 == 0:
#         print(i)
# a = 3
# b =4
# s = lambda a,b:a*b
# print(s)
# a = [1,2,3,4,5,6,7,8,9]
# #ls = [i for i in filter(lambda x: x%2 ==1, a)]
# ls = [i for i in a if i % 2 == 1]
# print(ls)

def str_to_int(s):
    if not isinstance(s, str):
        return "您传入的不是字符串"
    if not s.isnumeric():
        return "您传入的非存数字"

    print(str_to_int("1234"))
    print(str_to_int("12ewgfdfs34"))
    print(str_to_int(123))