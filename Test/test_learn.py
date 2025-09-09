s = ['a', 'b', 'e', 'd', 'e']
# print(sum(s, ""))  # This will raise TypeError: can only concatenate list (not "str") to list

tpl = (1, 12, [1, 2, 3], 'a', 'b', 'c')
print(tpl[2][1])
tpl[2][1] = 4
print(tpl[2][1])
print(tpl)

from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)

from collections import Counter
c = Counter(s)
print(c)

from collections import deque
d = deque()
d.append(1)
d.append(2)
d.append(3)
d.appendleft(0)
d.pop()
d.appendleft(2)
d.extend([4, 5])
d.extendleft([6, 7])
print(d)
d.insert(2, 8)
print(d)


from collections import defaultdict
dd = defaultdict(int)
dd['a'] += 1
dd['b'] += 2
dd['c'] += 3
dd['e'] *= 3

print(dd['e'])
listy = [1, 2, 3, 4, 5]
string_list = ['a', 'b', 'c', 'd', 'e']
f = map(lambda x: x*2, string_list)
f_int = map(lambda x: x**2, listy)
print(list(f))

def add(x,/, y):
    print(f'Adding {x} and {y}')
    return x + y

# add(y=2, x=3)  # This will work because of keyword arguments

# making argument positional
add(11, y=30)

#dynamic argument
def dynamic_args(*args, **kwargs):
    print(f'Positional arguments: {args}')
    print(f'Keyword arguments: {kwargs}')
dynamic_args(1, 2, 3, a=4, b=5)


# Higher order functions
def outer(x, y):
    def inner(a, b):
        return a + b + x + y
    return inner
print(type(outer(1, 2)))
inner_function = outer(1, 2)
# print(inner_function(1, 2))


#file handling
with open('test.txt', 'a+') as f:
    f.write('Hello, World!\n')
    f.write('This is a test file.\n')

import json
# data = {}
with open('test.json', 'r+') as f:
    data = json.load(f)
    print(f'data: {data}')
    for dt in data:
        print(f'dt: {data[dt]}')
        for ele in data[dt]:
            print(f'{ele.keys()}: {ele.values()}')
