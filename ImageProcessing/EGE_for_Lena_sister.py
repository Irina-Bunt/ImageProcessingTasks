n = '6666666'
max = 0

for i in range(len(n)):
    max = max + int(n[i]) * (7**i)
    print(max)

x = 0
for i in range(15, (max + 1), 16):
    c = ''
    while i > 0:
        i1 = i % 5
        i2 = i // 5
        i = i2
        c = str(i1) + c

    sum = 0
    d = str(c)

    for j in range(len(d)):
        if d.count(d[j]) % 2 != 0:
            sum = sum + 1
    if sum <= 1:
        x = x + 1

print(x)