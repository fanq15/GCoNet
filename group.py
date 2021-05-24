group = []
cnt = 0
epoch = -1
with open('group.txt', 'r') as f:
    for item in f.readlines():
        if cnt % 291 == 0:
            epoch += 1
        cnt += 1
        item = str(epoch) + ':' + item
        print(item)
        with open('./group_dir/group_' + str(epoch) + '.txt', 'a+') as f_new:
            f_new.write(item)
