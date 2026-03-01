with open('../datasets/extended_qa_dataset.txt', 'r') as f:
    lines = f.readlines()

    for line in lines:
        print(line)
        if 'ans:' in line.split():
            print(True)
        else:
            print(False)
            break