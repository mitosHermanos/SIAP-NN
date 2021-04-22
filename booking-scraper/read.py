import csv



file = open("./booking.csv", "r")
reader = csv.reader(file)
i = 0
for line in reader:
    t=line[0]+","+line[1]+","+line[2]+","+line[3]+","+line[4]+","+line[5]+","+line[6]+","+line[7]

    i = i + 1
    print(t)
    # print(line[3])
    # parsed = line[3].replace("\xa0\n\n", "").strip()
    # # print(parsed)
    #
    # split_list = parsed.split("·")
    # print(split_list[0])
    # print(split_list)
    # print(split_list[1])
    if i == 2:
        break
