out_file = open('data/PUBG_data_list3.txt', 'w')
count = 0
with open("data/PUBG_data_dump3.txt", encoding='utf-8') as f:
    for line in f:
        if 'full_text' in line:
            p = line.split('""')
            print(line)
            count += 1
            for word in p[3].split():
                if "@" not in word:  # and hashtags?
                    out_file.write(word + ' ')
            out_file.write("\n")

print(count)