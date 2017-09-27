import pickle
from collections import Counter
import csv

# verbatim_list = pickle.load(open('verbatims.pkl', 'rb'))
#
# out_dict = {'GENERAL CRASH': ['crash', 'freez'],
#             'FAILED INITIALIZE': ['fail', 'initializ', 'network', 'connect', 'black', 'inicia', 'internet', 'blank'],
#             'LOBBY': ['lobb', 'match', 'server', 'wait'],
#             'SPECTATOR': ['spectat'],
#             'CONTROLLER': ['control', 'move', 'input', 'button']}
# output_counter = Counter()
# other_list = []
#
# for v in verbatim_list:
#     found_toggle = 0
#     for title, word_list in out_dict.items():
#         if any([f in v.lower() for f in word_list]):
#             output_counter[title] += 1
#             found_toggle = 1
#     if not found_toggle:
#         output_counter['OTHER'] += 1
#         other_list.append(v)
#
# print(output_counter)
# for o in other_list:
#     print(o)
#
# outfile = open('verbatim_list.txt', 'w', encoding='utf-8')
# for e, z in enumerate(verbatim_list):
#     outfile.write('(' + str(e) + ') ' + z)

siuf_file = 'WinGamesSIUF.csv'
file_count = 0
verbatim_list = []
outfile = open('siuf_list.txt', 'w', encoding='utf-8')
with open(siuf_file, encoding='utf-8') as f:
    reader = csv.reader(f)
    for p in reader:
        file_count += 1
        if file_count == 1:
            print(p)
            continue
        if p[12] == '':
            verbatim_list.append(p[11])
        else:
            verbatim_list.append(p[12])
        outfile.write(verbatim_list[-1] + '\n')
print(file_count)
