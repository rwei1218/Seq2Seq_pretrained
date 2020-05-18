

def clean_infer(text_list):
    final_list = []
    for word in text_list:
        if word != '[CLS]':
            if word != '[SEP]':
                final_list.append(word)
            else:
                break
    return final_list
                
def clean_ref(text_list):
    final_list = []
    for word in text_list:
        if word != '[CLS]':
            if word != '[SEP]':
                final_list.append(word)
            else:
                break
    return final_list


def infer_text_clean(text_path, new_path):
    data = list(open(text_path, 'r', encoding='utf8').readlines())
    data = [line.strip().split('\t') for line in data]
    with open(new_path, 'w', encoding='utf8') as wtf:
        for index, line in enumerate(data, start=1):
            if len(line) != 2:
                print('error')
            else:
                infer = clean_infer(line[0].split())
                ref = clean_ref(line[1].split())
                wtf.write(str(index))
                wtf.write('\t')
                wtf.write(' '.join(infer))
                wtf.write('\t')
                wtf.write(' '.join(ref))
                wtf.write('\n')
    print('end')


def text_clean(text_path, new_path):
    data = list(open(text_path, 'r', encoding='utf8').readlines())
    data = [line.strip().split('\t') for line in data]
    with open(new_path, 'w', encoding='utf8') as wtf:
        for index, line in enumerate(data, start=1):
            if len(line) != 2:
                print('error')
            else:
                infer = clean_infer(line[0].split())
                # ref = clean_ref(line[1].split())
                wtf.write(str(index))
                wtf.write('\t')
                wtf.write(' '.join(infer))
                # wtf.write('\t')
                # wtf.write(' '.join(ref))
                wtf.write('\n')
    print('end')

def get_ref(text_path, new_path):
    data = list(open(text_path, 'r', encoding='utf8').readlines())
    data = [line.strip().split('\t') for line in data]
    with open(new_path, 'w', encoding='utf8') as wtf:
        for index, line in enumerate(data, start=1):
            if len(line) != 2:
                print('error')
            else:
                # infer = clean_infer(line[0].split())
                ref = clean_ref(line[1].split())
                wtf.write(str(index))
                # wtf.write('\t')
                # wtf.write(' '.join(infer))
                wtf.write('\t')
                wtf.write(' '.join(ref))
                wtf.write('\n')
    print('end')



if __name__ == '__main__':

    for i in [0, 1, 2, 3, 4]:
        text_path = 'output/giga/bert_transformer/'+ str(i) + '_infer_results.txt'
        new_path = 'output/giga/bert_transformer/processed_' + str(i) + '_infer_results.txt'
        text_clean(text_path, new_path)
    # text_path = 'output/giga/seq2seq_gru/'+ str(3) + '_infer_results.txt'
    # new_path = 'output/giga/seq2seq_ref.txt'
    # get_ref(text_path, new_path)
    # text_clean(text_path, new_path)
