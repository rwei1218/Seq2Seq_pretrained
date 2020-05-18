def predata(oldpath, newpath):
    src = open(oldpath + '.src', 'r', encoding='utf8').readlines()
    tgt = open(oldpath + '.tgt.tagged', 'r', encoding='utf8').readlines()
    src = [line.strip() for line in src]
    tgt = [line.strip() for line in tgt]
    with open(newpath,  'w', encoding='utf8') as wtf:
        for src_line, tgt_line in zip(src, tgt):
            src_line = src_line.replace('-lrb- cnn -rrb- ', '').replace('</t>', '').replace('<t>', '').replace('  ', ' ')
            tgt_line = tgt_line.replace('-lrb- cnn -rrb- ', '').replace('</t>', '').replace('<t>', '').replace('  ', ' ')
            wtf.write(src_line + '\t' + tgt_line + '\n')
    print('end')

if __name__ == '__main__':
    oldpath = 'data/cnndm/train.txt'
    newpath = 'data/cnndm/train.tsv'
    predata(oldpath, newpath)
