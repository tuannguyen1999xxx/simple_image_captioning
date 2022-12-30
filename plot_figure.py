import json
import matplotlib.pyplot as plt

path_stats = './models_vgglstm/out_caption_results/log_train.txt'
data = []
with open(path_stats,'r') as r:
    for data_line in r.readlines():
        data.append(json.loads(data_line))
    epochs = [x['epoch'] for x in data]
    bleu4 = [x['val_Bleu_4'] for x in data]
    cider = [x['val_CIDEr'] for x in data]

    plt.title("BLEU-4 and CIDEr score")
    plt.twinx()
    plt.plot(epochs,bleu4,'-*',label='BLEU-4 score')
    plt.plot(epochs, cider, '-^',label='CIDEr score')

    plt.ylabel('BLEU-4')
    plt.ylabel('CIDEr')
    plt.legend()
    plt.savefig('images/bleu4.jpg')
    plt.close()
    print(epochs)
    print(bleu4)
    print(cider)