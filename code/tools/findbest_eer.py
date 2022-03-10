import subprocess
import os
import sys
if not os.path.exists("./plot/s5a10_nds2"):
    os.mkdir("./plot/s5a10_nds2")

if not os.path.exists("./protocol_evalrst/s5a10_nds2"):
    os.mkdir("./protocol_evalrst/s5a10_nds2")

with open('plot/New_dataset3_shift5_alpha5.txt', 'w') as out:
    for i in range(20, 600, 10):
        return_code = subprocess.call('python protocols/protocol_2sessions.py --test_path ../../dataset/New_dataset2/L/ --out_path ./protocol_evalrst/s5a10_nds2/s5a10e{}_newds2_roc.npy --model_path ./protocol_ckpt/dppn_tlselect_ckpt_alpha_10_shift_5/ckpt_epoch_{}.pth --shift_size 5 --save_mmat True --nims 10'.format(i, i), stdout=out, shell=True)
        return_code = subprocess.call('python plot/calc_eer.py --src_npy ./protocol_evalrst/s5a10_nds2/s5a10e{}_newds2_roc.npy'.format(i), stdout=out, shell=True)
        sys.stdout.write("[*] {} / 600\r".format(i))
        sys.stdout.flush()
