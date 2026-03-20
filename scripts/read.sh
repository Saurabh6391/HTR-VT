python3 train.py --exp-name read \
--max-lr 3e-4 \
--mask-ratio 0.2 \
--train-bs 64 \
--val-bs 8 \
--weight-decay 0.5 \
--attn-mask-ratio 0.1 \
--max-span-length 8 \
--img-size 512 64 \
--proj 8 \
--dila-ero-max-kernel 2 \
--dila-ero-iter 1 \
--proba 0.5 \
--alpha 1 \
--total-iter 100000 \
READ

python3 test.py --exp-name read \
--max-lr 3e-4 \
--mask-ratio 0.2 \
--train-bs 64 \
--val-bs 8 \
--weight-decay 0.5 \
--attn-mask-ratio 0.1 \
--max-span-length 8 \
--img-size 512 64 \
--proj 8 \
--dila-ero-max-kernel 2 \
--dila-ero-iter 1 \
--proba 0.5 \
--alpha 1 \
--total-iter 100000 \
READ

