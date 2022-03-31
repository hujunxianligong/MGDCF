cd ../../../
python -u gdcf_run_unified.py  \
 HomoMGDCF --dataset light_gcn_yelp \
        --gpu_ids 1 \
        --emb_size 64 \
        --lr 5e-3 \
        --lr_decay 0.95 \
        --z_l2_coef 1e-4 \
        --num_negs 300 \
        --batch_size 8000 \
        --num_epochs 2000 \
        --adj_drop_rate 0.97 \
        --alpha 0.1 \
        --beta 0.9 \
        --num_iter 2 \
        --x_drop_rate 0.1 \
        --z_drop_rate 0.1 \
        --edge_drop_rate 0.5 \
        --output_dir results/gnn_speed