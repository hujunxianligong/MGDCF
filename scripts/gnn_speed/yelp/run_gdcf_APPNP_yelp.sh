python -u gdcf_run_unified.py  \
       APPNP --dataset light_gcn_yelp \
       --gpu_ids 2 \
       --emb_size 64 \
       --lr 1e-2 \
       --lr_decay 0.995 \
       --z_l2_coef 1e-4 \
       --num_negs 1 \
       --batch_size 8000 \
       --num_epochs 1000 \
       --adj_drop_rate 0.97 \
       --alpha 0.1 \
       --beta 0.9 \
       --num_iter 4 \
       --x_drop_rate 0.3 \
       --z_drop_rate 0.1 \
       --edge_drop_rate 0.5 \
        --output_dir results/gnn_mf_speed_one_neg