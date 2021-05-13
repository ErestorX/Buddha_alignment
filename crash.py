import os
import shutil

gt_files = [f for f in os.listdir("/home/hlemarchant/buddha_allign_report/")]
eval_files = [f for f in os.listdir("logs/pipeline2/Eval/")]

for eva in eval_files:
    for gt in gt_files:
        eva_id = eva.split("_")[1]
        gt_id = gt.split("_")[1]
        if eva_id == gt_id:
            shutil.copy("logs/pipeline2/Eval/" + eva, "/home/hlemarchant/report")
            shutil.copy("/home/hlemarchant/buddha_allign_report/" + gt, "/home/hlemarchant/report")
