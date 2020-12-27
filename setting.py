import logging.config
import os


# ========================= logger ================================
log_conf = './logger.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('malware')


# ============================ files ==============================

script_root = "" # need to set
data_root = "" # need to set 

design_matrix_path = os.path.join(data_root, "datahander/matrixes")
list_path = os.path.join(data_root, "data/reports/indexs")
split_file_tmp = os.path.join(data_root, "datahander/split_records/splited_ids_seed{}_split-0.6-0.2-0.2.json")


hin_path = os.path.join(script_root, "data")
pre_max = 25
feat_dim = 5000

model_path = os.path.join(script_root, "save_models")
embed_path = os.path.join(script_root, "visualation/embeddings")
report_file = os.path.join(script_root,"save_models","reports.csv")

cuda_device_id = 0
