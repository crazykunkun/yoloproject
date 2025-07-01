import logging
import yaml

from utils import time_it
from utils import setup_logger
from utils import (YOLOSERVER_ROOT,
                   CONFIGS_DIR,
                   MODEL_DIR,
                   CHECKPOINTS_DIR,
                   PRETRAINED_DIR,
                   DATA_DIR,

                   RAW_IMAGES_DIR,
                   ORIGINAL_ANNOTATIONS_DIR,

                   LOGS_DIR,
                   SCRIPTS_DIR,
                   RUNS_DIR
                   )

# 1
logger = setup_logger(
    base_path=LOGS_DIR,
    log_type="init_project",
    model_name=None,
    encoding="utf-8",
    log_level=logging.INFO,
    logger_name="Project Initialization Logger")


# 2
@time_it(name="初始化项目", logger_instance=logger)
def initialize_project():
    logger.info("项目初始化开始".center(60, "="))
    logger.info(f"当前项目的根目录: {YOLOSERVER_ROOT.resolve()}")
    created_dirs = []
    existing_dirs = []
    raw_data_status = []
    stand_data_to_create = [
        CONFIGS_DIR,
        DATA_DIR,
        RUNS_DIR,
        MODEL_DIR,
        CHECKPOINTS_DIR,

        PRETRAINED_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        DATA_DIR / "train" / "images",
        DATA_DIR / "train" / "labels",
        DATA_DIR / "val" / "images",
        DATA_DIR / "val" / "labels",
        DATA_DIR / "test" / "images",
        DATA_DIR / "test" / "labels",
    ]
    logger.info("开始创建核心的目录".center(60, "="))
    for d in stand_data_to_create:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                created_dirs.append(d)
                logger.info(f"成功创建目录: {d}")
            except Exception as e:
                logger.error(f"创建目录失败: {d}, 错误信息: {e}")
        else:
            existing_dirs.append(d)
            logger.info(f"目录已存在: {d}")

    logger.info(f"项目核心目录已创建，创建了{len(created_dirs)}个新目录".center(60, "="))

    # 3
    logger.info(f"开始检查原始数据集目录".center(60, "="))
    raw_dirs_to_check = {
        "原始图像文件": RAW_IMAGES_DIR,
        "原始标注文件": ORIGINAL_ANNOTATIONS_DIR,
    }
    for desc, raw_dir in raw_dirs_to_check.items():
        if not raw_dir.exists():
            logger.warning(f"{desc}目录不存在: {raw_dir}")
        else:
            if not any(raw_dir.iterdir()):
                msg = f"{desc}目录为空: {raw_dir}"
                logger.warning(msg)
            else:
                logger.info(f"{desc}目录检查通过: {raw_dir}")

    # 4
    logger.info("项目初始化结果汇总".center(60, "="))
    if created_dirs:
        logger.info(f"成功创建了{len(created_dirs)}个目录")
        for dir_path in created_dirs:
            logger.info(f"- {dir_path}")
    else:
        logger.info("没有创建任何新目录，所有必要目录已存在。")

    if existing_dirs:
        logger.info(f"以下{len(existing_dirs)}个目录已存在，无需创建")
        for dir_path in existing_dirs:
            logger.info(f"- {dir_path}")

    if raw_data_status:
        logger.info("原始数据集目录检查结果".center(60, "="))
        for s in raw_data_status:
            logger.info(s)

        logger.info("原始数据集目录检查完成".center(60, "="))

    # 5 写入数据路径到 data.yaml
    data_yaml_path = CONFIGS_DIR / "data.yaml"
    data_paths = {
        "train_images": str(DATA_DIR / "train" / "images"),
        "train_labels": str(DATA_DIR / "train" / "labels"),
        "val_images": str(DATA_DIR / "val" / "images"),
        "val_labels": str(DATA_DIR / "val" / "labels"),
        "test_images": str(DATA_DIR / "test" / "images"),
        "test_labels": str(DATA_DIR / "test" / "labels"),
    }
    try:
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_paths, f, allow_unicode=True)
        logger.info(f"数据路径已写入: {data_yaml_path}")
    except Exception as e:
        logger.error(f"写入 data.yaml 失败: {e}")


if __name__ == "__main__":
    initialize_project()
