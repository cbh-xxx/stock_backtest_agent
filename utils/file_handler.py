import os
import hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import sys
# 添加项目根目录到 Python 路径，解决直接运行时的导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Agent 项目目录
sys.path.insert(0, project_root)
from utils.logger_handler import logger


def get_file_md5_hex(filepath: str):     # 获取文件的md5的十六进制字符串

    if not os.path.exists(filepath):     # exists()判断路径是否存在
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):     # isfile()判断是否是文件
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096       # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:     # 必须二进制读取，使用"rb"参数
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            “
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            ”相当于一个while循环：
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):        # 返回文件夹内的文件列表（允许的文件后缀）
    files = []

    if not os.path.isdir(path):   # isdir()方法判断是不是文件夹
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):    # listdir()返回文件夹中文件的列表
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()