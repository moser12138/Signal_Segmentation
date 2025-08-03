"""
代码功能：将lst文件中的数据列表按照信号个数不同，信噪比snr不同，分到不同的lst文件中（训练数据只保留一半）
"""
import os

def split_lst_by_category_and_snr(lst_file, output_dir, prefix, save_half):
    with open(lst_file, 'r') as f:
        lines = f.readlines()

    categorized_files = {}

    for line in lines:
        image_path, label_path = line.strip().split()
        parts = image_path.split('/')  # 解析路径
        category = parts[2]  # 数据个数：1, 2, 3, 4, 5
        snr = parts[4]  # SNR 值：-5, 0, 5, 10, 15, 20

        key = f"{prefix}_{category}_{snr}.lst"
        if key not in categorized_files:
            categorized_files[key] = []

        categorized_files[key].append(line.strip())

    # 保存每个分类的 lst 文件
    os.makedirs(output_dir, exist_ok=True)
    for key, file_lines in categorized_files.items():
        if save_half:
            file_lines = file_lines[:len(file_lines) // 2]  # 只保留前一半数据

        output_file = os.path.join(output_dir, key)
        with open(output_file, 'w') as f:
            f.writelines("\n".join(file_lines))
        print(f"Saved {output_file} with {len(file_lines)} entries.")

# 主函数
def main():
    split_lst_by_category_and_snr('../data/list/train.lst', '../data/all_list2', 'train', save_half=True)
    split_lst_by_category_and_snr('../data/list/validation.lst', '../data/all_list2', 'validation', save_half=False)

if __name__ == "__main__":
    main()
