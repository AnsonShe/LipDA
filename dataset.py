# 统计真假视频的数量，生成list.json     第一步
import os
import json
import argparse
from pathlib import Path

class VideoListBuilder:
    """
    扫描 {root_dir}/
        0_real/         -> label 0
        1_fake/*/       -> label 1
    输出 JSON: [{'video':..., 'label':..., 'name':...}, ...]
    """
    VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')

    def __init__(self, root_dir, out_json):
        self.root = Path(root_dir)
        self.out  = Path(out_json)

    def build(self):
        items = []
        # -------- real --------
        # real_dir = self.root / 'Celeb-real'    # celebDF
        real_dir = self.root / '0_real'    # talkheadbench
        # real_dir = self.root / 'real'    # Ours
        # real_dir = self.root / '0_real'    # AVLip
        for p in sorted(real_dir.rglob('*')):
            if p.is_file() and p.suffix.lower() in self.VIDEO_EXTS:
                items.append({
                    'video': str(p),
                    'label': 0,
                    'name':  p.stem
                })

        # -------- fake --------   # Ours
        fake_dir = self.root / '1_fake'
        for model_dir in sorted(fake_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for p in sorted(model_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in self.VIDEO_EXTS:
                    items.append({
                        'video': str(p),
                        'label': 1,
                        'name':  f"{model_dir.name}_{p.stem}"
                    })
        
        # # AVLip、 celebDF
        # # fake_dir = self.root / '1_fake'    # AVLip
        # fake_dir = self.root / 'Celeb-synthesis'    # celebDF
        # # 不再遍历子目录，直接把 1_fake 当成“模型目录”
        # for p in sorted(fake_dir.rglob('*')):
        #     if p.is_file() and p.suffix.lower() in self.VIDEO_EXTS:
        #         items.append({
        #             'video': str(p),
        #             'label': 1,
        #             'name':  p.stem          # 如果还想保留“模型名”前缀，可改成 '1_fake_' + p.stem
        #         })

        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open('w') as f:
            json.dump(items, f, indent=2)
        print(f"Saved {len(items)} entries to {self.out}")
        return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build video list JSON from raw dataset')
    parser.add_argument('--root_dir', type=str, default='avlip/AVLips',
                        help='Root directory containing 0_real/ and 1_fake/')
    parser.add_argument('--out_json', type=str, default='lists.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    items = VideoListBuilder(args.root_dir, args.out_json).build()

    real_cnt = sum(1 for x in items if x['label'] == 0)
    fake_cnt = len(items) - real_cnt
    print(f"real: {real_cnt}  fake: {fake_cnt}")

    # 打印前 3 条做验证
    for it in items[:3]:
        print(it)