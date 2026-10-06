#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# train.py - 단일 전방 카메라 PilotNet 학습 (이미지 -> [angle, speed] 회귀)
#   데이터: /home/user/xycar_ws/dataset/<세션>/labels.csv + images/
#   사용:   python3 train.py --data /home/user/xycar_ws/dataset --epochs 40
#   출력:   e2e/model.pth (가중치) + e2e/norm.json (라벨 정규화 파라미터)
#   * 추론 노드도 같은 PilotNet 정의 + norm.json 으로 역정규화해 사용한다. *
#   * 전처리(상단크롭 -> 200x66 -> YUV)는 추론과 반드시 동일해야 한다. *
# ============================================================================

import os, csv, glob, json, argparse, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 전처리 규격 (추론과 동일): PilotNet 입력 크기 / 상단 40%(하늘) 크롭
IN_W, IN_H = 200, 66
CROP_TOP   = 0.40


def collect(root):
    # 데이터 수집: 모든 세션의 labels.csv 를 읽어 (이미지경로, angle, speed) 목록으로 합친다.
    samples = []
    for csv_path in glob.glob(os.path.join(root, '*', 'labels.csv')):
        base = os.path.dirname(csv_path)
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                p = os.path.join(base, 'images', row['image'])
                if os.path.exists(p):
                    samples.append((p, float(row['angle']), float(row['speed'])))
    return samples


# 학습용 데이터셋: 이미지를 추론과 동일하게 전처리하고, 라벨을 z-score 정규화해 반환.
class DriveSet(Dataset):
    def __init__(self, samples, norm, augment=False):
        self.samples = samples
        self.norm    = norm
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        # 이미지 전처리: 상단크롭 -> 200x66 리사이즈 -> YUV 변환 (추론과 동일)
        img = cv2.imread(path)
        h = img.shape[0]
        img = img[int(h * CROP_TOP):, :]
        img = cv2.resize(img, (IN_W, IN_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return img

    def __getitem__(self, i):
        # 한 샘플 구성: 이미지 로드 -> (옵션) 밝기 증강 -> 정규화 -> (입력텐서, [angle, speed] 정규화 라벨)
        #   좌우 플립은 좌회전 전용 코스라 정답이 깨지므로 사용하지 않는다.
        path, angle, speed = self.samples[i]
        img = self._load(path)
        a = angle
        if self.augment:
            if random.random() < 0.5:
                img = np.clip(img.astype(np.float32) *
                              random.uniform(0.6, 1.4), 0, 255).astype(np.uint8)
        x = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        an = (a     - self.norm['a_m']) / self.norm['a_s']
        sn = (speed - self.norm['s_m']) / self.norm['s_s']
        return torch.from_numpy(x), torch.tensor([an, sn], dtype=torch.float32)


# PilotNet: 합성곱 5층(특징 추출) + 완전연결 4층(판단) -> [angle, speed] 2개 출력.
#   추론 노드(e2e_drive/e2e_pure)와 완전히 동일한 구조여야 가중치가 호환된다.
class PilotNet(nn.Module):
    def __init__(self):
        # 신경망 계층 정의: Conv 5층 -> Flatten/Dropout -> FC 4층(마지막 2출력)
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        # 순전파: 이미지 -> 합성곱 특징 -> 완전연결 -> [angle, speed]
        return self.fc(self.conv(x))


def main():
    # 학습 진입점: 인자 파싱 -> 데이터 수집/정규화 -> 학습 루프 -> 최적 모델 저장

    # 하이퍼파라미터(데이터 경로, 출력 경로, 에폭/배치/학습률) 파싱
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/home/user/xycar_ws/dataset')
    ap.add_argument('--out',  default='/home/user/xycar_ws/e2e')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    # 데이터 수집 + 셔플(샘플이 너무 적으면 중단)
    os.makedirs(args.out, exist_ok=True)
    samples = collect(args.data)
    if len(samples) < 50:
        print('too few samples: {} (수집 먼저)'.format(len(samples)))
        return
    random.shuffle(samples)

    # 라벨 정규화 파라미터(평균/표준편차) 계산 후 norm.json 저장(추론 역정규화에 사용)
    angs = np.array([s[1] for s in samples], dtype=np.float32)
    sps  = np.array([s[2] for s in samples], dtype=np.float32)
    norm = {'a_m': float(angs.mean()), 'a_s': float(angs.std() + 1e-6),
            's_m': float(sps.mean()),  's_s': float(sps.std() + 1e-6)}
    json.dump(norm, open(os.path.join(args.out, 'norm.json'), 'w'), indent=2)
    print('samples={} norm={}'.format(len(samples), norm))

    # train/val 8:2 분할 + DataLoader 구성(학습셋만 밝기 증강)
    n_val = max(1, int(len(samples) * 0.2))
    val, train = samples[:n_val], samples[n_val:]
    tr_dl = DataLoader(DriveSet(train, norm, augment=True),
                       batch_size=args.bs, shuffle=True, num_workers=4)
    va_dl = DataLoader(DriveSet(val, norm, augment=False),
                       batch_size=args.bs, shuffle=False, num_workers=4)

    # 모델/옵티마이저/손실(MSE: 회귀) 준비
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = PilotNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    print('device={}'.format(dev))

    # 학습 곡선 기록 파일 준비
    hist_path = os.path.join(args.out, 'history.csv')
    hf = open(hist_path, 'w', newline='')
    hf.write('epoch,train_loss,val_loss\n')

    # 학습 루프: 매 에폭 학습 -> 검증 -> 검증오차 최저일 때만 model.pth 저장
    best = 1e9
    for ep in range(args.epochs):
        # 학습 단계
        net.train(); tl = 0.0
        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward(); opt.step()
            tl += loss.item() * len(x)
        tl /= len(train)

        # 검증 단계(기울기 계산 없음)
        net.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(dev), y.to(dev)
                vl += lossf(net(x), y).item() * len(x)
        vl /= len(val)

        # 최적(검증오차 최저) 모델 저장 + 로그/기록
        flag = ''
        if vl < best:
            best = vl
            torch.save(net.state_dict(), os.path.join(args.out, 'model.pth'))
            flag = ' *saved'
        print('ep {:3d}  train {:.4f}  val {:.4f}{}'.format(ep, tl, vl, flag))
        hf.write('{},{:.6f},{:.6f}\n'.format(ep, tl, vl)); hf.flush()

    hf.close()
    print('done. best val {:.4f} -> {}/model.pth'.format(best, args.out))
    print('history -> {}'.format(hist_path))


if __name__ == '__main__':
    main()
