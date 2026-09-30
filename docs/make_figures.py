#!/usr/bin/env python3
"""Generate the branch pipeline figures used by AGENTS.md.

    python docs/make_figures.py        # writes docs/img/*.svg

Hand-rolled SVG on purpose: no dependencies, diffs readably in git, and the
text is real text so it stays selectable and scales cleanly. Each figure paints
its own light background so it is legible on a dark page too (GitHub renders
markdown SVG inside <img>, which does not inherit the page theme).
"""
from __future__ import annotations

import pathlib
import unicodedata

OUT = pathlib.Path(__file__).resolve().parent / 'img'

W = 900                      # canvas width
PAD = 24
BOX_H = 46                   # default block height
GAP = 26                     # vertical gap between blocks (arrow lives here)

BG = '#fbfbfa'
INK = '#111827'
MUTED = '#4b5563'
LINE = '#9ca3af'

# One colour per module kind, shared across all four figures.
SRC_FILL, SRC_STROKE = '#dbeafe', '#2563eb'      # source view (입력 카메라)
NOVEL_FILL, NOVEL_STROKE = '#ffffff', '#6b7280'  # novel view (감독 대상)
DEPTH_FILL, DEPTH_STROKE = '#fef3c7', '#d97706'  # depth 추정
GS_FILL, GS_STROKE = '#f3f4f6', '#374151'        # feature / GSRegresser
MERGE_FILL, MERGE_STROKE = '#ede9fe', '#7c3aed'  # Gaussian 병합 · view 선택
LOSS_FILL, LOSS_STROKE = '#dcfce7', '#16a34a'    # loss


def tw(s: str, size: float) -> float:
    """Rough text width: Hangul/CJK ~1.0em, Latin ~0.55em."""
    w = 0.0
    for ch in s:
        w += 1.0 if unicodedata.east_asian_width(ch) in 'WF' else 0.55
    return w * size


class Fig:
    def __init__(self, title: str):
        self.el: list[str] = []
        self.y = PAD
        self.title = title
        self._text(W / 2, self.y + 16, title, 16, INK, weight='600')
        self.y += 40

    # ---------------- primitives ----------------
    def _text(self, x, y, s, size=13, fill=INK, weight='400', anchor='middle',
              family='-apple-system, "Segoe UI", "Noto Sans KR", sans-serif'):
        self.el.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{family}\' '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{_esc(s)}</text>')

    def _rect(self, x, y, w, h, fill, stroke, dash=False, rx=6):
        d = ' stroke-dasharray="5 4"' if dash else ''
        self.el.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{d}/>')

    def _line(self, x1, y1, x2, y2, head=True):
        m = ' marker-end="url(#a)"' if head else ''
        self.el.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{LINE}" stroke-width="1.4"{m}/>')

    # ---------------- building blocks ----------------
    def block(self, lines, fill=GS_FILL, stroke=GS_STROKE, width=None,
              dash=False, sizes=None):
        """One centred stage box. `lines` may be a str or a list of lines."""
        lines = [lines] if isinstance(lines, str) else lines
        sizes = sizes or ([13] + [11.5] * (len(lines) - 1))
        need = max(tw(l, s) for l, s in zip(lines, sizes)) + 44
        w = width or max(need, 260)
        h = max(BOX_H, 20 + 17 * len(lines))
        x = (W - w) / 2
        self._rect(x, self.y, w, h, fill, stroke, dash=dash)
        ty = self.y + h / 2 - 17 * (len(lines) - 1) / 2 + 5
        for l, s in zip(lines, sizes):
            self._text(W / 2, ty, l, s, INK if s >= 13 else MUTED)
            ty += 17
        self.y += h
        return h

    def row(self, items, fill=SRC_FILL, stroke=SRC_STROKE, dash=False, bw=170):
        """A horizontal row of small boxes; each item is a list of lines."""
        n = len(items)
        gap = 16
        total = n * bw + (n - 1) * gap
        x0 = (W - total) / 2
        h = 20 + 16 * max(len(i) for i in items)
        for k, lines in enumerate(items):
            x = x0 + k * (bw + gap)
            self._rect(x, self.y, bw, h, fill, stroke, dash=dash)
            ty = self.y + h / 2 - 16 * (len(lines) - 1) / 2 + 4
            for j, l in enumerate(lines):
                self._text(x + bw / 2, ty, l, 12.5 if j == 0 else 11,
                           INK if j == 0 else MUTED,
                           weight='600' if j == 0 else '400')
                ty += 16
        self.y += h
        return [(x0 + k * (bw + gap) + bw / 2) for k in range(n)]

    def arrow(self, label=None):
        y1, y2 = self.y, self.y + GAP
        self._line(W / 2, y1 + 3, W / 2, y2 - 3)
        if label:
            self._text(W / 2 + 10, (y1 + y2) / 2 + 4, label, 11, MUTED,
                       anchor='start')
        self.y = y2

    def caption(self, s):
        self.y += 6
        self._text(W / 2, self.y + 12, s, 11.5, MUTED)
        self.y += 20

    def fan(self, centres, to_y):
        """Converging/diverging lines from the figure centre to several columns."""
        mid = self.y + GAP / 2
        self._line(W / 2, self.y + 3, W / 2, mid, head=False)
        self.el.append(
            f'<line x1="{min(centres):.1f}" y1="{mid:.1f}" '
            f'x2="{max(centres):.1f}" y2="{mid:.1f}" stroke="{LINE}" stroke-width="1.4"/>')
        for cx in centres:
            self._line(cx, mid, cx, to_y - 3)
        self.y = to_y

    def merge(self, centres):
        """Several columns merging back to the centre."""
        mid = self.y + GAP / 2
        for cx in centres:
            self._line(cx, self.y + 3, cx, mid, head=False)
        self.el.append(
            f'<line x1="{min(centres):.1f}" y1="{mid:.1f}" '
            f'x2="{max(centres):.1f}" y2="{mid:.1f}" stroke="{LINE}" stroke-width="1.4"/>')
        self._line(W / 2, mid, W / 2, mid + GAP / 2 - 3)
        self.y = mid + GAP / 2

    def save(self, name):
        h = self.y + PAD
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h:.0f}" '
            f'viewBox="0 0 {W} {h:.0f}" role="img" aria-label="{_esc(self.title)}">'
            f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker></defs>'
            f'<rect width="{W}" height="{h:.0f}" fill="{BG}"/>')
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(head + ''.join(self.el) + '</svg>', encoding='utf-8')
        print(f'{name}  ({W}x{h:.0f})')


def _esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;'))


# --------------------------------------------------------------------------- #
def gps_gs():
    f = Fig('gps_gs — 공식 GPS-Gaussian+ 베이스라인')
    f.row([['lmain', 'camera 22139908', 's1 폴더 / 0.png'],
           ['rmain', 'camera 22139909', 's1 폴더 / 1.png']], bw=210)
    f.caption('source 2 view · rectified 데이터셋')
    f.arrow()
    f.block('UnetExtractor')
    f.arrow()
    f.block(['LoFTR', '같은 row 안에서만 cross-attention → rectified 입력이 필요한 이유'],
            DEPTH_FILL, DEPTH_STROKE)
    f.arrow()
    f.block(['RAFT-Stereo (3 iters) → disparity', 'depth = −disparity / Tf_x'],
            DEPTH_FILL, DEPTH_STROKE)
    f.arrow()
    f.block(['GSRegresser', 'rot / scale / opacity / Δdepth'])
    f.arrow()
    f.block('source 2개의 Gaussian 전부 병합', MERGE_FILL, MERGE_STROKE)
    f.arrow()
    f.row([['novel view 1개', 'train_novel_id [2,3,4,5] 중 무작위']],
          NOVEL_FILL, NOVEL_STROKE, dash=True, bw=330)
    f.caption('3DGS 래스터화 → 이 view 로 렌더한 뒤 GT 와 비교')
    f.arrow()
    f.block(['loss = 0.8·L1 + 0.2·(1−SSIM) + 0.5·chamfer',
             'chamfer — 두 source view point cloud 사이, 최인접 점 간 거리를 최소화'],
            LOSS_FILL, LOSS_STROKE)
    f.save('gps_gs.svg')


def dav3_2view():
    f = Fig('dav3_2view — RAFT-Stereo 를 Depth-Anything-3 로 교체')
    f.row([['lmain', 'camera 22139908', 's1 폴더 / 0.png'],
           ['rmain', 'camera 22139909', 's1 폴더 / 1.png']], bw=210)
    f.caption('source 2 view · 비정렬 데이터셋 (LoFTR 미사용이므로 rectification 불필요)')
    f.arrow()
    f.block(['DA3-Small backbone',
             'backbone 파라미터는 동결 — LoRA r8 / α16 (qkv, proj) 을 붙여 LoRA 만 학습'],
            DEPTH_FILL, DEPTH_STROKE)
    f.arrow()
    f.block(['UpsamplerV2', '기존 DAv3 의 DPT head 대신 적용'],
            DEPTH_FILL, DEPTH_STROKE)
    f.arrow()
    f.block(['GSRegresser', 'rot / scale / opacity / Δdepth'])
    f.arrow()
    f.block('source 2개의 Gaussian 전부 병합', MERGE_FILL, MERGE_STROKE)
    f.arrow()
    f.row([['novel view 1개', 'train_novel_id [2,3,4,5] 중 무작위']],
          NOVEL_FILL, NOVEL_STROKE, dash=True, bw=330)
    f.arrow()
    f.block('loss = 0.8·L1 + 0.2·(1−SSIM)   ·   chamfer 없음',
            LOSS_FILL, LOSS_STROKE)
    f.save('dav3_2view.svg')


def dav3_4view():
    f = Fig('dav3_4view — 카메라 4대 모두 입력, 가장 가까운 2대로 렌더')
    cx = f.row([['view0', 'camera 22139908', 's1 폴더 / 0.png'],
                ['view1', 'camera 22139909', 's1 폴더 / 1.png'],
                ['view2', 'camera 22139914', 's2 폴더 / 1.png'],
                ['view3', 'camera 22139906', 's3 폴더 / 1.png']], bw=205)
    f.caption('source 4 view · mv_source_chain [[1,0],[1,1],[2,1],[3,1]] 이 모으는 4대')
    f.merge(cx)
    f.block(['DA3-Small backbone — 4 view 를 한 번에 통과 (1회 실행)',
             'DAv3 backbone 의 multi-view attention 이 네 뷰 간 관계를 처리'],
            DEPTH_FILL, DEPTH_STROKE)
    f.arrow()
    f.block(['view 마다 따로: UnetExtractor → UpsamplerV2 → GSRegresser',
             '→ Gaussian 4 세트'])
    f.arrow()
    f.block(['select_nearest_views(k = 2)',
             '4 세트를 다 합치면 먼 시점에서 투영된 Gaussian 이 대부분이므로,',
             'novel 카메라에 가장 가까운 2 세트만 병합'],
            MERGE_FILL, MERGE_STROKE)
    f.arrow()
    f.row([['novel view 1개', 'nearest-2 로 렌더', 'novel_per_segment: False']],
          NOVEL_FILL, NOVEL_STROKE, dash=True, bw=360)
    f.arrow()
    f.block('loss = 0.8·L1 + 0.2·(1−SSIM)', LOSS_FILL, LOSS_STROKE)
    f.save('dav3_4view.svg')


def dav3_4view_mvs():
    f = Fig('dav3_4view_with_multiview_supervision — segment 마다 novel view 감독')
    cx = f.row([['view0', 'camera 22139908', 's1 폴더 / 0.png'],
                ['view1', 'camera 22139909', 's1 폴더 / 1.png'],
                ['view2', 'camera 22139914', 's2 폴더 / 1.png'],
                ['view3', 'camera 22139906', 's3 폴더 / 1.png']], bw=205)
    f.caption('source 4 view · dav3_4view 와 완전히 동일')
    f.merge(cx)
    f.block('DA3-Small backbone — 4 view 를 한 번에 통과 (1회 실행)',
            DEPTH_FILL, DEPTH_STROKE)
    f.arrow()
    f.block(['view 마다 GSRegresser → Gaussian 4 세트',
             'Gaussian 회귀는 step 당 1회'],
            GS_FILL, GS_STROKE)
    y_after = f.y + GAP
    f.fan([W / 2 - 250, W / 2, W / 2 + 250], y_after)
    nov = f.row([['novel view @ s1', 'nearest-2 로 렌더', 'novel_per_segment: True'],
                 ['novel view @ s2', 'nearest-2 로 렌더', 'novel_per_segment: True'],
                 ['novel view @ s3', 'nearest-2 로 렌더', 'novel_per_segment: True']],
                NOVEL_FILL, NOVEL_STROKE, dash=True, bw=240)
    f.merge(nov)
    f.block(['loss = ( 3 view 의 0.8·L1 + 0.2·(1−SSIM) ) / 3'],
            LOSS_FILL, LOSS_STROKE)
    f.save('dav3_4view_with_multiview_supervision.svg')


if __name__ == '__main__':
    gps_gs()
    dav3_2view()
    dav3_4view()
    dav3_4view_mvs()
