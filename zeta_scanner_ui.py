"""
Riemann-Siegel Scanner UI
=========================
PyQt6 wrapper around zeta_gpu_scan_v6_hp.py. Launches the scanner as a
subprocess, reads its stdout in a background thread, parses the tagged
@@STATUS@@ JSON events into live status widgets, and mirrors the full
timer output into a scrollable text pane.

Buttons:
  Start     -- spawn the scanner subprocess with the chosen config
  Pause     -- write the pause-flag file; scanner exits cleanly at next
               chunk boundary (checkpoint already saved). UI updates state.
  Resume    -- delete the pause-flag file and relaunch; existing checkpoint
               takes over -- picks up exactly where it stopped.
  Abort     -- terminate the subprocess immediately. Whatever checkpoint was
               last saved (after the last completed chunk) survives -- no
               data loss for completed chunks, current in-flight chunk work
               is discarded (which is fine, it wasn't checkpointed anyway).

Design principles:
  * Subprocess isolation: the scanner is a separate OS process. Killing it
    NEVER wedges the UI (or your terminal, or your CUDA context, or your
    venv). This is the whole reason to use a subprocess vs a thread.
  * Zero coupling to scanner internals: we ONLY talk via the CLI flags and
    stdout. The scanner has no idea a UI exists; it could equally well be
    driven by a shell script or a future distributed coordinator.
  * Auto-scroll live output with a size cap so a multi-day run doesn't eat
    your RAM through the QTextEdit's undo history.
  * All state transitions are debounced and idempotent so double-clicks
    or racy sequences (Pause then Abort, etc.) don't corrupt state.

Layout (approximate):
  [ Configuration ]      -- T_BASE, N_CHUNKS, checkpoint prefix, etc.
  [ Start / Pause / Abort ]     -- state-aware button row
  [ Progress bar + chunk / rate / ETA ]
  [ Cumulative stats: zeros located, required, short, violations ]
  [ Tightest pair: t, norm_gap, gap, Z ]
  [ Live output pane (scrollable, auto-follow) ]
"""

import sys, os, json, time, subprocess, signal, re, math
from pathlib import Path
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from collections import deque

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QPointF, QRectF
from PyQt6.QtGui import (QFont, QTextCursor, QAction, QPainter, QColor, QPen,
                         QBrush, QRadialGradient, QPainterPath, QPixmap)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QProgressBar, QGroupBox, QFileDialog,
    QMessageBox, QStatusBar, QSplitter
)


# --------------------------------------------------------------------------
# Configuration constants (UI defaults; can be overridden in the UI itself)
# --------------------------------------------------------------------------
SCANNER_SCRIPT_DEFAULT = "zeta_gpu_scan_v7_hp.py"
STATUS_TAG = "@@STATUS@@ "                # must match _STATUS_TAG in scanner
SAMPLE_TAG = "@@SAMPLES@@ "              # must match _SAMPLE_TAG in scanner
ZERO_TAG   = "@@ZERO@@ "                # must match _ZERO_TAG in scanner
EMIT_FLAG_NAME = "emit.flag"             # credit flag the UI grants
BOOTSTRAP_FILE = "bootstrap.json"        # precomputed cold-start data
LIVE_OUTPUT_MAX_LINES = 10_000            # scrollback cap in the text pane
PAUSE_FLAG_NAME = "pause.flag"            # created/deleted in cwd

# QSettings identity. On Windows these become the registry path
# HKCU\Software\SetiAstro\ZetaSweepUI. On macOS a plist in ~/Library/
# Preferences. On Linux an ini under ~/.config/SetiAstro/. Kept as
# module-level so `main()` can set them on QApplication before any
# QSettings() constructor runs.
SETTINGS_ORG = "SetiAstro"
SETTINGS_APP = "ZetaSweepUI"


# --------------------------------------------------------------------------
# Live visualization (native Qt painting -- no QtWebEngine dependency)
# --------------------------------------------------------------------------
# A separate window animating the scanner's progress from the SAME
# @@STATUS@@ events the main UI parses. No scanner changes: it consumes
# run_start / chunk_end dicts via feed(). Three panels, one shared clock:
#   CENTER  RH frontier scatter -- every record-tight pair (min norm_gap)
#           as norm_gap vs t, with a staircase frontier line; the current
#           record glows and the panel flashes magenta on a new record.
#   BOTTOM  rate heartbeat -- rate_t_per_s scrolling, the machine's pulse.
#   LEFT    discovery ticker -- chunks dropping in with zero counts, the
#           cumulative total, and a magenta mark on each record break.
# Magenta is reserved strictly for record-breaking tight pairs, so it
# means the same thing every time it fires.
# --------------------------------------------------------------------------

_VIZ_VOID   = QColor(7, 11, 20)
_VIZ_GRID   = QColor(21, 32, 51)
_VIZ_TRACE  = QColor(255, 179, 71)     # phosphor amber
_VIZ_TRACED = QColor(138, 106, 58)
_VIZ_INK    = QColor(201, 214, 232)
_VIZ_INKDIM = QColor(95, 122, 156)
_VIZ_ALERT  = QColor(255, 61, 139)     # record-break magenta -- nowhere else
_VIZ_OKC    = QColor(63, 208, 160)     # rate trace teal
_VIZ_GRAMOK = QColor(80, 220, 130)     # Gram's law holds (green)
_VIZ_GRAMBAD= QColor(255, 90, 150)     # Gram's law violated (magenta-pink)


class ZetaVizWindow(QMainWindow):
    """Top-level live visualization window. Consumes the V7 sample stream:
      * feed_samples(block) -- a chunk's full-resolution Z(t) grid
      * feed_zero(ev)       -- one located zero (dt-offset + ordinal)
      * feed_status(ev)     -- run_start / chunk_end (for rate + record flash)
    and grants scanner credits (touches the emit-flag) when ready for the next
    buffer. Until the first live buffer arrives it loops precomputed bootstrap
    data so the window is never dead. grant_first_credit() should be called
    once the scanner is launched so the first chunk emits."""
    creditReady = pyqtSignal()           # ask the UI to touch the emit flag

    def __init__(self, parent=None, bootstrap_path=None):
        super().__init__(parent)
        self.setWindowTitle("zeta_sweep - live visualization")
        self.resize(1150, 720)
        self._canvas = _VizCanvas(self, bootstrap_path=bootstrap_path)
        self._canvas.creditReady.connect(self.creditReady)
        self.setCentralWidget(self._canvas)

    def feed_samples(self, block):  self._canvas.feed_samples(block)
    def feed_zero(self, ev):        self._canvas.feed_zero(ev)
    def feed_status(self, ev):      self._canvas.feed_status(ev)
    def reset(self):                self._canvas.reset()


class _VizCanvas(QWidget):
    """Plays one chunk-buffer of samples at a time as synchronized panels,
    then asks for the next. CENTER: the parametric zeta(1/2+it) spiral -- the
    curve loops the origin once per zero; tight pairs pinch near the origin.
    Drawn at full resolution so the fine winding survives. BOTTOM: S(t), the
    argument-count fluctuation, the real thing (NOT scan rate). LEFT: the
    discovery ticker of zeros with their global ordinals. Until the first live
    buffer arrives it loops precomputed bootstrap data (which carries re/im and
    s directly), so the correct visuals show during the cold start too."""
    creditReady = pyqtSignal()

    def __init__(self, parent=None, bootstrap_path=None):
        super().__init__(parent)
        self.setMinimumSize(720, 460)

        # buffer queue: each item carries the full arrays for one chunk:
        #   {"dt":[], "re":[], "im":[], "s":[] or None, "zeros":[], "base":,
        #    "chunk":, "bootstrap":bool}
        self._queue = deque()
        self._cur = None
        self._play_i = 0
        self._zero_ptr = 0
        self._pending_zeros = deque()

        # display state
        self.zeros_total = 0
        self.best_ngap = None
        self.session_records = 0
        self.ticker = deque(maxlen=80)
        self._flash_until = 0.0
        self._first_live_seen = False

        # parametric spiral trail: recent (re, im) points. Short so old data
        # fades quickly and the current loop reads clearly (1/4 of prior 4000).
        self._spiral = deque(maxlen=1000)
        # Z(t) rolling window: (dt_abs, z) for the bottom trace
        self._z_window = deque(maxlen=2400)

        # bootstrap data (looped until first live buffer)
        self._bootstrap = None
        if bootstrap_path:
            try:
                with open(bootstrap_path, "r") as f:
                    self._bootstrap = json.load(f)
            except Exception:
                self._bootstrap = None
        self._load_bootstrap_buffer()

        # Separate playback speeds (samples advanced per ~33ms tick). Bootstrap
        # stays lively; live plays slower so the dense 1e13 winding is watchable.
        # Lower = slower. These only affect draw pace, never the switching/credit
        # logic below.
        self._playspeed_bootstrap = 10
        self._playspeed_live = 4
        self._anim = QTimer(self)
        self._anim.timeout.connect(self._advance)
        self._anim.start(33)

    # ---- external feeds --------------------------------------------------
    def feed_status(self, ev):
        k = ev.get("event", "")
        if k == "chunk_end":
            zt = ev.get("zeros_located_total")
            if zt is not None:
                try: self.zeros_total = int(zt)
                except (TypeError, ValueError): pass
            tp = ev.get("tightest")
            if isinstance(tp, dict) and tp.get("norm_gap") is not None:
                ng = float(tp["norm_gap"])
                if self.best_ngap is None or ng < self.best_ngap:
                    self.best_ngap = ng
                    self.session_records += 1
                    self._flash_until = time.monotonic() + 0.8

    def feed_zero(self, ev):
        # Zeros are emitted AFTER their chunk's @@SAMPLES@@ block (append_zeros
        # runs after process_chunk), so by the time a zero arrives its buffer is
        # already queued or playing. Attach it to that buffer's zero list so it
        # shows in the ticker as playback reaches it. If the buffer isn't found
        # (zero arrived before its samples, or samples were dropped), stash it
        # in _pending_zeros and feed_samples will pick it up.
        ch = ev.get("chunk")
        # currently-playing buffer?
        if self._cur is not None and not self._cur.get("bootstrap") \
                and self._cur.get("chunk") == ch:
            self._cur.setdefault("zeros", []).append(ev)
            # keep the play-position's zero list sorted so _advance finds it
            self._cur["zeros"].sort(key=lambda z: z.get("dt", 0.0))
            return
        # a queued buffer?
        for buf in self._queue:
            if not buf.get("bootstrap") and buf.get("chunk") == ch:
                buf.setdefault("zeros", []).append(ev)
                return
        # not found yet -> stash for feed_samples
        self._pending_zeros.append(ev)

    def feed_samples(self, block):
        ch = block.get("chunk")
        # collect any zeros that arrived BEFORE this block (rare -- usually they
        # come after). Zeros that arrive after attach via feed_zero above.
        zeros = [z for z in self._pending_zeros if z.get("chunk") == ch]
        for z in list(self._pending_zeros):
            if z.get("chunk") == ch:
                self._pending_zeros.remove(z)
        # Use .get() -- a block may lack re/im (a fine-resweep Z with no theta
        # stash). We still enqueue it so live data flows; the spiral just won't
        # draw for that one block while Z(t) still does. A hard block["re"]
        # here would raise KeyError inside the Qt slot, silently drop the
        # buffer, and leave the viz stuck looping bootstrap forever.
        buf = {"dt": block.get("dt", []),
               "re": block.get("re", []), "im": block.get("im", []),
               "z": block.get("z", []), "base": float(block.get("base", 1e13)),
               "chunk": ch, "zeros": zeros, "bootstrap": False,
               "grams": block.get("grams", [])}
        if not self._first_live_seen:
            self._first_live_seen = True
            self._queue.clear(); self._cur = None
        self._queue.append(buf)

    def reset(self):
        self._queue.clear(); self._cur = None; self._play_i = 0
        self._first_live_seen = False
        self._spiral.clear(); self._z_window.clear()
        self._load_bootstrap_buffer()

    # ---- bootstrap -------------------------------------------------------
    def _load_bootstrap_buffer(self):
        if not self._bootstrap:
            return
        b = self._bootstrap
        buf = {"dt": b.get("dt", []), "re": b.get("re", []),
               "im": b.get("im", []), "z": b.get("z", []),
               "base": 0.0, "chunk": None,
               "zeros": b.get("zeros", []), "bootstrap": True,
               "grams": b.get("grams", [])}
        self._queue.append(buf)

    # ---- playback engine -------------------------------------------------
    def _start_next_buffer(self):
        if not self._queue:
            if not self._first_live_seen:
                self._load_bootstrap_buffer()
            if not self._queue:
                return
        self._cur = self._queue.popleft()
        self._play_i = 0
        self._zero_ptr = 0
        self._cur["zeros"] = sorted(self._cur.get("zeros", []),
                                    key=lambda z: z.get("dt", 0.0))
        # Build a per-sample Gram-compliance array from the buffer's gram marks.
        # Each gram = [sample_index, parity, holds]; a gram opens a segment that
        # runs until the next gram. compliance[i] in {1 holds, 0 violates, -1
        # unknown/before first gram}. Used to colour the spiral trail.
        grams = self._cur.get("grams", [])
        n = len(self._cur.get("dt", []))
        comp = [-1] * n
        if grams and n:
            for gi in range(len(grams)):
                start_idx = grams[gi][0]
                end_idx = grams[gi+1][0] if gi+1 < len(grams) else n
                holds = grams[gi][2]
                for i in range(max(0, start_idx), min(n, end_idx)):
                    comp[i] = holds
        self._cur["_comp"] = comp
        if not self._cur.get("bootstrap"):
            self.creditReady.emit()

    def _advance(self):
        if self._cur is None:
            self._start_next_buffer()
            if self._cur is None:
                self.update(); return
        dt = self._cur["dt"]; re = self._cur["re"]; im = self._cur["im"]
        zt_arr = self._cur.get("z") or []
        base = self._cur["base"]
        n = len(dt)
        # spiral needs re/im aligned; Z(t) needs z aligned. Draw whichever are
        # present -- a block missing re/im (rare fine-resweep) still shows Z(t).
        has_spiral = (len(re) == n and len(im) == n)
        has_z = (len(zt_arr) == n)
        if n == 0 or (not has_spiral and not has_z):
            self._cur = None; self.update(); return
        _speed = self._playspeed_bootstrap if self._cur.get("bootstrap") \
            else self._playspeed_live
        end = min(n, self._play_i + _speed)
        _comp = self._cur.get("_comp")
        for i in range(self._play_i, end):
            if has_spiral:
                cc = _comp[i] if (_comp and i < len(_comp)) else -1
                self._spiral.append((re[i], im[i], cc))
            if has_z:
                self._z_window.append((base + dt[i], zt_arr[i]))
        # zeros reached in this step -> ticker
        if self._play_i < end and dt:
            cur_dt = dt[min(end-1, n-1)]
            zs = self._cur["zeros"]
            while self._zero_ptr < len(zs) and \
                    zs[self._zero_ptr].get("dt", 1e18) <= cur_dt:
                zz = zs[self._zero_ptr]; self._zero_ptr += 1
                self._on_zero_reached(zz, base)
        self._play_i = end
        if self._play_i >= n:
            self._cur = None
        self.update()

    def _on_zero_reached(self, zz, base):
        dt_abs = base + zz.get("dt", 0.0)
        self.ticker.appendleft({"ord": zz.get("ord"), "dt": dt_abs,
                                "bootstrap": self._cur.get("bootstrap", False)})

    # ---- paint -----------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, _VIZ_VOID)
        lw = max(240, int(W * 0.22))
        bh = max(120, int(H * 0.22))
        self._paint_ticker(p, QRectF(0, 0, lw, H))
        self._paint_spiral(p, QRectF(lw, 0, W - lw, H - bh))
        self._paint_z(p, QRectF(lw, H - bh, W - lw, bh))
        p.setPen(QPen(_VIZ_GRID, 1))
        p.drawLine(lw, 0, lw, H); p.drawLine(lw, H - bh, W, H - bh)
        now = time.monotonic()
        if now < self._flash_until:
            a = (self._flash_until - now) / 0.8
            g = QRadialGradient(lw + (W-lw)*0.5, (H-bh)*0.45, (W-lw)*0.4)
            c = QColor(_VIZ_ALERT); c.setAlphaF(0.16*a)
            g.setColorAt(0, c); g.setColorAt(1, QColor(255,61,139,0))
            p.fillRect(QRectF(lw, 0, W-lw, H-bh), QBrush(g))
        p.end()

    def _paint_spiral(self, p, r):
        p.save(); p.setClipRect(r)
        boot = (self._cur.get("bootstrap") if self._cur else False)
        p.setFont(QFont("Inter", 8, QFont.Weight.DemiBold)); p.setPen(_VIZ_INKDIM)
        label = "zeta(1/2 + it)  -  parametric" + \
                ("   (bootstrap)" if boot else "")
        p.drawText(QRectF(r.left()+16, r.top()+10, r.width()-32, 18),
                   Qt.AlignmentFlag.AlignLeft, label)
        if len(self._spiral) < 2:
            p.setPen(_VIZ_INKDIM); p.setFont(QFont("JetBrains Mono", 10))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "warming up...")
            p.restore(); return
        pts = list(self._spiral)
        # autoscale to the recent trail, centered on origin (points are
        # (re, im, compliance); compliance in {1 holds, 0 violates, -1 unknown})
        mx = max(1e-6, max(abs(pt[0]) for pt in pts))
        my = max(1e-6, max(abs(pt[1]) for pt in pts))
        m = max(mx, my) * 1.1
        cx = r.center().x(); cy = r.center().y()
        R = min(r.width(), r.height()) * 0.42 / m
        # graticule: axes + origin
        p.setPen(QPen(_VIZ_GRID, 1))
        p.drawLine(QPointF(cx - r.width()*0.42, cy),
                   QPointF(cx + r.width()*0.42, cy))
        p.drawLine(QPointF(cx, r.top()+28), QPointF(cx, r.bottom()-8))
        p.setBrush(QBrush(_VIZ_INK)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), 2, 2)     # origin
        # axis scale ticks: a "nice" unit fitting the current autoscale, marked
        # on the Re and Im axes with labels, so the loop size is readable.
        def _nice_unit(v):
            import math as _m
            if v <= 0: return 1.0
            e = _m.floor(_m.log10(v)); b = 10**e; f = v/b
            return (1 if f < 2 else (2 if f < 5 else 5)) * b
        unit = _nice_unit(m * 0.75)
        p.setFont(QFont("JetBrains Mono", 7))
        for sgn in (1, -1):
            xt = cx + sgn*unit*R
            if abs(xt-cx) > 8 and r.left() < xt < r.right():
                p.setPen(QPen(_VIZ_GRID, 1))
                p.drawLine(QPointF(xt, cy-4), QPointF(xt, cy+4))
                p.setPen(_VIZ_INKDIM)
                p.drawText(QRectF(xt-24, cy+6, 48, 12),
                           Qt.AlignmentFlag.AlignCenter, f"{sgn*unit:g}")
            yt = cy - sgn*unit*R
            if abs(yt-cy) > 8 and r.top() < yt < r.bottom():
                p.setPen(QPen(_VIZ_GRID, 1))
                p.drawLine(QPointF(cx-4, yt), QPointF(cx+4, yt))
                p.setPen(_VIZ_INKDIM)
                p.drawText(QRectF(cx+7, yt-7, 40, 14),
                           Qt.AlignmentFlag.AlignLeft, f"{sgn*unit:g}i")
        # trail: colour each segment by Gram-law compliance of its sample.
        # green = Gram's law holds, magenta = violates, amber = unknown (pre-
        # first-gram or bootstrap without gram data). Fade older segments out.
        flashing = time.monotonic() < self._flash_until
        npts = len(pts)
        for i in range(1, npts):
            ax, ay = pts[i-1][0], pts[i-1][1]
            bx, by = pts[i][0], pts[i][1]
            comp = pts[i][2]
            age = i / npts
            if flashing:
                col = QColor(_VIZ_ALERT)
            elif comp == 0:
                col = QColor(_VIZ_GRAMBAD)   # violation
            elif comp == 1:
                col = QColor(_VIZ_GRAMOK)     # holds
            else:
                col = QColor(_VIZ_TRACE)      # unknown
            col.setAlphaF(0.15 + 0.8*age)
            p.setPen(QPen(col, 0.5 + 1.6*age))
            p.drawLine(QPointF(cx + ax*R, cy - ay*R),
                       QPointF(cx + bx*R, cy - by*R))
        # leading dot
        lx, ly = pts[-1][0], pts[-1][1]
        p.setBrush(QBrush(_VIZ_ALERT if flashing else QColor(255,217,160)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx + lx*R, cy - ly*R), 3.2, 3.2)
        # readouts: current |zeta| and the autoscale extent (loop size)
        p.setFont(QFont("JetBrains Mono", 9)); p.setPen(_VIZ_INKDIM)
        p.drawText(QRectF(r.right()-200, r.top()+10, 184, 16),
                   Qt.AlignmentFlag.AlignRight,
                   f"|zeta| {(lx*lx+ly*ly)**0.5:.4f}")
        p.setPen(_VIZ_TRACED)
        p.drawText(QRectF(r.right()-200, r.top()+26, 184, 14),
                   Qt.AlignmentFlag.AlignRight, f"view +/-{m:.3g}")
        p.restore()

    def _paint_z(self, p, r):
        p.save(); p.setClipRect(r)
        p.setFont(QFont("Inter", 8, QFont.Weight.DemiBold)); p.setPen(_VIZ_INKDIM)
        p.drawText(QRectF(r.left()+16, r.top()+8, r.width()-32, 16),
                   Qt.AlignmentFlag.AlignLeft, "Z(t)  -  Riemann-Siegel")
        if len(self._z_window) < 2:
            p.restore(); return
        pts = list(self._z_window)
        x0 = pts[0][0]; x1 = pts[-1][0]; span = max(1e-9, x1 - x0)
        zmax = max(2.0, max(abs(v) for _, v in pts) * 1.1)
        mid = r.center().y() + 6
        amp = (r.height() - 46) * 0.5
        # zero line -- every zeta zero is where the trace crosses this
        p.setPen(QPen(_VIZ_GRID, 1))
        p.drawLine(QPointF(r.left()+16, mid), QPointF(r.right()-16, mid))
        def X(t): return r.left()+16 + (t-x0)/span*(r.width()-32)
        def Y(v): return mid - (v/zmax)*amp
        flashing = time.monotonic() < self._flash_until
        p.setPen(QPen(_VIZ_ALERT if flashing else _VIZ_TRACE, 1.4))
        path = QPainterPath(); first = True
        for (t, v) in pts:
            xx, yy = X(t), Y(v)
            if first: path.moveTo(xx, yy); first=False
            else: path.lineTo(xx, yy)
        p.drawPath(path)
        # leading dot
        lt, lv = pts[-1]
        p.setBrush(QBrush(_VIZ_TRACE)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(X(lt), Y(lv)), 3.0, 3.0)
        # axis readouts: the t window shown, and the Z vertical scale
        p.setFont(QFont("JetBrains Mono", 7)); p.setPen(_VIZ_INKDIM)
        p.drawText(QRectF(r.left()+16, r.bottom()-13, r.width()*0.6, 12),
                   Qt.AlignmentFlag.AlignLeft,
                   f"t: {x0:.2f} -> {x1:.2f}  (dt {x1-x0:.2f})")
        p.drawText(QRectF(r.right()-120, r.top()+8, 104, 12),
                   Qt.AlignmentFlag.AlignRight, f"Z: +/-{zmax:.2g}")
        p.restore()

    def _paint_ticker(self, p, r):
        p.save(); p.setClipRect(r)
        p.setFont(QFont("Inter", 11, QFont.Weight.Bold)); p.setPen(_VIZ_INK)
        p.drawText(QRectF(r.left()+16, r.top()+16, r.width()-32, 20),
                   Qt.AlignmentFlag.AlignLeft, "zeta_sweep")
        p.setFont(QFont("JetBrains Mono", 8)); p.setPen(_VIZ_INKDIM)
        p.drawText(QRectF(r.left()+16, r.top()+40, r.width()-32, 16),
                   Qt.AlignmentFlag.AlignLeft,
                   f"zeros located   {self.zeros_total:,}")
        p.setPen(_VIZ_ALERT if self.session_records else _VIZ_INKDIM)
        p.drawText(QRectF(r.left()+16, r.top()+58, r.width()-32, 16),
                   Qt.AlignmentFlag.AlignLeft,
                   f"records broken  {self.session_records}")
        if self.best_ngap is not None:
            p.setPen(_VIZ_INKDIM)
            p.drawText(QRectF(r.left()+16, r.top()+76, r.width()-32, 16),
                       Qt.AlignmentFlag.AlignLeft,
                       f"tightest ngap   {self.best_ngap:.6f}")
        p.setPen(QPen(_VIZ_GRID, 1))
        p.drawLine(QPointF(r.left()+16, r.top()+100),
                   QPointF(r.right()-16, r.top()+100))
        y = r.top() + 114
        p.setFont(QFont("JetBrains Mono", 9))
        for row in self.ticker:
            if y > r.bottom() - 30: break
            boot = row.get("bootstrap")
            oid = row.get("ord")
            oid_s = f"#{oid:,}" if isinstance(oid, int) else "#—"
            p.setPen(_VIZ_INKDIM)
            p.drawText(QRectF(r.left()+16, y, 120, 16),
                       Qt.AlignmentFlag.AlignLeft, oid_s)
            p.setPen(_VIZ_TRACED if boot else _VIZ_TRACE)
            p.drawText(QRectF(r.left()+120, y, r.width()-130, 16),
                       Qt.AlignmentFlag.AlignLeft, f"{row.get('dt',0):.4f}")
            y += 19
        p.setFont(QFont("Inter", 8, QFont.Weight.DemiBold)); p.setPen(_VIZ_INKDIM)
        p.drawText(QRectF(r.left()+16, r.bottom()-24, r.width()-32, 16),
                   Qt.AlignmentFlag.AlignLeft, "SETIASTRO")
        p.restore()






# --------------------------------------------------------------------------
# Subprocess reader thread
# --------------------------------------------------------------------------
# Reads stdout line by line, emits two Qt signals:
#   * lineReceived(str)     -- every raw line (for the text pane mirror)
#   * statusReceived(dict)  -- parsed @@STATUS@@ event (for widget updates)
# The distinction matters: a UI update from a Qt signal is thread-safe and
# happens on the GUI thread automatically. Doing widget updates from the
# reader thread directly would race and eventually crash Qt.
# --------------------------------------------------------------------------

class ScannerReader(QThread):
    lineReceived = pyqtSignal(str)
    statusReceived = pyqtSignal(dict)
    sampleReceived = pyqtSignal(dict)    # @@SAMPLES@@ dense Z(t) block
    zeroReceived = pyqtSignal(dict)      # @@ZERO@@ per-zero event
    processExited = pyqtSignal(int)      # exit code

    def __init__(self, proc, parent=None):
        super().__init__(parent)
        self.proc = proc
        self._stop = False

    def run(self):
        # Drain stdout until the process closes it (i.e. process exits).
        # `text=True` on the Popen means we get str lines, not bytes.
        try:
            for line in self.proc.stdout:
                if self._stop:
                    break
                line = line.rstrip("\r\n")
                if line.startswith(STATUS_TAG):
                    try:
                        self.statusReceived.emit(
                            json.loads(line[len(STATUS_TAG):]))
                    except json.JSONDecodeError:
                        self.lineReceived.emit(line)
                elif line.startswith(SAMPLE_TAG):
                    # dense Z(t) block: route to the viz, do NOT mirror to the
                    # text pane (it's ~1.6 MB and would flood the scrollback).
                    try:
                        self.sampleReceived.emit(
                            json.loads(line[len(SAMPLE_TAG):]))
                    except json.JSONDecodeError:
                        pass
                elif line.startswith(ZERO_TAG):
                    try:
                        self.zeroReceived.emit(
                            json.loads(line[len(ZERO_TAG):]))
                    except json.JSONDecodeError:
                        pass
                else:
                    self.lineReceived.emit(line)
        except Exception as e:
            self.lineReceived.emit(f"[reader error] {e}")
        finally:
            self.proc.wait()
            self.processExited.emit(self.proc.returncode or 0)

    def stop(self):
        self._stop = True


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class ScannerUI(QMainWindow):
    # Scanner run states -- gate button enabling on these
    STATE_IDLE      = "idle"      # no subprocess; ready to start
    STATE_RUNNING   = "running"   # subprocess alive, doing work
    STATE_PAUSING   = "pausing"   # pause flag set; waiting for clean exit
    STATE_PAUSED    = "paused"    # subprocess exited cleanly at chunk boundary
    STATE_ABORTING  = "aborting"  # sent terminate(); waiting for exit

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Riemann-Siegel Zero Scanner")
        # Wider default now that layout is horizontal: left column ~480px
        # of status widgets, rest for the terminal. User can drag the
        # splitter to reallocate. Window+splitter geometry persist across
        # launches via QSettings.
        self.resize(1400, 780)

        self.state = self.STATE_IDLE
        self.proc = None
        self.reader = None
        self.run_start_time = None
        self.chunks_completed_this_run = 0    # completed since last Start
        self.first_chunk_this_run = 0         # from run_start event
        self.total_chunks_target = 0
        self.chunk_t_current = 0.0            # for ETA math
        self.pause_flag_path = None
        self.last_wall = 0.0

        self._build_ui()
        # Restore any config the user saved in a previous session BEFORE
        # setting state (state affects widget-enabled state, which is fine
        # to compute after values are populated).
        self._load_settings()

        # Check if a checkpoint from a prior session exists. If it does
        # and indicates progress (next_chunk > 0), the user probably wants
        # to Resume rather than Start (which would re-run from the
        # start-chunk override). We enter STATE_PAUSED so Resume is
        # available. This handles the common case of "closed the UI after
        # a pause, reopened later, want to continue."
        if self._detect_existing_checkpoint():
            self._set_state(self.STATE_PAUSED)
            self.statusBar().showMessage(
                "Checkpoint found from a prior session. "
                "Click Resume to continue, or Start to begin fresh "
                "(Start with start-chunk override will re-run from that chunk).")
        else:
            self._set_state(self.STATE_IDLE)

        # Elapsed-time ticker (updates the elapsed/ETA labels once a second
        # even between chunks, so the display doesn't look frozen during a
        # long chunk).
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_elapsed)
        self._tick.start()

    # ------------------------- UI construction -------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Top-level: a HORIZONTAL splitter. Left pane = all the status
        # widgets stacked vertically (config, buttons, progress, stats,
        # tightest pair). Right pane = the live output terminal.
        #
        # Why a splitter and not a fixed HBoxLayout: users have different
        # monitor widths and preferences. Splitter lets them drag the divider
        # to give more space to output vs status. Position is persisted via
        # QSettings so it's restored across launches.
        #
        # Vertical growth (config panel getting tall) becomes horizontal
        # (config panel and output pane share the width), so the whole UI
        # scales gracefully as more config fields get added over time.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Wrap the splitter in a top-level VBox on the central widget --
        # gives it proper edge margins and lets the status bar sit beneath.
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self.main_splitter)

        # Left column: status widgets stacked vertically
        left_widget = QWidget()
        left_col = QVBoxLayout(left_widget)
        left_col.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.addWidget(left_widget)

        # We build all the status groups into `left_col` (was `root` before).
        # The output group at the very end is added to the SPLITTER, not to
        # left_col, so it becomes the right pane.
        root = left_col   # local alias so the existing addWidget() calls
                          # below keep working without a mass rename

        # --- Configuration group ---
        # Each config row has an "override" checkbox on the right so the user
        # sees clearly which fields will be passed as CLI flags to the scanner
        # (checked = pass) vs left to the scanner's module default (unchecked).
        # This keeps the "no flags" default behavior obviously visible.
        cfg_group = QGroupBox("Configuration")
        cfg = QGridLayout(cfg_group)

        # Row 0: scanner script
        cfg.addWidget(QLabel("Scanner script:"), 0, 0)
        self.script_edit = QLineEdit(SCANNER_SCRIPT_DEFAULT)
        cfg.addWidget(self.script_edit, 0, 1, 1, 2)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_script)
        cfg.addWidget(browse, 0, 3)

        # Row 1: working directory
        cfg.addWidget(QLabel("Working directory:"), 1, 0)
        self.cwd_edit = QLineEdit(os.getcwd())
        cfg.addWidget(self.cwd_edit, 1, 1, 1, 2)
        cwd_browse = QPushButton("Browse…")
        cwd_browse.clicked.connect(self._browse_cwd)
        cfg.addWidget(cwd_browse, 1, 3)

        # Row 2: pause flag path
        cfg.addWidget(QLabel("Pause flag file:"), 2, 0)
        self.flag_edit = QLineEdit(PAUSE_FLAG_NAME)
        self.flag_edit.setToolTip(
            "Path to the pause-flag file. Pause button creates it; scanner "
            "detects it between chunks and exits cleanly.")
        cfg.addWidget(self.flag_edit, 2, 1, 1, 3)

        # Row 3: T_BASE override
        cfg.addWidget(QLabel("T_BASE (start height):"), 3, 0)
        self.tbase_edit = QLineEdit("1e13")
        self.tbase_edit.setToolTip(
            "Starting height on the critical line. Passed as --t-base when "
            "the checkbox is on. Uses the scanner's module default otherwise.")
        cfg.addWidget(self.tbase_edit, 3, 1)
        self.tbase_override = QCheckBox("Override")
        cfg.addWidget(self.tbase_override, 3, 2)

        # Row 4: N_CHUNKS override
        cfg.addWidget(QLabel("Total chunks (N_CHUNKS):"), 4, 0)
        self.nchunks_edit = QSpinBox()
        self.nchunks_edit.setRange(1, 10_000_000)
        self.nchunks_edit.setValue(100)
        self.nchunks_edit.setToolTip(
            "Upper bound on chunk index. The scanner processes chunks in "
            "[start_chunk, N_CHUNKS). Passed as --n-chunks when the checkbox "
            "is on.")
        cfg.addWidget(self.nchunks_edit, 4, 1)
        self.nchunks_override = QCheckBox("Override")
        cfg.addWidget(self.nchunks_override, 4, 2)

        # Row 5: start-chunk override (for distributed work: 'do chunks 200-300')
        cfg.addWidget(QLabel("Start chunk (--start-chunk):"), 5, 0)
        self.startchunk_edit = QSpinBox()
        self.startchunk_edit.setRange(0, 10_000_000)
        self.startchunk_edit.setValue(0)
        self.startchunk_edit.setToolTip(
            "Force the starting chunk index, IGNORING any resume checkpoint. "
            "Set this with N_CHUNKS to compute an explicit chunk range -- "
            "e.g. start=200 + N_CHUNKS=300 computes chunks 200 through 299. "
            "Passed as --start-chunk when the checkbox is on.")
        cfg.addWidget(self.startchunk_edit, 5, 1)
        self.startchunk_override = QCheckBox("Override")
        self.startchunk_override.setToolTip(
            "When on, ignores the checkpoint's next_chunk and starts at the "
            "value above. Useful for a coordinator assigning explicit ranges.")
        cfg.addWidget(self.startchunk_override, 5, 2)

        # Row 6: output prefix (for distributed work: separate output files)
        cfg.addWidget(QLabel("Output prefix (--output-prefix):"), 6, 0)
        self.prefix_edit = QLineEdit("")
        self.prefix_edit.setPlaceholderText("e.g. chunks_200_300  (files: {prefix}_hits.csv, ...)")
        self.prefix_edit.setToolTip(
            "Rename all output files. Files become {prefix}_checkpoint.json, "
            "{prefix}_hits.csv, {prefix}_count.csv, {prefix}_zeros.csv. Use "
            "this to keep contributor output separate. Passed as "
            "--output-prefix when the checkbox is on.")
        cfg.addWidget(self.prefix_edit, 6, 1, 1, 1)
        self.prefix_override = QCheckBox("Override")
        cfg.addWidget(self.prefix_override, 6, 2)

        # Row 7: reset-to-defaults button. Wipes saved QSettings and reloads
        # the coded defaults into every field. Useful if the saved config
        # gets into a weird state or the user wants a clean slate.
        self.btn_reset_cfg = QPushButton("Reset config to defaults")
        self.btn_reset_cfg.setToolTip(
            "Clear all saved settings and restore the built-in defaults. "
            "Does not affect the running scanner or any output files.")
        self.btn_reset_cfg.clicked.connect(self._reset_settings)
        cfg.addWidget(self.btn_reset_cfg, 7, 0, 1, 4)

        root.addWidget(cfg_group)

        # --- Button row ---
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_resume = QPushButton("Resume")
        self.btn_abort = QPushButton("Abort")
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_abort):
            b.setMinimumWidth(100)
            b.setMinimumHeight(32)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_resume.clicked.connect(self._on_resume)
        self.btn_abort.clicked.connect(self._on_abort)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_resume)
        btn_row.addWidget(self.btn_abort)
        # Live visualization window (opens on demand; fed from _on_status).
        self._viz = None
        self.btn_viz = QPushButton("Show Visualizations")
        self.btn_viz.setMinimumWidth(150)
        self.btn_viz.setMinimumHeight(32)
        self.btn_viz.clicked.connect(self._open_viz)
        btn_row.addWidget(self.btn_viz)
        btn_row.addStretch(1)
        self.state_label = QLabel("State: idle")
        self.state_label.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        btn_row.addWidget(self.state_label)
        root.addLayout(btn_row)

        # --- Progress row ---
        prog_group = QGroupBox("Progress")
        prog = QGridLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        prog.addWidget(self.progress_bar, 0, 0, 1, 4)

        prog.addWidget(QLabel("Chunk:"), 1, 0)
        self.chunk_label = QLabel("—")
        prog.addWidget(self.chunk_label, 1, 1)
        prog.addWidget(QLabel("Rate:"), 1, 2)
        self.rate_label = QLabel("—")
        prog.addWidget(self.rate_label, 1, 3)

        prog.addWidget(QLabel("Elapsed:"), 2, 0)
        self.elapsed_label = QLabel("—")
        prog.addWidget(self.elapsed_label, 2, 1)
        prog.addWidget(QLabel("ETA:"), 2, 2)
        self.eta_label = QLabel("—")
        prog.addWidget(self.eta_label, 2, 3)

        prog.addWidget(QLabel("Last chunk wall:"), 3, 0)
        self.wall_label = QLabel("—")
        prog.addWidget(self.wall_label, 3, 1)
        prog.addWidget(QLabel("Refined:"), 3, 2)
        self.refined_label = QLabel("—")
        prog.addWidget(self.refined_label, 3, 3)

        root.addWidget(prog_group)

        # --- Cumulative stats ---
        stats_group = QGroupBox("Cumulative statistics")
        stats = QGridLayout(stats_group)
        stats.addWidget(QLabel("Zeros located:"), 0, 0)
        self.zloc_label = QLabel("0")
        stats.addWidget(self.zloc_label, 0, 1)
        stats.addWidget(QLabel("Required (nzeros):"), 0, 2)
        self.zreq_label = QLabel("0")
        stats.addWidget(self.zreq_label, 0, 3)

        stats.addWidget(QLabel("Short after refine:"), 1, 0)
        self.zshort_label = QLabel("0")
        stats.addWidget(self.zshort_label, 1, 1)
        stats.addWidget(QLabel("Violations (survived):"), 1, 2)
        self.viol_label = QLabel("0")
        self.viol_label.setStyleSheet("font-weight: bold;")
        stats.addWidget(self.viol_label, 1, 3)
        root.addWidget(stats_group)

        # --- Tightest pair leaderboard entry ---
        tp_group = QGroupBox("Tightest pair")
        tp = QGridLayout(tp_group)

        # Three t-values stacked with matching-digit alignment so you can SEE
        # how tight the pair is at a glance. Fixed-width font, same field
        # width, same fractional precision. The extremum is in the middle
        # because that's the parabolic vertex sitting between the two zeros.
        # Left and right zeros are reconstructed as extremum +/- gap/2 in
        # DECIMAL arithmetic -- naive float64 subtraction at t~1e13 quantizes
        # both to the same ULP and would misreport the gap by ~50%.
        _tp_fixed = QFont("Consolas", 10)

        tp.addWidget(QLabel("t (left zero):"), 0, 0)
        self.tp_t_left_label = QLabel("—")
        self.tp_t_left_label.setFont(_tp_fixed)
        self.tp_t_left_label.setToolTip(
            "Location of the LEFT zero of the tight pair, reconstructed as "
            "t_extremum - gap/2 in decimal arithmetic (float64 can't "
            "distinguish the two zeros at heights >= 1e13, so we use Decimal "
            "to preserve the sub-ULP difference for display).")
        tp.addWidget(self.tp_t_left_label, 0, 1, 1, 3)

        tp.addWidget(QLabel("t (extremum):"), 1, 0)
        self.tp_t_label = QLabel("—")
        self.tp_t_label.setFont(_tp_fixed)
        self.tp_t_label.setToolTip(
            "Location of the parabolic-vertex extremum BETWEEN the two zeros "
            "of the tight pair. This is what the scanner stores as the hit's "
            "'t'; the two bracketing zero locations are derived from it.")
        tp.addWidget(self.tp_t_label, 1, 1, 1, 3)

        tp.addWidget(QLabel("t (right zero):"), 2, 0)
        self.tp_t_right_label = QLabel("—")
        self.tp_t_right_label.setFont(_tp_fixed)
        self.tp_t_right_label.setToolTip(
            "Location of the RIGHT zero of the tight pair, reconstructed as "
            "t_extremum + gap/2 in decimal arithmetic.")
        tp.addWidget(self.tp_t_right_label, 2, 1, 1, 3)

        tp.addWidget(QLabel("zero index (left):"), 3, 0)
        self.tp_zi_label = QLabel("—")
        self.tp_zi_label.setFont(_tp_fixed)
        self.tp_zi_label.setToolTip(
            "Ordinal index of the LEFT zero of the tight pair. The right "
            "zero's index is one more. This is the standard way of "
            "referring to a Riemann zeta zero in the literature "
            "(e.g. 'the 10^13-th zero').")
        tp.addWidget(self.tp_zi_label, 3, 1, 1, 3)

        tp.addWidget(QLabel("norm_gap:"), 4, 0)
        self.tp_ngap_label = QLabel("—")
        tp.addWidget(self.tp_ngap_label, 4, 1)
        tp.addWidget(QLabel("gap:"), 4, 2)
        self.tp_gap_label = QLabel("—")
        tp.addWidget(self.tp_gap_label, 4, 3)
        tp.addWidget(QLabel("Z at extremum:"), 5, 0)
        self.tp_z_label = QLabel("—")
        tp.addWidget(self.tp_z_label, 5, 1)
        tp.addWidget(QLabel("kind:"), 5, 2)
        self.tp_kind_label = QLabel("—")
        tp.addWidget(self.tp_kind_label, 5, 3)
        root.addWidget(tp_group)

        # Trailing stretch pushes all left-column widgets to the top when the
        # splitter pane is taller than their combined natural height. Without
        # this, Qt would distribute empty vertical space between the groups
        # (making them look awkwardly spread out on a tall window).
        root.addStretch(1)

        # --- Live output pane (RIGHT COLUMN of the horizontal splitter) ---
        out_group = QGroupBox("Live output")
        out_v = QVBoxLayout(out_group)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 9))
        self.output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        # Set the document's max-block-count so old lines auto-evict, keeping
        # memory bounded on multi-day runs.
        self.output.document().setMaximumBlockCount(LIVE_OUTPUT_MAX_LINES)
        out_v.addWidget(self.output)

        out_btns = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.output.clear)
        out_btns.addStretch(1)
        out_btns.addWidget(clear_btn)
        out_v.addLayout(out_btns)

        # Add to the RIGHT side of the horizontal splitter (not to left_col).
        # Stretch factors: left column stays compact (0), right column grows
        # to fill horizontal space (1). User can drag the divider to override.
        self.main_splitter.addWidget(out_group)
        self.main_splitter.setStretchFactor(0, 0)   # left: content-sized
        self.main_splitter.setStretchFactor(1, 1)   # right: expandable
        # Sensible initial split (left ~480px, rest to output) -- overridden
        # by any saved splitter state in _load_settings.
        self.main_splitter.setSizes([480, 700])

        # Also give the left column a minimum width so it doesn't get
        # crushed to nothing by an aggressive drag on the divider.
        left_widget.setMinimumWidth(400)

        # --- Status bar ---
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")

    # ------------------------- Browsers -------------------------
    def _browse_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select scanner script", os.getcwd(),
            "Python files (*.py);;All files (*)")
        if path:
            self.script_edit.setText(path)

    def _browse_cwd(self):
        path = QFileDialog.getExistingDirectory(
            self, "Working directory", os.getcwd())
        if path:
            self.cwd_edit.setText(path)

    # ------------------------- State machine -------------------------
    def _set_state(self, state):
        self.state = state
        self.state_label.setText(f"State: {state}")

        # Color the state label so it's visible at a glance
        colors = {
            self.STATE_IDLE:     "#666666",
            self.STATE_RUNNING:  "#1a7f37",
            self.STATE_PAUSING:  "#bf8700",
            self.STATE_PAUSED:   "#8250df",
            self.STATE_ABORTING: "#cf222e",
        }
        self.state_label.setStyleSheet(
            f"font-weight: bold; padding: 4px 12px; color: {colors.get(state, 'black')};")

        # Button enable matrix per state
        # Start is available when IDLE or PAUSED. In PAUSED state, Start
        # means "ignore the checkpoint and start fresh from the configured
        # start-chunk" -- the user might want this if they need to redo
        # a range. Resume is the safer option (uses checkpoint's next_chunk).
        can_start  = state in (self.STATE_IDLE, self.STATE_PAUSED)
        can_resume = state in (self.STATE_PAUSED,)
        can_pause  = state in (self.STATE_RUNNING,)
        can_abort  = state in (self.STATE_RUNNING, self.STATE_PAUSING)
        self.btn_start.setEnabled(can_start)
        self.btn_resume.setEnabled(can_resume)
        self.btn_pause.setEnabled(can_pause)
        self.btn_abort.setEnabled(can_abort)

        # Config editable only when idle/paused (not during a live run)
        cfg_editable = state in (self.STATE_IDLE, self.STATE_PAUSED)
        self.script_edit.setEnabled(cfg_editable)
        self.cwd_edit.setEnabled(cfg_editable)
        self.flag_edit.setEnabled(cfg_editable)
        self.tbase_edit.setEnabled(cfg_editable)
        self.tbase_override.setEnabled(cfg_editable)
        self.nchunks_edit.setEnabled(cfg_editable)
        self.nchunks_override.setEnabled(cfg_editable)
        self.startchunk_edit.setEnabled(cfg_editable)
        self.startchunk_override.setEnabled(cfg_editable)
        self.prefix_edit.setEnabled(cfg_editable)
        self.prefix_override.setEnabled(cfg_editable)
        self.btn_reset_cfg.setEnabled(cfg_editable)

    # ------------------------- Start / Pause / Resume / Abort -------------------------
    def _spawn_scanner(self, is_resume=False):
        """Launch the scanner subprocess with current UI config.

        Any override checkbox that's checked adds the corresponding CLI flag;
        anything unchecked lets the scanner use its module default. This
        keeps the default UI launch (all checkboxes off) equivalent to just
        running `python zeta_gpu_scan_v6_hp.py` at the command line.

        When is_resume=True, the --start-chunk flag is NEVER passed, even if
        the override checkbox is checked. Resume means "let the checkpoint's
        next_chunk take over." The start-chunk override is for the FIRST
        launch of a distributed work unit, not for subsequent resumes within
        that work unit.
        """
        script = self.script_edit.text().strip()
        cwd = self.cwd_edit.text().strip() or os.getcwd()
        # Resolve script path (allow both absolute and cwd-relative)
        if os.path.isabs(script):
            if not os.path.exists(script):
                QMessageBox.critical(self, "Script not found",
                    f"Can't find scanner script:\n{script}")
                return False
        else:
            if not os.path.exists(os.path.join(cwd, script)):
                QMessageBox.critical(self, "Script not found",
                    f"Can't find scanner script:\n{script}\n"
                    f"in working directory:\n{cwd}")
                return False

        # Base argv -- always include the two "plumbing" flags for UI operation
        self.pause_flag_path = os.path.join(cwd, self.flag_edit.text().strip() or PAUSE_FLAG_NAME)
        try:
            if os.path.exists(self.pause_flag_path):
                os.remove(self.pause_flag_path)   # clear stale flag
        except OSError:
            pass

        argv = [sys.executable, "-u", script,       # -u = unbuffered stdout
                "--json-progress",
                "--pause-flag", self.pause_flag_path]

        # V7 live-visualization sample emission. Always pass the flag + path;
        # the scanner only actually emits when the UI has granted a credit
        # (dropped the emit flag file), which the viz window does when it's
        # open and ready. So with the viz closed, no credit is ever granted
        # and the scanner emits nothing -- zero overhead, exactly like v6.
        self.emit_flag_path = os.path.join(cwd, EMIT_FLAG_NAME)
        try:
            if os.path.exists(self.emit_flag_path):
                os.remove(self.emit_flag_path)   # clear stale credit
        except OSError:
            pass
        argv += ["--emit-samples", "--emit-flag", self.emit_flag_path]

        # --- Config-override flags (only when the checkbox is on) ---
        if self.tbase_override.isChecked():
            tbase_txt = self.tbase_edit.text().strip()
            try:
                float(tbase_txt)     # validate parseable
            except ValueError:
                QMessageBox.critical(self, "Invalid T_BASE",
                    f"'{tbase_txt}' is not a valid number.")
                return False
            argv += ["--t-base", tbase_txt]

        if self.nchunks_override.isChecked():
            argv += ["--n-chunks", str(self.nchunks_edit.value())]

        if self.startchunk_override.isChecked() and not is_resume:
            sc_val = self.startchunk_edit.value()
            # Sanity check: start must be < N_CHUNKS. If the user overrode
            # both, catch the mistake here before the scanner logs 0 chunks.
            if self.nchunks_override.isChecked():
                nc_val = self.nchunks_edit.value()
                if sc_val >= nc_val:
                    QMessageBox.critical(self, "Invalid chunk range",
                        f"Start chunk ({sc_val}) must be less than "
                        f"N_CHUNKS ({nc_val}). The scanner processes chunks "
                        f"in [start_chunk, N_CHUNKS), so this range is empty.")
                    return False
            argv += ["--start-chunk", str(sc_val)]
        elif is_resume:
            self.output.append(
                "[UI] Resume: ignoring start-chunk override, letting "
                "checkpoint's next_chunk take over.")

        if self.prefix_override.isChecked():
            pref = self.prefix_edit.text().strip()
            if not pref:
                QMessageBox.critical(self, "Empty output prefix",
                    "Output prefix override is checked but the field is "
                    "empty. Provide a prefix (e.g. 'chunks_200_300') or "
                    "uncheck the override.")
                return False
            argv += ["--output-prefix", pref]

        # Launch
        try:
            self.output.append(f"$ {' '.join(argv)}")
            self.proc = subprocess.Popen(
                argv, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                # Windows: CREATE_NEW_PROCESS_GROUP so we can send Ctrl+Break
                # cleanly if we want (we prefer pause flag / terminate instead,
                # but this makes a signal-based path possible later).
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               if os.name == "nt" else 0))
        except Exception as e:
            QMessageBox.critical(self, "Failed to launch",
                f"Could not launch scanner:\n{e}")
            return False

        # Reader thread
        self.reader = ScannerReader(self.proc)
        self.reader.lineReceived.connect(self._on_line)
        self.reader.statusReceived.connect(self._on_status)
        self.reader.sampleReceived.connect(self._on_sample)
        self.reader.zeroReceived.connect(self._on_zero)
        self.reader.processExited.connect(self._on_process_exited)
        self.reader.start()

        self.run_start_time = datetime.now()
        self.chunks_completed_this_run = 0
        self._set_state(self.STATE_RUNNING)
        self.statusBar().showMessage(f"Scanner started (PID {self.proc.pid}).")
        # If the viz window is already open, grant the first emit credit now
        # that emit_flag_path exists, so the first chunk streams its samples.
        if self._viz is not None:
            self._grant_credit()
        return True

    def _on_start(self):
        if self.state not in (self.STATE_IDLE, self.STATE_PAUSED):
            return
        # If we're in PAUSED state (checkpoint exists), warn the user that
        # Start will use the start-chunk override (if checked) and may
        # re-process already-completed chunks. Resume is usually the right
        # choice.
        if self.state == self.STATE_PAUSED:
            if self.startchunk_override.isChecked():
                sc_val = self.startchunk_edit.value()
                msg = (
                    f"A checkpoint exists from a prior run. Clicking Start "
                    f"with the start-chunk override will begin at chunk "
                    f"{sc_val}, which may RE-PROCESS chunks that were already "
                    f"completed and DUPLICATE rows in the zeros CSV.\n\n"
                    f"Use RESUME instead to continue from where the checkpoint "
                    f"left off.\n\n"
                    f"Are you sure you want to Start from chunk {sc_val}?")
            else:
                msg = (
                    "A checkpoint exists from a prior run. Start will begin "
                    "from the checkpoint's next_chunk (same as Resume in this "
                    "case, since no start-chunk override is set).\n\n"
                    "Continue?")
            confirm = QMessageBox.question(
                self, "Start with existing checkpoint?", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if confirm != QMessageBox.StandardButton.Yes:
                return
        # Persist the config used to launch this run, so next launch of the
        # UI restores the same setup automatically. Done BEFORE spawn so the
        # config is saved even if spawn fails for some reason.
        self._save_settings()
        self._spawn_scanner()

    def _on_pause(self):
        if self.state != self.STATE_RUNNING or self.proc is None:
            return
        # Create the pause flag file; scanner will notice it between chunks
        try:
            Path(self.pause_flag_path).touch()
        except Exception as e:
            QMessageBox.warning(self, "Pause flag error",
                f"Could not create pause flag file:\n{e}")
            return
        self._set_state(self.STATE_PAUSING)
        self.statusBar().showMessage(
            "Pause requested -- waiting for current chunk to finish "
            "(this may take several minutes at 1e13).")
        self.output.append("[UI] Pause flag set. Waiting for clean exit at next chunk boundary…")

    def _on_resume(self):
        if self.state != self.STATE_PAUSED:
            return
        # Make sure the flag is gone (defensive)
        if self.pause_flag_path and os.path.exists(self.pause_flag_path):
            try:
                os.remove(self.pause_flag_path)
            except OSError:
                pass
        self.output.append("[UI] Resuming from checkpoint...")
        self._spawn_scanner(is_resume=True)

    def _on_abort(self):
        if self.state not in (self.STATE_RUNNING, self.STATE_PAUSING):
            return
        confirm = QMessageBox.question(
            self, "Confirm abort",
            "Abort will terminate the scanner immediately. Work in the "
            "current in-flight chunk will be lost (previous chunks are "
            "safely checkpointed). Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._set_state(self.STATE_ABORTING)
        self.output.append("[UI] Sending terminate() to scanner subprocess…")
        try:
            self.proc.terminate()
        except Exception as e:
            self.output.append(f"[UI] terminate() failed: {e}")
        # If it doesn't die in a few seconds, escalate to kill
        QTimer.singleShot(5000, self._escalate_kill_if_needed)

    def _escalate_kill_if_needed(self):
        if self.proc is not None and self.proc.poll() is None:
            self.output.append("[UI] Scanner didn't exit in 5s -- sending kill().")
            try:
                self.proc.kill()
            except Exception:
                pass

    # ------------------------- Line + status callbacks -------------------------
    def _on_line(self, line: str):
        # Mirror every non-status line to the output pane.
        self.output.append(line)
        # Auto-scroll to the bottom
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output.setTextCursor(cursor)

    def _open_viz(self):
        """Open (or re-focus) the live visualization window. It plays the V7
        sample stream: full-resolution Z(t), the S(t)/zeros staircase, and the
        ticker. Until the first live buffer arrives it loops precomputed
        bootstrap data. Opening the window grants the scanner its first emit
        credit, so the next chunk begins streaming samples."""
        if self._viz is None:
            cwd = os.path.dirname(os.path.abspath(__file__))
            boot = os.path.join(cwd, BOOTSTRAP_FILE)
            self._viz = ZetaVizWindow(
                self, bootstrap_path=boot if os.path.exists(boot) else None)
            # credit handshake: the viz asks for a buffer -> we touch the flag.
            self._viz.creditReady.connect(self._grant_credit)
        self._viz.show()
        self._viz.raise_()
        self._viz.activateWindow()
        # grant the first credit so the next chunk emits (if a scan is running)
        self._grant_credit()
        # Safety net: the credit handshake can stall if a grant is missed (a
        # race between flag creation and the scanner's chunk-top check, or a
        # buffer-finish re-grant that didn't fire). A slow timer re-grants when
        # the viz is idle (no live buffer queued or playing) so it self-heals
        # instead of freezing on bootstrap. Cheap: touches a file every 3s only
        # while the viz is open AND a scan is running AND nothing is queued.
        if not hasattr(self, "_credit_timer"):
            self._credit_timer = QTimer(self)
            self._credit_timer.timeout.connect(self._credit_safety_net)
            self._credit_timer.start(3000)

    def _credit_safety_net(self):
        if self._viz is None:
            return
        if getattr(self, "emit_flag_path", None) is None:
            return
        if self.state != self.STATE_RUNNING:
            return
        c = self._viz._canvas
        # If we have no live buffer waiting and aren't currently playing a live
        # buffer, we're idle w.r.t. the scanner -> make sure a credit is out.
        playing_live = (c._cur is not None and not c._cur.get("bootstrap"))
        has_queued_live = any(not b.get("bootstrap") for b in c._queue)
        if not playing_live and not has_queued_live:
            # only (re)grant if the flag isn't already sitting there uneaten
            path = self.emit_flag_path
            if not os.path.exists(path):
                self._grant_credit()

    def _grant_credit(self):
        """Create the emit-flag file: 'scanner, please fill one buffer.' The
        scanner consumes (deletes) it when a chunk emits; the viz calls this
        again once it's finished playing that buffer. No-op if no scan is
        running (flag path unset)."""
        path = getattr(self, "emit_flag_path", None)
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write("1")
        except OSError:
            pass

    def _on_sample(self, block: dict):
        if self._viz is not None:
            self._viz.feed_samples(block)

    def _on_zero(self, ev: dict):
        if self._viz is not None:
            self._viz.feed_zero(ev)

    def _on_status(self, event: dict):
        # Route each event to its widget update
        kind = event.get("event", "")
        if kind == "run_start":
            self.first_chunk_this_run = int(event.get("start_chunk", 0))
            self.total_chunks_target = int(event.get("n_chunks", 0))
            self.chunk_t_current = float(event.get("chunk_t", 0.0))
            self.progress_bar.setRange(0, max(1, self.total_chunks_target))
            self.progress_bar.setValue(self.first_chunk_this_run)
            self.progress_bar.setFormat(
                f"chunk %v / %m ({self.first_chunk_this_run}/"
                f"{self.total_chunks_target} resume)")
            # Populate ALL resume-time cumulative widgets so the UI is
            # meaningful before the first chunk of this session completes
            # (which at 1e13 can be 6+ minutes -- otherwise everything reads
            # '—' during that first chunk).
            self.zloc_label.setText(f"{event.get('zeros_located_resume', 0):,}")
            self.zreq_label.setText(f"{event.get('zeros_required_resume', 0):,}")
            self.zshort_label.setText(str(event.get('zeros_short_resume', 0)))
            self.viol_label.setText(str(event.get('violations_resume', 0)))
            self.chunk_label.setText(f"{self.first_chunk_this_run} (resuming)")
            self._update_tightest_pair(event.get("tightest_resume"))
            self.statusBar().showMessage(
                f"Run start: T_BASE={event.get('t_base')}, "
                f"chunks {self.first_chunk_this_run}..{self.total_chunks_target}")

        elif kind == "chunk_start":
            self.chunk_label.setText(f"{event.get('chunk')} (starting)")

        elif kind == "chunk_end":
            c = int(event.get("chunk", 0))
            self.chunks_completed_this_run += 1
            self.progress_bar.setValue(c + 1)
            self.chunk_label.setText(str(c + 1))
            self.wall_label.setText(f"{event.get('wall', 0.0):.1f} s")
            self.last_wall = float(event.get("wall", 0.0))
            # Show all three rates so recent optimizations are visible without
            # waiting for the cumulative average to catch up: instantaneous
            # (this chunk alone), 10-chunk moving average (smooth but current),
            # and cumulative (whole-session, useful for long-run ETAs).
            r_inst = event.get('rate_t_per_s', 0.0)   # renamed to inst upstream
            r_recent = event.get('rate_recent_t_per_s', r_inst)
            r_cum = event.get('rate_cumulative_t_per_s', r_inst)
            self.rate_label.setText(
                f"{r_inst:.2f} t/s   (10-chunk avg: {r_recent:.2f}, "
                f"cum: {r_cum:.2f})")
            # Stash recent-rate-inverted (= recent avg wall time per chunk)
            # so the ETA ticker can use it instead of the session average,
            # which lags badly after any speedup lands.
            if r_recent > 0 and self.chunk_t_current > 0:
                self._recent_avg_wall = self.chunk_t_current / r_recent
            self.refined_label.setText("yes" if event.get("refined") else "no")
            self.zloc_label.setText(f"{event.get('zeros_located_total', 0):,}")
            self.zreq_label.setText(f"{event.get('zeros_required_total', 0):,}")
            # Shortfall display: distinguish confirmed unresolved misses from
            # pending-seam-noise that may self-heal on the next chunk. The
            # scanner's `short` counter no longer includes pending shortfalls
            # (they only count once confirmed as a real miss). We show both:
            # "5" if fully confirmed, "0 (2 pending)" if the current chunk was
            # short and we're waiting to see if the next chunk absorbs it.
            _short_total = int(event.get("zeros_short_total", 0))
            _pending = int(event.get("pending_short", 0))
            if _pending > 0:
                self.zshort_label.setText(f"{_short_total} ({_pending} pending seam)")
            else:
                self.zshort_label.setText(str(_short_total))

            # Also mention any seam events that happened this chunk in the
            # status bar -- so the user sees when auto-recovery fires and
            # what it did. These are transient (overwritten on the next
            # chunk_end), which is fine; the persistent record lives in the
            # log output pane and (for real misses) the shortfall counter.
            _healed = int(event.get("self_healed", 0))
            _rec = int(event.get("recovered", 0))
            _rm = int(event.get("real_miss", 0))
            if _rec > 0:
                self.statusBar().showMessage(
                    f"Chunk {event.get('chunk')}: ultra-fine recovered "
                    f"{_rec} zeros from seam boundary noise.", 8000)
            elif _healed > 0:
                self.statusBar().showMessage(
                    f"Chunk {event.get('chunk')}: seam boundary noise "
                    f"({_healed} zeros) self-healed with prior chunk. "
                    f"No real miss.", 5000)
            elif _rm > 0:
                self.statusBar().showMessage(
                    f"Chunk {event.get('chunk')}: {_rm} REAL MISS zero(s) "
                    f"unrecovered by ultra-fine resweep. See log.", 15000)

            self._update_tightest_pair(event.get("tightest"))

        elif kind == "violation_survived":
            # Update violation counter and pop a modal -- this is a huge deal
            try:
                current = int(self.viol_label.text())
            except ValueError:
                current = 0
            self.viol_label.setText(str(current + 1))
            self.viol_label.setStyleSheet("font-weight: bold; color: red;")
            QMessageBox.critical(self, "RH VIOLATION SURVIVED dps=200",
                f"A candidate RH violation survived escalation to dps=200:\n\n"
                f"  t = {event.get('t')}\n"
                f"  Z = {event.get('Z')}\n"
                f"  kind = {event.get('kind')}\n"
                f"  chunk = {event.get('chunk')}\n\n"
                f"This is either a bug or a genuine finding. Investigate.")

        elif kind == "paused":
            self._set_state(self.STATE_PAUSED)
            self.statusBar().showMessage(
                f"Paused at chunk {event.get('next_chunk')}. "
                "Click Resume to continue.")

        elif kind == "interrupted":
            self.statusBar().showMessage("Scanner interrupted (Ctrl+C).")

        elif kind == "run_end":
            complete = event.get("complete", False)
            zloc = event.get("zeros_located_total", 0)
            self.statusBar().showMessage(
                f"Run complete: {zloc:,} zeros located. "
                + ("All accounted for." if complete else "SHORT -- see logs."))

        # Forward every event to the live visualization window if it's open.
        # feed() ignores kinds it doesn't use, and self-paces its own repaint,
        # so this is cheap even during a burst of chunk_end events.
        # Forward status events to the live visualization (rate, records).
        # Sample/zero streams go through _on_sample/_on_zero separately.
        if self._viz is not None:
            self._viz.feed_status(event)

    def _on_process_exited(self, code: int):
        self.output.append(f"[UI] Scanner exited with code {code}.")
        # Transition to the right post-exit state:
        #   STATE_PAUSING → STATE_PAUSED (normal pause completion)
        #   STATE_PAUSED  → stay PAUSED  (the "paused" status event already
        #                   transitioned us; don't clobber it back to IDLE)
        #   anything else → STATE_IDLE   (abort, natural end, crash)
        if self.state in (self.STATE_PAUSING, self.STATE_PAUSED):
            self._set_state(self.STATE_PAUSED)
            self.statusBar().showMessage(
                f"Paused (scanner exited with code {code}). "
                "Click Resume to continue.")
        else:
            self._set_state(self.STATE_IDLE)
            self.statusBar().showMessage(f"Scanner exited (code {code}). Ready.")
        self.proc = None

    def _update_tightest_pair(self, tp):
        """Populate the tightest-pair widget from a dict payload. Called on
        both run_start (tightest_resume field) and chunk_end (tightest field).
        Both payloads share the same shape: t, z, kind, gap, norm_gap, and --
        for post-index scans -- zero_index. Older resumed checkpoints won't
        have zero_index; we show '—' for those, still correctly showing all
        the other fields.

        The three t values (left zero / extremum / right zero) are displayed
        with matching-digit alignment so the tightness is visually obvious.
        Left and right are computed as extremum +/- gap/2 using DECIMAL
        arithmetic: at 1e13, float64's ULP is ~2e-3, so naive float
        subtraction of gap/2 (~1e-3) would either quantize to zero (both
        zeros displayed identically) or to the wrong ULP (misreporting the
        difference by ~50%). Decimal preserves the full precision the gap
        was measured to (~1e-11), anchored to the extremum's storage value.
        """
        if not tp:
            return
        t_ext = tp.get('t', 0)
        gap = tp.get('gap')

        # Format the three t values with matched precision so they align
        # visually. We use 10 fractional digits -- enough to show the
        # sub-thousandths difference for even tighter pairs than the
        # current 0.01 norm_gap record.
        T_FMT_DIGITS = 10
        try:
            t_ext_str = f"{t_ext:.{T_FMT_DIGITS}f}"
        except (TypeError, ValueError):
            t_ext_str = str(t_ext)
        self.tp_t_label.setText(t_ext_str)

        if gap is not None and gap > 0:
            # Set Decimal precision high enough that the digit-level
            # subtraction doesn't lose accuracy. 30 significant digits is
            # comfortably more than we display.
            getcontext().prec = 40
            try:
                d_ext = Decimal(repr(float(t_ext)))
                d_half = Decimal(repr(float(gap))) / Decimal(2)
                d_left = d_ext - d_half
                d_right = d_ext + d_half
                # Format both with the same total width so digits line up
                # under the extremum row
                left_str  = f"{d_left:.{T_FMT_DIGITS}f}"
                right_str = f"{d_right:.{T_FMT_DIGITS}f}"
                self.tp_t_left_label.setText(left_str)
                self.tp_t_right_label.setText(right_str)
            except Exception:
                self.tp_t_left_label.setText("—")
                self.tp_t_right_label.setText("—")
        else:
            # No precise gap available (rare edge case) -- can't split
            self.tp_t_left_label.setText("(no precise gap)")
            self.tp_t_right_label.setText("(no precise gap)")

        self.tp_ngap_label.setText(f"{tp.get('norm_gap', 0):.5f}")
        self.tp_gap_label.setText(f"{gap:.6e}" if gap else "—")
        self.tp_z_label.setText(f"{tp.get('z', 0):+.3e}")
        self.tp_kind_label.setText(tp.get('kind', '—'))
        zi = tp.get('zero_index')
        if zi is not None:
            # Show "#N (right: N+1)" so it's clear the ordinal names the LEFT
            # zero of the pair and the tight pair is between #N and #N+1.
            try:
                zi_int = int(zi)
                self.tp_zi_label.setText(f"#{zi_int:,} (right: #{zi_int+1:,})")
            except (TypeError, ValueError):
                self.tp_zi_label.setText(str(zi))
        else:
            self.tp_zi_label.setText("—")

    # ------------------------- Elapsed / ETA ticker -------------------------
    def _update_elapsed(self):
        if self.run_start_time is None or self.state == self.STATE_IDLE:
            self.elapsed_label.setText("—")
            self.eta_label.setText("—")
            return
        elapsed = datetime.now() - self.run_start_time
        self.elapsed_label.setText(str(elapsed).split(".")[0])
        # ETA uses the recent chunk rate (last 10 chunks avg) if we have
        # one, otherwise falls back to session average. Recent rate is much
        # more useful when an optimization lands mid-run: cumulative rate
        # takes hours to catch up to a new speedup, whereas 10-chunk avg
        # reflects reality within ~30-40 minutes.
        if (self.chunks_completed_this_run > 0
            and self.total_chunks_target > 0):
            remaining_chunks = (self.total_chunks_target
                                - (self.first_chunk_this_run + self.chunks_completed_this_run))
            if remaining_chunks > 0:
                # Prefer the recent-rate the scanner just emitted (via
                # self._last_recent_wall stashed by _on_status). Fall back
                # to session average when we don't have one yet.
                recent_avg = getattr(self, "_recent_avg_wall", None)
                if recent_avg is not None and recent_avg > 0:
                    avg_per_chunk = recent_avg
                else:
                    avg_per_chunk = elapsed.total_seconds() / self.chunks_completed_this_run
                eta_secs = int(avg_per_chunk * remaining_chunks)
                self.eta_label.setText(str(timedelta(seconds=eta_secs)))
            else:
                self.eta_label.setText("done")
        else:
            self.eta_label.setText("—")

    # ------------------------- Settings persistence -------------------------
    # Config values persist across runs via QSettings -- on Windows this uses
    # the registry (HKCU\Software\SetiAstro\ZetaSweepUI), on macOS a plist
    # file, on Linux an ini file. No manual paths, no config-file bookkeeping.
    # Called on __init__ (load) and closeEvent + Start (save), so a user who
    # sets up 200-chunk overrides once never has to re-type them.
    #
    # Not persisted: anything derived from a running scanner (state, elapsed
    # time, PID, live output pane -- those are per-run, not per-config).

    def _load_settings(self):
        """Restore config field values from persistent storage. Silently
        skips any missing keys, which is what you want on the first launch
        (all fields keep their code-level defaults)."""
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        # Text fields
        self.script_edit.setText(
            s.value("script_path", SCANNER_SCRIPT_DEFAULT, type=str))
        self.cwd_edit.setText(
            s.value("working_dir", os.getcwd(), type=str))
        self.flag_edit.setText(
            s.value("pause_flag", PAUSE_FLAG_NAME, type=str))
        self.tbase_edit.setText(
            s.value("tbase_text", "1e13", type=str))
        self.prefix_edit.setText(
            s.value("output_prefix", "", type=str))
        # Numeric fields
        self.nchunks_edit.setValue(int(s.value("nchunks", 100, type=int)))
        self.startchunk_edit.setValue(int(s.value("startchunk", 0, type=int)))
        # Checkboxes -- QSettings stores bools as strings on some backends
        # (Windows registry uses "true"/"false"), so coerce via a helper.
        def _as_bool(v, default=False):
            if isinstance(v, bool): return v
            if v is None: return default
            return str(v).lower() in ("true", "1", "yes")
        self.tbase_override.setChecked(_as_bool(s.value("tbase_override")))
        self.nchunks_override.setChecked(_as_bool(s.value("nchunks_override")))
        self.startchunk_override.setChecked(_as_bool(s.value("startchunk_override")))
        self.prefix_override.setChecked(_as_bool(s.value("prefix_override")))
        # Layout geometry: window size/position and splitter divider location.
        # Qt gives us QByteArray blobs for these; restoreGeometry/restoreState
        # handle any format quirks and no-op cleanly on missing keys.
        geom = s.value("window_geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        split = s.value("splitter_state")
        if split is not None:
            self.main_splitter.restoreState(split)

    def _save_settings(self):
        """Persist the current config field values. Called on close and when
        Start is clicked (so a config used to launch a run is saved even if
        the user never closes the window)."""
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        s.setValue("script_path", self.script_edit.text())
        s.setValue("working_dir", self.cwd_edit.text())
        s.setValue("pause_flag", self.flag_edit.text())
        s.setValue("tbase_text", self.tbase_edit.text())
        s.setValue("output_prefix", self.prefix_edit.text())
        s.setValue("nchunks", self.nchunks_edit.value())
        s.setValue("startchunk", self.startchunk_edit.value())
        s.setValue("tbase_override", self.tbase_override.isChecked())
        s.setValue("nchunks_override", self.nchunks_override.isChecked())
        s.setValue("startchunk_override", self.startchunk_override.isChecked())
        s.setValue("prefix_override", self.prefix_override.isChecked())
        # Layout geometry (see _load_settings)
        s.setValue("window_geometry", self.saveGeometry())
        s.setValue("splitter_state", self.main_splitter.saveState())
        s.sync()   # flush to backend immediately

    def _detect_existing_checkpoint(self):
        """Check if a checkpoint file from a prior session exists in the
        working directory. Returns True if the checkpoint shows progress
        (next_chunk > 0), meaning a Resume makes sense.

        Uses the current UI settings (cwd, prefix) to figure out the
        checkpoint filename, matching the same logic the scanner uses:
          - No prefix override: "zeta_scan_v6_1p0e13.json"
          - Prefix override checked: "{prefix}_checkpoint.json"
        """
        cwd = self.cwd_edit.text().strip() or os.getcwd()
        if self.prefix_override.isChecked():
            pref = self.prefix_edit.text().strip()
            if pref:
                ckpt_name = f"{pref}_checkpoint.json"
            else:
                return False    # prefix override on but empty, can't determine file
        else:
            ckpt_name = "zeta_scan_v6_1p0e13.json"
        ckpt_path = os.path.join(cwd, ckpt_name)
        if not os.path.exists(ckpt_path):
            return False
        try:
            import json
            with open(ckpt_path) as f:
                data = json.load(f)
            nc = data.get("next_chunk", 0)
            if nc > 0:
                self.output.append(
                    f"[UI] Found existing checkpoint: {ckpt_name} "
                    f"(next_chunk={nc}, zeros_located="
                    f"{data.get('zeros_located', '?')})")
                return True
        except Exception:
            pass
        return False

    def _reset_settings(self):
        """Wipe all saved settings and reload the coded defaults into the
        UI. Confirmation dialog first -- this is a destructive action."""
        r = QMessageBox.question(
            self, "Reset saved configuration?",
            "Clear all saved config (T_BASE, N_CHUNKS, paths, override "
            "checkboxes, etc.) and restore the built-in defaults?\n\n"
            "The running scanner (if any) is not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return
        s = QSettings(SETTINGS_ORG, SETTINGS_APP)
        s.clear()
        s.sync()
        # Push defaults back into the widgets
        self.script_edit.setText(SCANNER_SCRIPT_DEFAULT)
        self.cwd_edit.setText(os.getcwd())
        self.flag_edit.setText(PAUSE_FLAG_NAME)
        self.tbase_edit.setText("1e13")
        self.prefix_edit.setText("")
        self.nchunks_edit.setValue(100)
        self.startchunk_edit.setValue(0)
        self.tbase_override.setChecked(False)
        self.nchunks_override.setChecked(False)
        self.startchunk_override.setChecked(False)
        self.prefix_override.setChecked(False)
        # Restore default window size and splitter divider position
        self.resize(1400, 780)
        self.main_splitter.setSizes([480, 700])
        self.statusBar().showMessage("Configuration reset to defaults.")

    # ------------------------- Shutdown safety -------------------------
    def closeEvent(self, event):
        # Always persist config on close, even if the user is aborting a
        # running scanner -- their config choices should survive.
        self._save_settings()
        if self.state in (self.STATE_RUNNING, self.STATE_PAUSING):
            r = QMessageBox.question(
                self, "Scanner still running",
                "The scanner is still running. Close anyway?\n\n"
                "The scanner subprocess will be terminated. Previously "
                "completed chunks are safely checkpointed.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            try:
                if self.proc is not None:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        event.accept()


# --------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # cleaner cross-platform look
    # QSettings uses these to pick a per-user storage location automatically.
    # Setting them here means we never have to pass them to QSettings() --
    # every QSettings() constructor picks them up implicitly.
    app.setOrganizationName(SETTINGS_ORG)
    app.setApplicationName(SETTINGS_APP)
    win = ScannerUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()