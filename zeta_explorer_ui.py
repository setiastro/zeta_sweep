"""
zeta_explorer_ui.py
===================
Interactive explorer for Lehmer near-misses (tight zero pairs) of the
Riemann zeta function. Computes Z(t), S(t), and the parametric
zeta(1/2+it) trace in the complex plane around a user-specified t value,
using multiprocessing to soak all CPU cores.

Four synchronized panels, all color-coded by t position:
  - Z(t) trace with zero crossings and extremum marker
  - S(t) = N_observed(t) - N_smooth(t), the zero-counting correction
  - Parametric zeta(1/2+it) in the complex plane (full view)
  - Parametric zeta(1/2+it) zoomed near the origin (the near-miss loop)

All panels are interactive (pan/zoom via the matplotlib toolbar).
Includes preset targets for known interesting Lehmer pairs.

Requires: PyQt6, matplotlib, numpy, mpmath
"""

import sys, os, time, json
import numpy as np
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,
                                                 NavigationToolbar2QT)
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QPushButton, QDoubleSpinBox, QSpinBox,
    QComboBox, QProgressBar, QLabel, QStatusBar, QMessageBox,
    QFileDialog, QSplitter
)


# =====================================================================
# Multiprocessing worker (top-level function for pickling)
# =====================================================================

def _compute_point(args):
    """Compute Z(t) and theta(t) for a single t value at given dps.
    Returns (z_float, theta_float). Must be a top-level function so
    multiprocessing can pickle it."""
    from mpmath import mp, mpf, siegelz, siegeltheta
    t_str, dps = args
    mp.dps = dps
    t = mpf(t_str)
    z = float(siegelz(t))
    th = float(siegeltheta(t))
    return (z, th)


# =====================================================================
# Background computation thread
# =====================================================================

class ComputeThread(QThread):
    """Runs the multiprocessing computation off the GUI thread so the
    UI stays responsive. Emits progress updates and the final result."""
    progress = pyqtSignal(int, int)       # (done, total)
    finished = pyqtSignal(dict)           # result payload
    error    = pyqtSignal(str)

    def __init__(self, t_center, half_width, step, dps, parent=None):
        super().__init__(parent)
        self.t_center = t_center
        self.half_width = half_width
        self.step = step
        self.dps = dps

    def run(self):
        try:
            t_vals = np.arange(self.t_center - self.half_width,
                               self.t_center + self.half_width, self.step)
            n = len(t_vals)
            if n < 3:
                self.error.emit("Too few sample points. Increase half-width "
                                "or decrease step size.")
                return

            # Build argument list with full-precision string representation.
            # Must convert numpy float64 to Python float first, because
            # repr(np.float64(x)) produces "np.float64(x)" which mpmath
            # can't parse.
            args = [(repr(float(t)), self.dps) for t in t_vals]

            ncpu = max(1, cpu_count() - 1)   # leave 1 core for the UI
            self.progress.emit(0, n)
            wall0 = time.time()

            z_vals = np.empty(n)
            theta_vals = np.empty(n)

            with Pool(processes=ncpu) as pool:
                for i, (z, th) in enumerate(pool.imap(
                        _compute_point, args, chunksize=max(1, n // (ncpu * 4)))):
                    z_vals[i] = z
                    theta_vals[i] = th
                    if (i + 1) % max(1, n // 100) == 0 or i == n - 1:
                        self.progress.emit(i + 1, n)

            wall = time.time() - wall0

            # Derive parametric zeta trace: zeta(1/2+it) = Z(t) * exp(-i*theta)
            zeta_re = z_vals * np.cos(-theta_vals)
            zeta_im = z_vals * np.sin(-theta_vals)

            # S(t) from sign-change counting
            from mpmath import mpf as _mpf, pi as _pi, log as _log
            def N_smooth(t):
                t = _mpf(str(t))
                return float(t / (2*_pi) * _log(t / (2*_pi))
                             - t / (2*_pi) + _mpf('7') / 8)

            N_cur = N_smooth(t_vals[0])
            s_vals = np.empty(n)
            prev_sign = 1 if z_vals[0] >= 0 else -1
            for k in range(n):
                cur_sign = 1 if z_vals[k] >= 0 else -1
                if cur_sign != prev_sign:
                    N_cur += 1
                s_vals[k] = N_cur - N_smooth(t_vals[k])
                prev_sign = cur_sign

            # Find the tightest |Z| extremum
            imin = int(np.argmin(np.abs(z_vals)))

            self.finished.emit({
                't_vals': t_vals, 'z_vals': z_vals,
                'zeta_re': zeta_re, 'zeta_im': zeta_im,
                's_vals': s_vals, 'imin': imin,
                't_center': self.t_center,
                'half_width': self.half_width,
                'step': self.step, 'dps': self.dps,
                'n_points': n, 'n_cores': ncpu,
                'wall_secs': wall,
            })
        except Exception as e:
            self.error.emit(str(e))


# =====================================================================
# Known interesting targets (presets)
# =====================================================================

PRESETS = [
    ("Classic Lehmer pair (t ~ 7005.08)",
     7005.083, 0.3, 0.001, 30),
    ("Tight pair at t ~ 25,500,002,773.72",
     25500002773.7237205505, 0.6, 0.0015, 50),
    ("First zero neighborhood (t ~ 14.13)",
     14.134725, 1.0, 0.002, 30),
    ("Close pair at t ~ 7954.34",
     7954.3362, 0.4, 0.001, 30),
    ("Zeros #4 and #5 (t ~ 30-33)",
     31.7, 2.5, 0.005, 30),
]


# =====================================================================
# Main window
# =====================================================================

class ExplorerUI(QMainWindow):

    CMAP = cm.turbo

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zeta Near-Miss Explorer  (SetiAstro)")
        self.compute_thread = None
        self.last_result = None
        self._build_ui()
        self._load_settings()
        self.statusBar().showMessage("Ready. Enter a t value or pick a preset.")

    # ---- UI construction ------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # -- Top row: config + presets + buttons --------------------------
        top = QHBoxLayout()

        # Config group
        cfg = QGroupBox("Parameters")
        form = QFormLayout(cfg)

        self.tcenter_edit = QDoubleSpinBox()
        self.tcenter_edit.setDecimals(10)
        self.tcenter_edit.setRange(1.0, 1e18)
        self.tcenter_edit.setValue(7005.083)
        self.tcenter_edit.setStepType(
            QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        form.addRow("t center:", self.tcenter_edit)

        self.halfwidth_edit = QDoubleSpinBox()
        self.halfwidth_edit.setDecimals(4)
        self.halfwidth_edit.setRange(0.01, 100.0)
        self.halfwidth_edit.setValue(0.3)
        form.addRow("Half-width:", self.halfwidth_edit)

        self.step_edit = QDoubleSpinBox()
        self.step_edit.setDecimals(6)
        self.step_edit.setRange(0.0001, 1.0)
        self.step_edit.setValue(0.001)
        form.addRow("Step:", self.step_edit)

        self.dps_edit = QSpinBox()
        self.dps_edit.setRange(15, 200)
        self.dps_edit.setValue(30)
        form.addRow("DPS (precision):", self.dps_edit)

        top.addWidget(cfg)

        # Presets + buttons
        right_col = QVBoxLayout()

        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("(custom)")
        for name, *_ in PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        preset_layout.addWidget(self.preset_combo)
        right_col.addWidget(preset_group)

        btn_layout = QHBoxLayout()
        self.btn_compute = QPushButton("Compute")
        self.btn_compute.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px 20px; }")
        self.btn_compute.clicked.connect(self._on_compute)
        btn_layout.addWidget(self.btn_compute)

        self.btn_save = QPushButton("Save PNG")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(self.btn_save)
        right_col.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m  (%p%)")
        right_col.addWidget(self.progress)

        # Info label for results summary
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { color: #ccc; }")
        right_col.addWidget(self.info_label)

        right_col.addStretch()
        top.addLayout(right_col)

        main_layout.addLayout(top)

        # -- Matplotlib canvas --------------------------------------------
        self.fig = Figure(figsize=(16, 10), dpi=100)
        self.fig.patch.set_facecolor('#0b0f1a')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, stretch=1)

        self.resize(1400, 900)

    # ---- Presets --------------------------------------------------------

    def _on_preset(self, idx):
        if idx <= 0:
            return    # "(custom)" selected, do nothing
        _, t_center, half_width, step, dps = PRESETS[idx - 1]
        self.tcenter_edit.setValue(t_center)
        self.halfwidth_edit.setValue(half_width)
        self.step_edit.setValue(step)
        self.dps_edit.setValue(dps)

    # ---- Compute --------------------------------------------------------

    def _on_compute(self):
        if self.compute_thread is not None and self.compute_thread.isRunning():
            QMessageBox.information(self, "Busy",
                "A computation is already running. Please wait.")
            return

        t_center = self.tcenter_edit.value()
        half_width = self.halfwidth_edit.value()
        step = self.step_edit.value()
        dps = self.dps_edit.value()

        n_points = int(2 * half_width / step)
        if n_points > 50000:
            r = QMessageBox.question(self, "Large computation",
                f"This will compute {n_points:,} points, which may take "
                f"a while. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return

        self.btn_compute.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setMaximum(n_points)
        self.info_label.setText(f"Computing {n_points:,} points at dps={dps} "
                                f"using {max(1, cpu_count()-1)} cores...")
        self.statusBar().showMessage("Computing...")

        self.compute_thread = ComputeThread(t_center, half_width, step, dps)
        self.compute_thread.progress.connect(self._on_progress)
        self.compute_thread.finished.connect(self._on_result)
        self.compute_thread.error.connect(self._on_error)
        self.compute_thread.start()

    def _on_progress(self, done, total):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_error(self, msg):
        self.btn_compute.setEnabled(True)
        self.statusBar().showMessage(f"Error: {msg}")
        QMessageBox.critical(self, "Computation error", msg)

    def _on_result(self, result):
        self.last_result = result
        self.btn_compute.setEnabled(True)
        self.btn_save.setEnabled(True)

        n = result['n_points']
        wall = result['wall_secs']
        imin = result['imin']
        z_min = result['z_vals'][imin]
        t_min = result['t_vals'][imin]
        rate = n / wall if wall > 0 else 0

        self.info_label.setText(
            f"Done: {n:,} points in {wall:.1f}s "
            f"({rate:.0f} pts/s, {result['n_cores']} cores)\n"
            f"Tightest |Z|: {abs(z_min):.4e} at "
            f"t = {t_min:.10f}")
        self.statusBar().showMessage(
            f"Computed {n:,} points in {wall:.1f}s. "
            f"Use the toolbar to pan/zoom the plots.")

        self._draw_plot(result)

    # ---- Plotting -------------------------------------------------------

    def _draw_plot(self, r):
        self.fig.clear()

        t_vals  = r['t_vals'];   z_vals  = r['z_vals']
        zeta_re = r['zeta_re'];  zeta_im = r['zeta_im']
        s_vals  = r['s_vals'];   imin    = r['imin']

        # Color fraction along t (0..1)
        cfrac = (t_vals - t_vals[0]) / max(t_vals[-1] - t_vals[0], 1e-30)

        def colored_line(ax, x, y, lw=1.4):
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=self.CMAP, linewidth=lw)
            lc.set_array(cfrac[:-1])
            ax.add_collection(lc)
            return lc

        # Common axis styling
        def style_ax(ax, title, xl='', yl=''):
            ax.set_facecolor('#0b0f1a')
            for s in ax.spines.values():
                s.set_color('#4a5568')
            ax.tick_params(colors='#a0aec0', labelsize=8)
            ax.set_title(title, color='#f0f4f8', fontsize=10, pad=6)
            ax.set_xlabel(xl, color='#e2e8f0', fontsize=9)
            ax.set_ylabel(yl, color='#e2e8f0', fontsize=9)
            ax.grid(True, alpha=0.2, color='#4a5568')

        # ---- Panel 1: Z(t) ----
        ax1 = self.fig.add_subplot(2, 2, 1)
        colored_line(ax1, t_vals, z_vals)
        ax1.axhline(0, color='#4a5568', lw=0.7)
        ax1.scatter([t_vals[imin]], [z_vals[imin]], c='#fc8181', s=50,
                     zorder=5, edgecolors='white', linewidths=0.6)
        ax1.set_xlim(t_vals[0], t_vals[-1])
        z_pad = 0.1 * max(abs(z_vals.min()), abs(z_vals.max())) + 0.01
        ax1.set_ylim(z_vals.min() - z_pad, z_vals.max() + z_pad)
        style_ax(ax1,
                 f"Z(t)    tightest |Z| = {abs(z_vals[imin]):.3e} "
                 f"@ t = {t_vals[imin]:.6f}",
                 yl='Z(t)')

        # Mark zero crossings
        sc_mask = (z_vals[:-1] >= 0) != (z_vals[1:] >= 0)
        sc_idx = np.nonzero(sc_mask)[0]
        for si in sc_idx:
            za, zb = z_vals[si], z_vals[si+1]
            ta, tb = t_vals[si], t_vals[si+1]
            tz = ta - za * (tb - ta) / (zb - za) if zb != za else 0.5*(ta+tb)
            ax1.axvline(tz, color='#f6e05e', alpha=0.5, lw=0.7)
            ax1.plot(tz, 0, 'o', color='#f6e05e', markersize=6,
                      markeredgecolor='#fefcbf', markeredgewidth=0.5, zorder=4)

        # ---- Panel 2: S(t) ----
        ax2 = self.fig.add_subplot(2, 2, 3)
        colored_line(ax2, t_vals, s_vals)
        ax2.axhline(0, color='#4a5568', lw=0.7)
        ax2.axhline(2, color='#fc8181', lw=0.9, ls='--', alpha=0.8)
        ax2.axhline(-2, color='#fc8181', lw=0.9, ls='--', alpha=0.8)
        ax2.text(t_vals[-1], 2.05, '|S| = 2', color='#fc8181', fontsize=8,
                  ha='right', va='bottom')
        ax2.set_xlim(t_vals[0], t_vals[-1])
        s_pad = 0.3
        ax2.set_ylim(s_vals.min() - s_pad, s_vals.max() + s_pad)
        style_ax(ax2, 'S(t) = N_observed - N_smooth', xl='t', yl='S(t)')

        # ---- Panel 3: parametric full ----
        ax3 = self.fig.add_subplot(2, 2, 2)
        colored_line(ax3, zeta_re, zeta_im)
        ax3.plot(0, 0, '+', color='#fbd38d', markersize=14,
                  markeredgewidth=1.5, zorder=3)
        re_pad = 0.1 * max(abs(zeta_re.min()), abs(zeta_re.max())) + 0.01
        im_pad = 0.1 * max(abs(zeta_im.min()), abs(zeta_im.max())) + 0.01
        ax3.set_xlim(zeta_re.min() - re_pad, zeta_re.max() + re_pad)
        ax3.set_ylim(zeta_im.min() - im_pad, zeta_im.max() + im_pad)
        ax3.axhline(0, color='#4a5568', lw=0.4)
        ax3.axvline(0, color='#4a5568', lw=0.4)
        ax3.set_aspect('equal')
        style_ax(ax3, 'zeta(1/2+it) parametric  (color = t)',
                 xl='Re(zeta)', yl='Im(zeta)')

        # ---- Panel 4: parametric zoomed ----
        ax4 = self.fig.add_subplot(2, 2, 4)
        colored_line(ax4, zeta_re, zeta_im)
        ax4.plot(0, 0, '+', color='#fbd38d', markersize=14,
                  markeredgewidth=1.5, zorder=3)
        # Auto-zoom: find the scale of the near-miss region
        # (where |zeta| is smallest) and set limits around that
        min_dist = np.min(np.hypot(zeta_re, zeta_im))
        zoom_radius = max(min_dist * 5, 0.05)
        ax4.set_xlim(-zoom_radius, zoom_radius)
        ax4.set_ylim(-zoom_radius, zoom_radius)
        ax4.axhline(0, color='#4a5568', lw=0.4)
        ax4.axvline(0, color='#4a5568', lw=0.4)
        ax4.set_aspect('equal')
        style_ax(ax4,
                 f'zoomed near origin  (min |zeta| = {min_dist:.4e})',
                 xl='Re(zeta)', yl='Im(zeta)')

        # ---- Colorbar in its own dedicated axes on the far right ----
        # Manually positioned so it never overlaps the plot panels.
        self.fig.subplots_adjust(hspace=0.32, wspace=0.28,
                                  left=0.06, right=0.88,
                                  top=0.94, bottom=0.07)
        cbar_ax = self.fig.add_axes([0.91, 0.07, 0.015, 0.87])
        sm = cm.ScalarMappable(
            cmap=self.CMAP,
            norm=plt.Normalize(vmin=t_vals[0], vmax=t_vals[-1]))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('t (color key across all panels)',
                        color='#a0aec0', fontsize=9)
        cbar.ax.tick_params(colors='#a0aec0', labelsize=8)
        self.canvas.draw()

    # ---- Save -----------------------------------------------------------

    def _on_save(self):
        if self.last_result is None:
            return
        tc = self.last_result['t_center']
        default_name = f"zeta_explorer_t{tc:.0f}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", default_name,
            "PNG files (*.png);;All files (*)")
        if path:
            self.fig.savefig(path, dpi=150, facecolor='#0b0f1a',
                              bbox_inches='tight')
            self.statusBar().showMessage(f"Saved: {path}")

    # ---- Settings persistence -------------------------------------------

    def _load_settings(self):
        s = QSettings("SetiAstro", "ZetaExplorer")
        geo = s.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        tc = s.value("t_center")
        if tc is not None:
            self.tcenter_edit.setValue(float(tc))
        hw = s.value("half_width")
        if hw is not None:
            self.halfwidth_edit.setValue(float(hw))
        st = s.value("step")
        if st is not None:
            self.step_edit.setValue(float(st))
        dps = s.value("dps")
        if dps is not None:
            self.dps_edit.setValue(int(dps))

    def _save_settings(self):
        s = QSettings("SetiAstro", "ZetaExplorer")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("t_center", self.tcenter_edit.value())
        s.setValue("half_width", self.halfwidth_edit.value())
        s.setValue("step", self.step_edit.value())
        s.setValue("dps", self.dps_edit.value())

    def closeEvent(self, event):
        self._save_settings()
        if self.compute_thread and self.compute_thread.isRunning():
            self.compute_thread.wait(3000)
        super().closeEvent(event)


# =====================================================================
# Entry point
# =====================================================================

def main():
    app = QApplication(sys.argv)

    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#16213e"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4fd1c5"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0b0f1a"))
    app.setPalette(palette)

    win = ExplorerUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()