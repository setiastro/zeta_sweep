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

import sys, os, json, time, subprocess, signal, re
from pathlib import Path
from datetime import datetime, timedelta

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QProgressBar, QGroupBox, QFileDialog,
    QMessageBox, QStatusBar, QSplitter
)


# --------------------------------------------------------------------------
# Configuration constants (UI defaults; can be overridden in the UI itself)
# --------------------------------------------------------------------------
SCANNER_SCRIPT_DEFAULT = "zeta_gpu_scan_v6_hp.py"
STATUS_TAG = "@@STATUS@@ "                # must match _STATUS_TAG in scanner
LIVE_OUTPUT_MAX_LINES = 10_000            # scrollback cap in the text pane
PAUSE_FLAG_NAME = "pause.flag"            # created/deleted in cwd


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
                    payload = line[len(STATUS_TAG):]
                    try:
                        event = json.loads(payload)
                        self.statusReceived.emit(event)
                    except json.JSONDecodeError:
                        # malformed status -- log to text pane anyway so the
                        # user sees it, but don't try to route as an event
                        self.lineReceived.emit(line)
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
        self.resize(1100, 800)

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
        root = QVBoxLayout(central)

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
        tp.addWidget(QLabel("t:"), 0, 0)
        self.tp_t_label = QLabel("—")
        self.tp_t_label.setFont(QFont("Consolas", 10))
        tp.addWidget(self.tp_t_label, 0, 1, 1, 3)
        tp.addWidget(QLabel("norm_gap:"), 1, 0)
        self.tp_ngap_label = QLabel("—")
        tp.addWidget(self.tp_ngap_label, 1, 1)
        tp.addWidget(QLabel("gap:"), 1, 2)
        self.tp_gap_label = QLabel("—")
        tp.addWidget(self.tp_gap_label, 1, 3)
        tp.addWidget(QLabel("Z at extremum:"), 2, 0)
        self.tp_z_label = QLabel("—")
        tp.addWidget(self.tp_z_label, 2, 1)
        tp.addWidget(QLabel("kind:"), 2, 2)
        self.tp_kind_label = QLabel("—")
        tp.addWidget(self.tp_kind_label, 2, 3)
        root.addWidget(tp_group)

        # --- Live output pane ---
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
        root.addWidget(out_group, stretch=1)

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
        can_start  = state in (self.STATE_IDLE,)
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

    # ------------------------- Start / Pause / Resume / Abort -------------------------
    def _spawn_scanner(self):
        """Launch the scanner subprocess with current UI config.

        Any override checkbox that's checked adds the corresponding CLI flag;
        anything unchecked lets the scanner use its module default. This
        keeps the default UI launch (all checkboxes off) equivalent to just
        running `python zeta_gpu_scan_v6_hp.py` at the command line.
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

        if self.startchunk_override.isChecked():
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
        self.reader.processExited.connect(self._on_process_exited)
        self.reader.start()

        self.run_start_time = datetime.now()
        self.chunks_completed_this_run = 0
        self._set_state(self.STATE_RUNNING)
        self.statusBar().showMessage(f"Scanner started (PID {self.proc.pid}).")
        return True

    def _on_start(self):
        if self.state != self.STATE_IDLE:
            return
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
        self.output.append("[UI] Resuming from checkpoint…")
        self._spawn_scanner()

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
            tp = event.get("tightest_resume")
            if tp:
                self.tp_t_label.setText(f"{tp.get('t', 0):.6f}")
                self.tp_ngap_label.setText(f"{tp.get('norm_gap', 0):.5f}")
                gap = tp.get('gap')
                self.tp_gap_label.setText(f"{gap:.6e}" if gap else "—")
                self.tp_z_label.setText(f"{tp.get('z', 0):+.3e}")
                self.tp_kind_label.setText(tp.get('kind', '—'))
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
            self.rate_label.setText(f"{event.get('rate_t_per_s', 0.0):.2f} t/s")
            self.refined_label.setText("yes" if event.get("refined") else "no")
            self.zloc_label.setText(f"{event.get('zeros_located_total', 0):,}")
            self.zreq_label.setText(f"{event.get('zeros_required_total', 0):,}")
            self.zshort_label.setText(str(event.get("zeros_short_total", 0)))

            tp = event.get("tightest")
            if tp:
                self.tp_t_label.setText(f"{tp.get('t', 0):.6f}")
                self.tp_ngap_label.setText(f"{tp.get('norm_gap', 0):.5f}")
                gap = tp.get('gap')
                self.tp_gap_label.setText(f"{gap:.6e}" if gap else "—")
                self.tp_z_label.setText(f"{tp.get('z', 0):+.3e}")
                self.tp_kind_label.setText(tp.get('kind', '—'))

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

    def _on_process_exited(self, code: int):
        self.output.append(f"[UI] Scanner exited with code {code}.")
        # If we were pausing and the process exited cleanly, we're paused.
        if self.state == self.STATE_PAUSING:
            self._set_state(self.STATE_PAUSED)
        else:
            # Otherwise (abort, natural end, crash), go back to idle.
            self._set_state(self.STATE_IDLE)
            self.statusBar().showMessage(f"Scanner exited (code {code}). Ready.")
        self.proc = None
        # Reader thread will finish on its own; nothing to clean up here.

    # ------------------------- Elapsed / ETA ticker -------------------------
    def _update_elapsed(self):
        if self.run_start_time is None or self.state == self.STATE_IDLE:
            self.elapsed_label.setText("—")
            self.eta_label.setText("—")
            return
        elapsed = datetime.now() - self.run_start_time
        self.elapsed_label.setText(str(elapsed).split(".")[0])
        # ETA: if we have completed at least one chunk this run, extrapolate
        # from the average chunk wall time; otherwise unknown.
        if (self.chunks_completed_this_run > 0
            and self.total_chunks_target > 0):
            remaining_chunks = (self.total_chunks_target
                                - (self.first_chunk_this_run + self.chunks_completed_this_run))
            if remaining_chunks > 0:
                avg_per_chunk = elapsed.total_seconds() / self.chunks_completed_this_run
                eta_secs = int(avg_per_chunk * remaining_chunks)
                self.eta_label.setText(str(timedelta(seconds=eta_secs)))
            else:
                self.eta_label.setText("done")
        else:
            self.eta_label.setText("—")

    # ------------------------- Shutdown safety -------------------------
    def closeEvent(self, event):
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
    win = ScannerUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()