import logging, os, traceback
from datetime import datetime, timedelta

import pyaudio
from six import print_
from six.moves import queue
import numpy as np
from dragonfly import get_engine, Grammar, Mouse

_log = logging.getLogger(os.path.basename(__file__))


################################################################################################################################################################

def _safely_func(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
    return wrapper

def _safely(*funcs):
    ret = None
    for func in funcs:
        try:
            ret = func()
        except Exception as e:
            traceback.print_exc()
    return ret

class State(object):
    """docstring for State"""

    instances = []

    def __init__(self, timeout=None, lockout=None, lockout_oneway=True, hi_trig=None, lo_trig=None):
        """
        timeout: time in seconds after last activation, that state is deactivated
        lockout: time in seconds after last activation, during which state cannot be activated (or deactivated if lockout_oneway=False)
        lockout_oneway: if True, only lockout from activating again after activation; otherwise lockout from activating or deactivating (after activation)
        """
        self.timeout = timeout
        self.lockout = lockout
        self.lockout_oneway = lockout_oneway
        self.hi_trig = hi_trig    # only called when state is set/checked/polled!
        self.lo_trig = lo_trig    # only called when state is set/checked/polled!
        self._state = False
        self.timeout_time = None
        self.lockout_time = None
        State.instances.append(self)

    @property
    def state(self):
        if self.timeout_time and datetime.today() >= self.timeout_time:
            self.state = False
            self.timeout_time = None
        return self._state

    @state.setter
    def state(self, value):
        # State.state.fset(s, value)
        # State.state.__set__(s, value)
        if self.lockout_time and (not self.lockout_oneway or value):
            if datetime.today() < self.lockout_time:
                return
            else:
                self.lockout_time = None
        if bool(self._state) != bool(value):
            if value:
                if self.hi_trig: _safely(lambda: self.hi_trig())
            else:
                if self.lo_trig: _safely(lambda: self.lo_trig())
        self._state = value
        if value:
            if self.timeout:
                self.timeout_time = datetime.today() + timedelta(seconds=self.timeout)
            if self.lockout:
                self.lockout_time = datetime.today() + timedelta(seconds=self.lockout)
        else:
            self.timeout_time = None

    def __nonzero__(self):
        return bool(self.state)

    def set(self, value=True):
        initial_value = self.state
        self.state = value
        return (initial_value != self.state)
    def activate(self, force=False):
        return self.set(True)
    def deactivate(self, force=False):
        return self.set(False)


################################################################################################################################################################

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""
    # Date: ???

    FORMAT = pyaudio.paInt16
    RATE = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, buffer_s=0, flush_queue=True, start=True, input_device_index=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data, block=False)
        self.sample_rate = self.RATE
        self.flush_queue = flush_queue
        self.buffer_queue = queue.Queue(maxsize=(buffer_s * 1000 // self.block_duration_ms))
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.block_size,
            stream_callback=proxy_callback,
            input_device_index=input_device_index,
            start=bool(start),
        )
        self.active = True
        # _log.info("%s: streaming audio from microphone: %i sample_rate, %i block_duration_ms", self, self.sample_rate, self.block_duration_ms)

    block_size = property(lambda self: int(self.sample_rate / float(self.BLOCKS_PER_SECOND)))
    block_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        self.active = False

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    def read(self, realtime=False, blocking=True):
        """Return a block of audio data, blocking if necessary. If realtime, discard old blocks and return the most recent block."""
        block = None
        iters = 0
        while (blocking and block is None) or (realtime and not self.buffer_queue.empty()):
            if blocking:
                block = self.buffer_queue.get()
            else:
                try:
                    block = self.buffer_queue.get_nowait()
                except queue.Empty:
                    block = None
            iters += 1
        if realtime and iters > 1: _log.warning("dropped %d blocks to maintain realtime", iters-1)
        return block

    def read_loop(self, callback, realtime=False, flush_queue=True):
        """Block looping reading, repeatedly passing a block of audio data to callback."""
        while self.active or (flush_queue and not self.buffer_queue.empty()):
            block = self.read(realtime=realtime)
            callback(block)

    def read_coro(self, callback, enabled_func=None, action=None):
        # if not state.pa: state.pa = pyaudio.PyAudio()
        # if not state.stream:
        #     state.stream = state.pa.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        #     state.stream.start_stream()
        # if state.stream.is_stopped(): state.stream.start_stream()
        try:
            # state.context = get_context()
            # if action: state.action = action
            while True:
                # enabled = not bool(dfly_state.sleeping)
                enabled = (enabled_func is None) or _safely(lambda: bool(enabled_func()))
                if not self.buffer_queue.empty():
                    block = self.read()
                    callback(block)
                    # scroll_func()
                yield
        except GeneratorExit:
            _log.debug("read_coro: closed")
        finally:
            # state.stream.stop_stream()
            # if 0:
            #     state.stream.close()
            #     state.pa.terminate()
            # if state.active:
            #     do_action(False)
            # state.context = None
            pass

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

Audio.RATE = 44100
Audio.BLOCKS_PER_SECOND = 10


################################################################################################################################################################

class Processor(object):
    def __init__(self, audio, recognizer, length_s=None):
        """spectrogram[<#block>, <frequency/10>]"""
        length_s = length_s or 5
        self.audio = audio
        self.recognizer = recognizer
        self.window = np.hamming(self.audio.block_size)
        self.spectrum_scale = 2 * np.sum(self.window)
        self.K = 94
        self.freqs = np.fft.rfftfreq(self.audio.block_size, 1./self.audio.RATE)
        self.SPECTROGRAM_LEN = length_s * self.audio.BLOCKS_PER_SECOND
        self.priming = self.SPECTROGRAM_LEN
        # self.history_data = np.zeros(self.SPECTROGRAM_LEN * self.audio.block_size, dtype=np.int16)
        self.spectrogram = np.zeros((self.SPECTROGRAM_LEN, len(self.freqs)))

        # self.ivaps_state = mode.State(hi_trig=lambda: _log.info("ivaps went hi!"), lo_trig=lambda: _log.info("ivaps went lo!"))
        # self.humming_state = mode.State(hi_trig=lambda: _log.info("humming went hi!"), lo_trig=lambda: _log.info("humming went lo!"))
        # self.humming_freqs = np.full(self.HUMMING_LEN, np.inf)
        # self.humming_gaussians = np.tile(np.array([(0, np.inf, np.inf)], dtype=[('amplitude', 'f8'), ('freq', 'f8'), ('sigma', 'f8')]), self.HUMMING_LEN)
        # self.talking_gaussians = np.tile(np.array([(0, np.inf, np.inf)], dtype=[('amplitude', 'f8'), ('freq', 'f8'), ('sigma', 'f8')]), self.HUMMING_LEN)

    @_safely_func
    def callback(self, waveform):
        data = np.fromstring(waveform, dtype=np.int16)
        # self.history_data = np.hstack((self.history_data[self.audio.block_size:], data))
        data = data.astype(np.float64) / 32767 * self.window
        self.spectrum = np.abs(np.fft.rfft(data))
        self.spectrum *= self.spectrum_scale
        self.spectrum = 20 * np.log10(self.spectrum)
        # self.spectrum += self.K
        self.spectrogram = np.vstack((self.spectrogram[1:], self.spectrum))

        # plot_spectrum(self.freqs, self.spectrum)

        if self.priming:
            self.priming -= 1
            if not self.priming: _log.info("Primed!")
            else: return

        _safely(lambda: self.recognizer(self.spectrogram))

        return


def plot_spectrum(freqs, spectrum, spectrum2=None, logx=True, logy=False, ylim=None, xy=None):
    import matplotlib.pyplot as plt
    global plot_spectrum_lines
    lines = [spectrum] + ([spectrum2] if spectrum2 is not None else [])
    # plt.cla()
    if plot_spectrum_lines is None:
        if logx: plt.xscale('log')
        if logy: plt.yscale('log')
        if ylim: plt.ylim(ylim)
        plt.grid(True)
        plot_spectrum_lines = []
    else:
        while plot_spectrum_lines: plot_spectrum_lines.pop(0).remove()
    for i, line in enumerate(lines):
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        plot_line, = plt.plot(freqs, line, colors[i])
        plot_spectrum_lines.append(plot_line)
    if xy:
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+%d+%d" % xy)
    plt.pause(0.1)
plot_spectrum_lines = None

def setup_show_spectrum():
    def timer_func():
        block = audio.read(realtime=True, blocking=False)
        if block is not None:
            processor.callback(block)
    global timer
    audio = Audio()
    recognizer = lambda spectrogram: plot_spectrum(processor.freqs, spectrogram[-1])
    processor = Processor(audio, recognizer, length_s=1)
    timer = get_engine().create_timer(timer_func, 0.02)


################################################################################################################################################################

def test_recognizer(spectrogram):
    print_(spectrogram)

def make_hmm_recognizer(action=None, freq_min=100, freq_max=200, db_min=100, peak_freq_width=40, peak_db_height=30, ms_before_action=300):
    def recognizer(spectrogram):
        freq_idxs = spectrogram.argmax(axis=1)
        amps = spectrogram[np.ix_(range(len(freq_idxs)))[0], freq_idxs].astype(int)
        freqs = freq_idxs * 10
        steps_before_action = int(float(ms_before_action) / 1000 * Audio.BLOCKS_PER_SECOND)
        state_sustain = (abs(freqs[-1] - freqs[-2]) <= 10) and (db_min <= amps[-1])
        peak_freq_idx_radius = int(peak_freq_width / 20)
        # state_sustain = (abs(freqs[-1] - freqs[-2]) <= 10) and (db_min <= amps[-1]) and np.all(spectrogram[..., ()])
        # mean_freq = freqs[-steps_before_action:].mean()
        # state_sustain = np.all(np.abs(freqs[-steps_before_action:] - mean_freq) <= 10) and (db_min <= amps[-1])
        # state_start = state_sustain and np.all(freq_min <= freqs[-steps_before_action:]) and np.all(freqs[-steps_before_action:] <= freq_max) and np.all(db_min <= amps[-steps_before_action:])
        state_start = all([
            state_sustain,
            np.all(freq_min < freqs[-steps_before_action:]),
            np.all(freqs[-steps_before_action:] < freq_max),
            np.all(np.abs(freqs[-steps_before_action:] - freqs[-steps_before_action:].mean()) <= 10),
            np.all(db_min < amps[-steps_before_action:]),
            np.all(amps[:-steps_before_action] < db_min),
            # np.all(spectrogram[-steps_before_action:, (freq_idxs[-steps_before_action:].min()-peak_freq_idx_radius, freq_idxs[-steps_before_action:].max()+peak_freq_idx_radius)] < amps[-steps_before_action:]),
            np.all(spectrogram[-steps_before_action:, freq_idxs[-steps_before_action:].min()-peak_freq_idx_radius] < amps[-steps_before_action:]),
            np.all(spectrogram[-steps_before_action:, freq_idxs[-steps_before_action:].max()-peak_freq_idx_radius] < amps[-steps_before_action:]),
        ])
        # print_(bool(state), state_start, state_sustain)
        state.set(bool((not state and state_start) or (state and state_sustain)))
        # if state: from IPython import embed; embed()
        if state and action:
            action.execute()
    state = State(lockout=0.5)
    state = State(lockout=0.5, hi_trig=lambda: print_("recognizer_state hi!"), lo_trig=lambda: print_("recognizer_state lo!"))
    return recognizer

def setup():
    def timer_func():
        block = audio.read(realtime=True, blocking=False)
        # print_(len(block) if block is not None else None)
        if block is not None:
            processor.callback(block)
    global timer
    audio = Audio()
    recognizer = make_hmm_recognizer(action=Mouse("wheeldown"))
    processor = Processor(audio, recognizer, length_s=1)
    timer = get_engine().create_timer(timer_func, 0.02)

################################################################################################################################################################

# setup_show_spectrum()
setup()

grammar = Grammar("noise_recognition")
# grammar.add_rule()
grammar.load()
