import pyaudio
import wave
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import welch, hanning
import matplotlib.pyplot as plt
from argparse import ArgumentParser

class AudioFile(object):

    def __init__(self, filename, chunk=1024, output_device=None):
        FORMAT = pyaudio.paInt16
        self.filename = filename
        self.wavfile = wave.open(self.filename, 'rb')

        self.pyaudio = pyaudio.PyAudio()

        self.format = self.pyaudio.get_format_from_width(self.wavfile.getsampwidth())
        self.sample_rate = self.wavfile.getframerate()
        self.channels = self.wavfile.getnchannels()
        self.chunk = chunk
        self.window = hanning(self.chunk, True)

        if output_device is None:
            self.output_device_index = self.pyaudio.get_default_output_device_info()['index']
        else:
            self.output_device_index = output_device

        self.stream = self.pyaudio.open(
            rate=self.sample_rate,
            format=self.format,
            channels=self.channels,
            output=True,
            output_device_index=self.output_device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback)
        self.data = np.zeros(self.chunk)

    def callback(self, in_data, frame_count, time_info, status):
        data = self.wavfile.readframes(frame_count)
        self.data = self.str_to_float(data)
        return (data, pyaudio.paContinue)

    def str_to_float(self, data):
        MAX_INT = 32768.0
        return np.fromstring(data, dtype=np.int16) / MAX_INT

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

    def psd(self):
        f, Pxxf = welch(
            self.data,
            self.sample_rate,
            nperseg=self.chunk,
            return_onesided=True, 
            detrend=None)
        dbhz = 10*np.log10(Pxxf)
        return f, dbhz

    def fft(self):
        f = fftfreq(self.chunk, 1.0/self.sample_rate)[:self.chunk//2]
        amp = fft(self.window * self.data)
        amp = np.abs(amp[:self.chunk//2]) * 2 / (self.chunk)
        return f, amp

def main(args):

    audio = AudioFile(args.filename, chunk=args.chunk, output_device=args.output_device)
    audio.start()

    fig, (ax_t, ax_f, ax_p) = plt.subplots(3, figsize=(8,6))

    # fake data
    xt = np.arange(0, audio.chunk)
    yt = y_p = np.zeros(audio.chunk)
    xf = fftfreq(audio.chunk, 1.0/audio.sample_rate)
    xf, yf = audio.fft()
    xp, yp = audio.psd()

    # xt
    line_t, = ax_t.plot(xt, yt, 'b-', lw=2)
    ax_t.set_title('Audio Waveform')
    ax_t.set_xlabel('samples')
    ax_t.set_ylabel('volume')
    ax_t.set_ylim(-1, 1)

    # fft
    line_f, = ax_f.plot(xf, yf, 'r-', lw=1)
    ax_f.set_title('FFT')
    ax_f.set_xlabel('frequency')
    ax_f.set_ylabel('Amplitude')
    ax_f.set_xlim(20, audio.sample_rate/2)
    ax_f.set_ylim(0, 0.01)

    # psd
    line_p, = ax_p.plot(xp, yp, 'r-', lw=1)
    ax_p.set_title('PSD')
    ax_p.set_xlabel('frequency')
    ax_p.set_ylabel('dB/Hz')
    ax_p.set_xlim(20, audio.sample_rate/2)
    ax_p.set_ylim(-150, -10)

    plt.show(block=False)

    try:
        while audio.stream.is_active():
            # time domain
            line_t.set_ydata(audio.data)

            # frequency domain
            _, yf = audio.fft()
            line_f.set_ydata(yf)

            # psd
            _, yp = audio.psd()
            line_p.set_ydata(yp)

            fig.canvas.draw()
            fig.canvas.flush_events()
            pass
    except KeyboardInterrupt:
        audio.stop()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename', help='file name', type=str)
    parser.add_argument('-o', '--output_device', help='output device #', type=int)
    parser.add_argument('--chunk', help='chunk size', type=int, default=1024*4)

    args = parser.parse_args()
    print(args)

    main(args)
