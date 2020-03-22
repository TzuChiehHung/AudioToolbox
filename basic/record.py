import pyaudio
import wave
from argparse import ArgumentParser

def main(args):

    def callback(in_data, frame_count, time_info, status):
        frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    FORMAT = pyaudio.paInt16

    p = pyaudio.PyAudio()

    if args.input is None:
        args.input = p.get_default_input_device_info()['index']

    stream = p.open(
        rate=args.rate,
        format=FORMAT,
        channels=args.channels,
        frames_per_buffer=args.chunk,
        input=True,
        input_device_index=args.input,
        stream_callback=callback)

    frames = []

    print('start recording')
    print('press Ctrl-C to stop recording')
    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print('done recording')
        stream.stop_stream()
        stream.close()
        p.terminate()

    wf = wave.open(args.filename, 'wb')
    wf.setnchannels(args.channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(args.rate)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename', help='file name', type=str, default='output.wav')
    parser.add_argument('-c', '--channels', help='channels', type=int, default=1)
    parser.add_argument('-r', '--rate', help='sample rate', type=int, default=44100)
    parser.add_argument('-i', '--input', help='input device #', type=int)
    parser.add_argument('--chunk', help='chunk size', type=int, default=1024)

    args = parser.parse_args()

    main(args)
