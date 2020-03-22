import pyaudio
import wave
from argparse import ArgumentParser

def main(args):

    def callback(in_data, frame_count, time_info, status):
        return (in_data, pyaudio.paContinue)

    FORMAT = pyaudio.paInt16

    p = pyaudio.PyAudio()

    if args.input is None:
        args.input = p.get_default_input_device_info()['index']
    if args.output is None:
        args.output = p.get_default_output_device_info()['index']

    stream = p.open(
        rate=args.rate,
        format=FORMAT,
        channels=args.channels,
        input=True,
        input_device_index=args.input,
        output=True,
        output_device_index=args.output,
        frames_per_buffer=args.chunk,
        stream_callback=callback)

    stream.start_stream()

    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--channels', help='channels', type=int, default=1)
    parser.add_argument('-r', '--rate', help='sample rate', type=int, default=44100)
    parser.add_argument('-i', '--input', help='input device #', type=int)
    parser.add_argument('-o', '--output', help='output device #', type=int)
    parser.add_argument('--chunk', help='chunk size', type=int, default=1024)

    args = parser.parse_args()

    main(args)
