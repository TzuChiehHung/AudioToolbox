import pyaudio
import wave
from argparse import ArgumentParser

def main(args):

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)

    wf = wave.open(args.filename, 'rb')
    p = pyaudio.PyAudio()

    if args.output is None:
        args.output = p.get_default_output_device_info()['index']

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
        output_device_index=args.output,
        stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        pass

    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename', help='file name', type=str)
    parser.add_argument('-o', '--output', help='output device #', type=int)

    args = parser.parse_args()

    main(args)
