#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from libs import AudioFileStream, AudioVisualization

def main(args):

    audio = AudioFileStream(
        args.filename,
        chunk=args.chunk,
        output_device=args.output_device)
    audio.start()

    if args.visualize:
        vis = AudioVisualization(audio)
        vis.show()
    else:
        try:
            while audio.stream.is_active():
                pass
        except KeyboardInterrupt:
            audio.stop()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename', help='file name', type=str)
    parser.add_argument('-o', '--output_device', help='output device #', type=int)
    parser.add_argument('--chunk', help='chunk size', type=int, default=1024*4)
    parser.add_argument('-v', '--visualize', action='store_true')

    args = parser.parse_args()
    print(args)

    main(args)
