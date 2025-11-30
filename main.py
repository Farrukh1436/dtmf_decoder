#!/usr/bin/env python3


# Usage:
#    python main.py (audio_file.wav)
#    python main.py --no-plots (audio_file.wav)  # run without visualization
#    python main.py --help                       # show help

# imports
import sys
import os
import argparse

from dtmf_decoder import DTMFDecoder
from config import DTMFConfig
from dtmf_mapping import DTMFMapper
from visualization import create_dtmf_reference_plot


def main():
    """Main entry point for the DTMF decoder."""
    parser = argparse.ArgumentParser(
        description="DTMF Signal Decoder - Decode phone numbers from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py 1.wav              # Decode with visualization
  python main.py --no-plots 1.wav   # Decode without plots
  python main.py --show-table        # Show DTMF frequency table
"""
    )
    
    parser.add_argument( 'audio_file', 
        nargs='?', 
        default='1.wav',
        help='Path to WAV audio file (default: 1.wav)'
    )
    parser.add_argument( '--no-plots', 
        action='store_true',
        help='Disable visualization plots for faster processing'
    )
    parser.add_argument( '--show-table', 
        action='store_true',
        help='Display DTMF frequency reference table'
    )
    parser.add_argument( '--config', 
        help='Show current configuration settings'
    )
    args = parser.parse_args()
    
    # Show DTMF table if requested
    if args.show_table:
        mapper = DTMFMapper()
        mapper.display_dtmf_table()
        if args.audio_file == '1.wav' and len(sys.argv) == 2:  # Only --show-table was specified
            create_dtmf_reference_plot()
            return
    
    # Show configuration if requested
    if args.config:
        show_configuration()
        if args.audio_file == '1.wav' and len(sys.argv) == 2:  # Only --config was specified
            return
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"[ERROR] Audio file not found: {args.audio_file}")
        print("\nAvailable files in current directory:")
        for file in os.listdir('.'):
            if file.endswith(('.wav', '.ogg', '.mp3')):
                print(f"  - {file}")
        return 1
    
    # Print banner
    print_banner()
    
    # Initialize decoder
    enable_plotting = not args.no_plots
    decoder = DTMFDecoder(enable_plotting=enable_plotting)
    
    # Decode the audio file
    try:
        phone_number = decoder.decode_audio_file(args.audio_file)
        
        if phone_number:
            print(f"\n🎯 SUCCESS: Decoded phone number: {phone_number}")
        else:
            print("\n❌ No valid DTMF sequence detected")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Decoding interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR during decoding: {e}")
        return 1
    
    return 0

def print_banner():
    #print application banner.
    print("="*60)
    print("         🕵️  DTMF SIGNAL DECODER - TEAM 4  🕵️")
    print("          Secret Agent Phone Number Decoder")
    print("="*60)
    print()

def show_configuration():
    # display current configuration settings.
    config = DTMFConfig()
    
    print("\n📋 Current Configuration:")
    print("-" * 40)
    print(f"Frequency Tolerance:     ±{config.FREQ_TOLERANCE} Hz")
    print(f"Min Signal Duration:     {config.MIN_SIGNAL_DURATION_MS} ms")
    print(f"Min Silence Duration:    {config.MIN_SILENCE_DURATION_MS} ms")
    print(f"Duplicate Gap Threshold: {config.MAX_SAME_DIGIT_GAP_MS} ms")
    print(f"Frame Size:              {config.FRAME_SIZE_MS} ms")
    print(f"Hop Size:                {config.HOP_SIZE_MS} ms")
    print(f"Min SNR Ratio:           {config.MIN_SNR_RATIO}:1")
    print(f"FFT Size:                {config.MIN_FFT_SIZE} samples")
    print("-" * 40)

def quick_decode(file_path: str, show_plots: bool = False) -> str:
 
    decoder = DTMFDecoder(enable_plotting=show_plots)
    return decoder.decode_audio_file(file_path)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)