"""DTMF Frequency Mapping and Character Lookup"""

from typing import Dict, Tuple, Optional, List
from config import DTMFConfig

class DTMFMapper:
    """
    Handles DTMF frequency to character mapping and frequency matching.
    """
    
    def __init__(self):
        self.dtmf_map = self._build_dtmf_map()
    
    def _build_dtmf_map(self) -> Dict[Tuple[int, int], str]:
        """
        Build the frequency pair to character mapping dictionary.
        
        Returns:
            Dictionary mapping (low_freq, high_freq) to character
        """
        # Standard telephone keypad layout
        keypad = [
            ['1', '2', '3', 'A'],
            ['4', '5', '6', 'B'],
            ['7', '8', '9', 'C'],
            ['*', '0', '#', 'D']
        ]
        
        mapping = {}
        for row, low_freq in enumerate(DTMFConfig.LOW_FREQS):
            for col, high_freq in enumerate(DTMFConfig.HIGH_FREQS):
                character = keypad[row][col]
                mapping[(low_freq, high_freq)] = character
        
        return mapping
    
    def find_nearest_frequency(self, target: float, valid_freqs: List[int]) -> Optional[int]:
        """
        Find the nearest valid DTMF frequency within tolerance.
        
        Args:
            target: Detected frequency from FFT
            valid_freqs: List of valid DTMF frequencies
            
        Returns:
            Nearest valid frequency or None if outside tolerance
        """
        best_match = None
        min_diff = float('inf')
        
        for freq in valid_freqs:
            diff = abs(target - freq)
            if diff <= DTMFConfig.FREQ_TOLERANCE and diff < min_diff:
                min_diff = diff
                best_match = freq
        
        return best_match
    
    def decode_frequency_pair(self, low_freq: float, high_freq: float) -> Optional[str]:
        """
        Decode a pair of detected frequencies to a DTMF character.
        
        Args:
            low_freq: Detected low frequency
            high_freq: Detected high frequency
            
        Returns:
            Decoded character or None if no valid mapping
        """
        # Find nearest valid frequencies
        matched_low = self.find_nearest_frequency(low_freq, DTMFConfig.LOW_FREQS)
        matched_high = self.find_nearest_frequency(high_freq, DTMFConfig.HIGH_FREQS)
        
        if matched_low is not None and matched_high is not None:
            return self.dtmf_map.get((matched_low, matched_high))
        
        return None
    
    def get_frequency_info(self, character: str) -> Optional[Tuple[int, int]]:
        """
        Get the frequency pair for a given DTMF character.
        
        Args:
            character: DTMF character (0-9, A-D, *, #)
            
        Returns:
            Tuple of (low_freq, high_freq) or None if invalid character
        """
        for freq_pair, char in self.dtmf_map.items():
            if char == character:
                return freq_pair
        return None
    
    def display_dtmf_table(self):
        """
        Print the DTMF frequency table in a readable format.
        """
        print("\nDTMF Frequency Table:")
        print("        1209 Hz  1336 Hz  1477 Hz  1633 Hz")
        
        keypad = [
            ['1', '2', '3', 'A'],
            ['4', '5', '6', 'B'],
            ['7', '8', '9', 'C'],
            ['*', '0', '#', 'D']
        ]
        
        for row, low_freq in enumerate(DTMFConfig.LOW_FREQS):
            print(f"{low_freq} Hz    {keypad[row][0]}        {keypad[row][1]}        {keypad[row][2]}        {keypad[row][3]}")
        print()
