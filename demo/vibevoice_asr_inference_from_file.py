#!/usr/bin/env python
"""
VibeVoice ASR Batch Inference Demo Script

This script supports batch inference for ASR model and compares results
between batch processing and single-sample processing.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import json
import re
from typing import List, Dict, Any, Optional, Union
from functools import wraps

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


class VibeVoiceASRBatchInference:
    """Batch inference wrapper for VibeVoice ASR model."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda", 
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa"
    ):
        """
        Initialize the ASR batch inference pipeline.
        
        Args:
            model_path: Path to the pretrained model
            device: Device to run inference on (cuda, mps, xpu, cpu, auto)
            dtype: Data type for model weights
            attn_implementation: Attention implementation to use ('flash_attention_2', 'sdpa', 'eager')
        """
        print(f"Loading VibeVoice ASR model from {model_path}")
        
        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )
        
        # Load model with specified attention implementation
        print(f"Using attention implementation: {attn_implementation}")
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model = self.model.to(device)
        
        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.dtype = dtype
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _prepare_generation_config(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> dict:
        """Prepare generation configuration."""
        config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Beam search vs sampling
        if num_beams > 1:
            config["num_beams"] = num_beams
            config["do_sample"] = False  # Beam search doesn't use sampling
        else:
            config["do_sample"] = do_sample
            # Only set temperature and top_p when sampling is enabled
            if do_sample:
                config["temperature"] = temperature
                config["top_p"] = top_p
        
        return config
    
    def transcribe_batch(
        self,
        audio_inputs: List,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files/arrays in a single batch.
        
        Args:
            audio_inputs: List of audio file paths or (array, sampling_rate) tuples
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling
            
        Returns:
            List of transcription results
        """
        if len(audio_inputs) == 0:
            return []

        batch_size = len(audio_inputs)
        print(f"\nProcessing batch of {batch_size} audio(s)...")
        
        # Process all audio together
        inputs = self.processor(
            audio=audio_inputs,
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Print batch info
        print(f"  Input IDs shape: {inputs['input_ids'].shape}")
        print(f"  Speech tensors shape: {inputs['speech_tensors'].shape}")
        print(f"  Attention mask shape: {inputs['attention_mask'].shape}")
        
        # Generate
        generation_config = self._prepare_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode outputs for each sample in the batch
        results = []
        input_length = inputs['input_ids'].shape[1]
        
        for i, audio_input in enumerate(audio_inputs):
            # Get generated tokens for this sample (excluding input tokens)
            generated_ids = output_ids[i, input_length:]
            
            # Remove padding tokens from the end
            # Find the first eos_token or pad_token
            eos_positions = (generated_ids == self.processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[:eos_positions[0] + 1]
            
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Parse structured output
            try:
                transcription_segments = self.processor.post_process_transcription(generated_text)
            except Exception as e:
                print(f"Warning: Failed to parse structured output: {e}")
                transcription_segments = []
            
            # Get file name based on input type
            if isinstance(audio_input, str):
                file_name = audio_input
            elif isinstance(audio_input, dict) and 'id' in audio_input:
                file_name = audio_input['id']
            else:
                file_name = f"audio_{i}"
            
            results.append({
                "file": file_name,
                "raw_text": generated_text,
                "segments": transcription_segments,
                "generation_time": generation_time / batch_size,
            })
        
        print(f"  Total generation time: {generation_time:.2f}s")
        print(f"  Average time per sample: {generation_time/batch_size:.2f}s")
        
        return results
    
    def transcribe_with_batching(
        self,
        audio_inputs: List,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files/arrays with automatic batching.
        
        Args:
            audio_inputs: List of audio file paths or (array, sampling_rate) tuples
            batch_size: Number of samples per batch
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling
            
        Returns:
            List of transcription results
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(audio_inputs), batch_size):
            batch_inputs = audio_inputs[i:i + batch_size]
            print(f"\n{'='*60}")
            print(f"Processing batch {i//batch_size + 1}/{(len(audio_inputs) + batch_size - 1)//batch_size}")
            
            batch_results = self.transcribe_batch(
                batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
            )
            all_results.extend(batch_results)
        
        return all_results

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _shift_segments(self, segments: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
        shifted = []
        for seg in segments:
            curr = dict(seg)
            for key in ["start_time", "end_time", "start", "end"]:
                if key in curr:
                    numeric = self._safe_float(curr[key])
                    if numeric is not None:
                        curr[key] = numeric + offset
            shifted.append(curr)
        return shifted

    def transcribe_with_vad(
        self,
        audio_inputs: List,
        vad_runner,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> List[Dict[str, Any]]:
        all_results = []
        for i, audio_input in enumerate(audio_inputs):
            if isinstance(audio_input, str):
                file_name = audio_input
                audio_data = vad_runner._load_audio(audio_input)
            elif isinstance(audio_input, dict) and "array" in audio_input:
                file_name = audio_input.get("id", f"dataset_audio_{i}")
                audio_data = audio_input["array"]
            else:
                file_name = f"audio_{i}"
                audio_data = audio_input

            vad_segments = vad_runner.detect(audio_input if isinstance(audio_input, str) else audio_data)
            if not vad_segments:
                all_results.append(
                    {
                        "file": file_name,
                        "raw_text": "",
                        "segments": [],
                        "generation_time": 0.0,
                        "vad_segments": [],
                        "chunks": [],
                    }
                )
                continue

            chunk_inputs = []
            for seg in vad_segments:
                f1 = int(seg["start"] * vad_runner._sample_rate)
                f2 = int(seg["end"] * vad_runner._sample_rate)
                # chunk_inputs.append((audio_data[f1:f2], vad_runner._sample_rate))
                chunk_inputs.append(audio_data[f1:f2])

            chunk_results = self.transcribe_with_batching(
                chunk_inputs,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
            )

            merged_text_parts = []
            merged_segments = []
            merged_chunks = []
            generation_time = 0.0
            import pdb;pdb.set_trace()
            for idx, chunk_result in enumerate(chunk_results):
                seg_meta = vad_segments[idx]
                start_offset = float(seg_meta["start"])
                shifted_segments = self._shift_segments(chunk_result.get("segments", []), start_offset)
                merged_segments.extend(shifted_segments)
                raw_text = chunk_result.get("raw_text", "")
                if raw_text:
                    merged_text_parts.append(raw_text)
                generation_time += float(chunk_result.get("generation_time", 0.0))
                merged_chunks.append(
                    {
                        "start": seg_meta["start"],
                        "end": seg_meta["end"],
                        "raw_text": raw_text,
                        "segments": shifted_segments,
                    }
                )

            all_results.append(
                {
                    "file": file_name,
                    "raw_text": "\n".join(merged_text_parts),
                    "segments": merged_segments,
                    "generation_time": generation_time,
                    "vad_segments": vad_segments,
                    "chunks": merged_chunks,
                }
            )
        return all_results


class WhisperXPhase1VAD:
    def __init__(
        self,
        vad_method: str = "pyannote",
        vad_onset: float = 0.500,
        vad_offset: float = 0.363,
        chunk_size: int = 30,
        device: str = "cpu",
        device_index: int = 1,
    ):
        from whisperx.audio import load_audio, SAMPLE_RATE
        from whisperx.vads import Vad, Pyannote, Silero

        self._load_audio = load_audio
        self._sample_rate = SAMPLE_RATE
        self._Vad = Vad
        self._Pyannote = Pyannote
        self._vad_params = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
            "chunk_size": chunk_size,
        }

        if vad_method == "silero":
            self.vad_model = Silero(**self._vad_params)
        elif vad_method == "pyannote":
            device_vad = f"cuda:{device_index}" if (device == "cuda" or device == "auto") else device
            self.vad_model = Pyannote(torch.device(device_vad), token=None, **self._vad_params)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    def detect(self, audio: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        if isinstance(audio, str):
            audio = self._load_audio(audio)

        if issubclass(type(self.vad_model), self._Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = self._Pyannote.preprocess_audio(audio)
            merge_chunks = self._Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": self._sample_rate})
        vad_segments = merge_chunks(
            vad_segments,
            self._vad_params["chunk_size"],
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        return vad_segments


def run_phase1_vad_only(audio_inputs: List[Any], args) -> List[Dict[str, Any]]:
    runner = WhisperXPhase1VAD(
        vad_method=args.vad_method,
        vad_onset=args.vad_onset,
        vad_offset=args.vad_offset,
        chunk_size=args.chunk_size,
        device=args.device,
        device_index=args.vad_device_index,
    )

    results = []
    for i, audio_input in enumerate(audio_inputs):
        if isinstance(audio_input, str):
            file_name = audio_input
            audio_data = audio_input
        elif isinstance(audio_input, dict) and "array" in audio_input:
            file_name = audio_input.get("id", f"dataset_audio_{i}")
            audio_data = audio_input["array"]
        else:
            file_name = f"audio_{i}"
            audio_data = audio_input

        vad_segments = runner.detect(audio_data)
        speech_duration = sum(seg["end"] - seg["start"] for seg in vad_segments)
        results.append(
            {
                "file": file_name,
                "num_segments": len(vad_segments),
                "speech_duration": speech_duration,
                "segments": vad_segments,
            }
        )
    return results


def print_result(result: Dict[str, Any]):
    """Pretty print a single transcription result."""
    print(f"\nFile: {result['file']}")
    print(f"Generation Time: {result['generation_time']:.2f}s")
    print(f"\n--- Raw Output ---")
    print(result['raw_text'][:500] + "..." if len(result['raw_text']) > 500 else result['raw_text'])
    
    if result['segments']:
        print(f"\n--- Structured Output ({len(result['segments'])} segments) ---")
        # for seg in result['segments'][:50]:  # Show first 50 segments
        for seg in result['segments']:  # Show first 50 segments
            print(f"[{seg.get('start_time', 'N/A')} - {seg.get('end_time', 'N/A')}] "
                  f"Speaker {seg.get('speaker_id', 'N/A')}: {seg.get('text', '')}")
        # if len(result['segments']) > 50:
        #     print(f"  ... and {len(result['segments']) - 50} more segments")


def load_dataset_and_concatenate(
    dataset_name: str,
    split: str,
    max_duration: float,
    num_audios: int,
    target_sr: int = 24000
) -> Optional[List[np.ndarray]]:
    """
    Load a HuggingFace dataset and concatenate audio samples into long audio chunks.
    (Note, just for demo purpose, not for benchmark evaluation)
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'openslr/librispeech_asr')
        split: Dataset split to use (e.g., 'test', 'test.other')
        max_duration: Maximum duration in seconds for each concatenated audio
        num_audios: Number of concatenated audios to create
        target_sr: Target sample rate (default: 24000)
    
    Returns:
        List of concatenated audio arrays, or None if loading fails
    """
    try:
        from datasets import load_dataset
        import torchcodec # just for decode audio in datasets
    except ImportError:
        print("Please install it with: pip install datasets torchcodec")
        return None        
    
    print(f"\nLoading dataset: {dataset_name} (split: {split})")
    print(f"Will create {num_audios} concatenated audio(s), each up to {max_duration:.1f}s ({max_duration/3600:.2f} hours)")
    
    try:
        # Use streaming to avoid downloading the entire dataset
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        print(f"Dataset loaded in streaming mode")
        
        concatenated_audios = []  # List of concatenated audio metadata
        
        # Create multiple concatenated audios based on num_audios
        current_chunks = []
        current_duration = 0.0
        current_samples_used = 0
        sample_idx = 0
        
        for sample in dataset:
            if len(concatenated_audios) >= num_audios:
                break
                
            if 'audio' not in sample:
                continue
            
            audio_data = sample['audio']
            audio_array = audio_data['array']
            sr = audio_data['sampling_rate']
            
            # Resample if needed
            if sr != target_sr:
                duration = len(audio_array) / sr
                new_length = int(duration * target_sr)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            chunk_duration = len(audio_array) / target_sr
            
            # Check if adding this chunk exceeds max_duration
            if current_duration + chunk_duration > max_duration:
                remaining_duration = max_duration - current_duration
                if remaining_duration > 0.5:  # Only add if > 0.5s remaining
                    samples_to_take = int(remaining_duration * target_sr)
                    current_chunks.append(audio_array[:samples_to_take])
                    current_duration += remaining_duration
                    current_samples_used += 1
                
                # Save current concatenated audio and start a new one
                if current_chunks:
                    concatenated_audios.append({
                        'array': np.concatenate(current_chunks),
                        'duration': current_duration,
                        'samples_used': current_samples_used,
                    })
                    print(f"  Created audio {len(concatenated_audios)}: {current_duration:.1f}s from {current_samples_used} samples")
                
                # Reset for next concatenated audio
                current_chunks = []
                current_duration = 0.0
                current_samples_used = 0
                
                if len(concatenated_audios) >= num_audios:
                    break
            
            current_chunks.append(audio_array)
            current_duration += chunk_duration
            current_samples_used += 1
            
            sample_idx += 1
            if sample_idx % 100 == 0:
                print(f"  Processed {sample_idx} samples...")
        
        # Don't forget the last batch if it has content
        if current_chunks and len(concatenated_audios) < num_audios:
            concatenated_audios.append({
                'array': np.concatenate(current_chunks),
                'duration': current_duration,
                'samples_used': current_samples_used,
            })
            print(f"  Created audio {len(concatenated_audios)}: {current_duration:.1f}s from {current_samples_used} samples")
        
        if not concatenated_audios:
            print("Warning: No audio samples found in dataset")
            return None
        
        # Extract arrays and print summary
        result = [a['array'] for a in concatenated_audios]
        total_duration = sum(a['duration'] for a in concatenated_audios)
        total_samples = sum(a['samples_used'] for a in concatenated_audios)
        print(f"\nCreated {len(result)} concatenated audio(s), total {total_duration:.1f}s ({total_duration/60:.1f} min) from {total_samples} samples")
        
        return result
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Batch Inference Demo")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--audio_files", 
        type=str, 
        nargs='+',
        required=False,
        help="Paths to audio files for transcription"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=False,
        help="Directory containing audio files for batch transcription"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="HuggingFace dataset name (e.g., 'openslr/librispeech_asr')"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'test', 'test.other', 'test.clean')"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=3600.0,
        help="Maximum duration in seconds for concatenated dataset audio (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for processing multiple files"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else ("xpu" if torch.backends.xpu.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu") ),
        choices=["cuda", "cpu", "mps","xpu", "auto"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search. Use 1 for greedy/sampling"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation to use. 'auto' will select the best available for your device (flash_attention_2 for CUDA, sdpa for MPS/CPU/XPU)"
    )
    parser.add_argument(
        "--phase1_vad_only",
        action="store_true",
        help="Run WhisperX-style Phase1 VAD only and skip ASR transcription"
    )
    parser.add_argument(
        "--phase1_vad_then_transcribe",
        action="store_true",
        help="Run WhisperX-style VAD first, then feed chunks into ASR transcription"
    )
    parser.add_argument(
        "--vad_method",
        type=str,
        default="pyannote",
        choices=["pyannote", "silero"],
        help="VAD backend for Phase1 VAD mode"
    )
    parser.add_argument(
        "--vad_onset",
        type=float,
        default=0.400,
        help="Onset threshold for VAD"
    )
    parser.add_argument(
        "--vad_offset",
        type=float,
        default=0.363,
        help="Offset threshold for VAD"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Chunk size (seconds) for VAD merge"
    )
    parser.add_argument(
        "--vad_device_index",
        type=int,
        default=0,
        help="Device index for CUDA VAD backend"
    )
    parser.add_argument(
        "--vad_output_json",
        type=str,
        default="",
        help="Optional output json path for Phase1 VAD segments"
    )
    
    args = parser.parse_args()
    
    # Auto-detect best attention implementation based on device
    if args.attn_implementation == "auto":
        if args.device == "cuda" and torch.cuda.is_available():
            try:
                import flash_attn
                args.attn_implementation = "flash_attention_2"
            except ImportError:
                print("flash_attn not installed, falling back to sdpa")
                args.attn_implementation = "sdpa"
        else:
            # MPS/XPU/CPU don't support flash_attention_2
            args.attn_implementation = "sdpa"
        print(f"Auto-detected attention implementation: {args.attn_implementation}")
    
    # Collect audio files
    audio_files = []
    concatenated_audio = None  # For storing concatenated dataset audio
    
    if args.audio_files:
        audio_files.extend(args.audio_files)
    
    if args.audio_dir:
        import glob
        for ext in ["*.wav", "*.mp3", "*.flac", "*.mp4", "*.m4a", "*.webm"]:
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
    
    if args.dataset:
        concatenated_audio = load_dataset_and_concatenate(
            dataset_name=args.dataset,
            split=args.split,
            max_duration=args.max_duration,
            num_audios=args.batch_size,
        )
        if concatenated_audio is None:
            return
    
    if len(audio_files) == 0 and concatenated_audio is None:
        print("No audio files provided. Please specify --audio_files, --audio_dir, or --dataset.")
        return
    
    if audio_files:
        print(f"\nAudio files to process ({len(audio_files)}):")
        for f in audio_files:
            print(f"  - {f}")
    
    if concatenated_audio:
        print(f"\nConcatenated dataset audios: {len(concatenated_audio)} audio(s)")
    
    # Combine all audio inputs
    all_audio_inputs = audio_files + (concatenated_audio or [])
    
    print("\n" + "="*80)
    print(f"Processing {len(all_audio_inputs)} audio(s)")
    print("="*80)

    if args.phase1_vad_only:
        vad_results = run_phase1_vad_only(all_audio_inputs, args)
        for result in vad_results:
            print("\n" + "-" * 60)
            print(f"File: {result['file']}")
            print(f"VAD Segments: {result['num_segments']}")
            print(f"Speech Duration: {result['speech_duration']:.2f}s")
            if result["segments"]:
                preview = result["segments"][:10]
                print(json.dumps(preview, indent=2, ensure_ascii=False))
        if args.vad_output_json:
            with open(args.vad_output_json, "w", encoding="utf-8") as f:
                json.dump(vad_results, f, indent=2, ensure_ascii=False)
            print(f"\nSaved VAD results to: {args.vad_output_json}")
        return

    if args.device == "mps":
        model_dtype = torch.float32
    elif args.device == "xpu":
        model_dtype = torch.float32
    elif args.device == "cpu":
        model_dtype = torch.float32
    else:
        model_dtype = torch.bfloat16

    asr = VibeVoiceASRBatchInference(
        model_path=args.model_path,
        device=args.device,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation
    )
    
    do_sample = args.temperature > 0
    
    if args.phase1_vad_then_transcribe:
        vad_runner = WhisperXPhase1VAD(
            vad_method=args.vad_method,
            vad_onset=args.vad_onset,
            vad_offset=args.vad_offset,
            chunk_size=args.chunk_size,
            device=args.device,
            device_index=args.vad_device_index,
        )
        all_results = asr.transcribe_with_vad(
            all_audio_inputs,
            vad_runner=vad_runner,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=do_sample,
            num_beams=args.num_beams,
        )
    else:
        all_results = asr.transcribe_with_batching(
            all_audio_inputs,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=do_sample,
            num_beams=args.num_beams,
        )
    
    # Print results
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    for result in all_results:
        print("\n" + "-"*60)
        print_result(result)


if __name__ == "__main__":
    main()
