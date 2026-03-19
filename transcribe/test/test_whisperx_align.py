import sys
import os
# Add parent directory to path to allow importing whisperx_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from whisperx_utils.alignment import align, load_align_model
from whisperx_utils.schema import AlignedTranscriptionResult, TranscriptionResult
from whisperx_utils.audio import load_audio

# results = [({'segments': [{'text': " Welcome back to The Deep Dive. We're doing something a little different today.", 'start': 0.031, 'end': 4.351, 'avg_logprob': -0.1683477850423919}, {'text': ' Yeah, a bit of a time and place shift for this one. Right. So I want you to mentally transport yourself to a very specific setting.', 'start': 4.452, 'end': 11.185, 'avg_logprob': -0.09699704906632824}, {'text': ' It is late February, the year is 2026, and we are standing right in the middle of China.', 'start': 11.185, 'end': 16.585, 'avg_logprob': -0.11793999373912811}, {'text': " And if you follow the lunar calendar, you know we have just entered a very specific zodiac year. It's the year of the firehorse.", 'start': 16.737, 'end': 24.938, 'avg_logprob': -0.08204010705794057}, {'text': ' the firehorse. I mean, that just sounds, it sounds really intense. It is. Yeah.', 'start': 24.938, 'end': 30.018, 'avg_logprob': -0.2696085382591594}], 'language': 'en'}, '/home/ysl/workspace/develop/video-tool-box/output.m4a')]
# results = [({'segments': [{'text': " Welcome back to The Deep Dive. We're doing something a little different today.", 'start': 0.031, 'end': 4.351}, {'text': ' Yeah, a bit of a time and place shift for this one. Right. So I want you to mentally transport yourself to a very specific setting.', 'start': 4.452, 'end': 11.185}, {'text': ' It is late February, the year is 2026, and we are standing right in the middle of China.', 'start': 11.185, 'end': 16.585}, {'text': " And if you follow the lunar calendar, you know we have just entered a very specific zodiac year. It's the year of the firehorse.", 'start': 16.737, 'end': 24.938}, {'text': ' the firehorse. I mean, that just sounds, it sounds really intense. It is. Yeah.', 'start': 24.938, 'end': 30.018}], 'language': 'en'}, '/home/ysl/workspace/develop/video-tool-box/output.m4a')]
audio_path = os.path.join(os.path.dirname(__file__), 'resource', 'DDS course_70s.m4a')
results = [({'segments': [{"start":0.0,"end":4.6,"text":"[Silence]"},{"start":4.6,"end":17.62,"Speaker":0,"text":"大家好。DDS培训课程内容包括四个方面：一、为什么需要DDS防护？二、DDS在工厂的分类。"},{"start":17.62,"end":24.38,"Speaker":0,"text":"第三，DDS常见问题分享。第四，DDS防护一般法则。"},{"start":25.6,"end":28.16,"Speaker":0,"text":"下面我们来看一组图片。"},{"start":28.16,"end":30.38,"text":"[Silence]"},{"start":30.38,"end":33.19,"Speaker":0,"text":"大家从图片中发现了什么？"},{"start":33.19,"end":37.13,"text":"[Silence]"},{"start":37.13,"end":50.7,"Speaker":0,"text":"是不是有不同程度的凹陷、划伤？那么让你来选择这些产品，你会选择吗？我相信你们的回答一定是no。"},{"start":50.7,"end":55.56,"text":"[Silence]"},{"start":55.56,"end":64.92,"Speaker":0,"text":"下面是我们今天DDS课程的第一个内容。为什么我们要进行DDS培训？"},{"start":66.35,"end":70.01,"Speaker":0,"text":"主要有两个原因，首先，第一，"}], 'language': 'zh'}, audio_path)]
align_language = "zh"
device = "cpu"
align_model =None
model_dir = None
model_cache_only = None
interpolate_method = "nearest"
return_char_alignments = False
print_progress = False
audio = load_audio(results[0][1])

tmp_results = results
results = []
align_model, align_metadata = load_align_model(
    align_language, device, model_name=align_model, model_dir=model_dir, model_cache_only=model_cache_only
)
for result, audio_path in tmp_results:
    # >> Align
    if len(tmp_results) > 1:
        input_audio = audio_path
    else:
        # lazily load audio from part 1
        input_audio = audio

    if align_model is not None and len(result["segments"]) > 0:
        if result.get("language", "en") != align_metadata["language"]:
            # load new language
            print(
                f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
            )
            align_model, align_metadata = load_align_model(
                result["language"], device, model_dir=model_dir, model_cache_only=model_cache_only
            )
        print("Performing alignment...")
        result: AlignedTranscriptionResult = align(
            result["segments"],
            align_model,
            align_metadata,
            input_audio,
            device,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments,
            print_progress=print_progress,
        )
    results.append((result, audio_path))
