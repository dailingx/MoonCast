
import sys
sys.path.append(".")
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize, detokenize_noref, detokenize_streaming, detokenize_noref_streaming
import torch
import os
from glob import glob
import base64
import io
import torchaudio
from transformers import AutoModelForCausalLM, GenerationConfig
import librosa
from tqdm import tqdm
from pydub import AudioSegment

class Model(object):
    def __init__(self):

        
        self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
        self.speech_token_offset = 163840
        print(self.extra_tokens)
        self.assistant_ids = self.tokenizer.encode("assistant") # [110866]
        self.user_ids = self.tokenizer.encode("user") # [1495]
        self.audio_ids = self.tokenizer.encode("audio") # [26229]
        self.spk_0_ids = self.tokenizer.encode("0") # [501] 
        self.spk_1_ids = self.tokenizer.encode("1") # [503] 

        self.msg_end = self.extra_tokens.msg_end # 260
        self.user_msg_start = self.extra_tokens.user_msg_start # 261
        self.assistant_msg_start = self.extra_tokens.assistant_msg_start # 262
        self.name_end = self.extra_tokens.name_end # 272
        self.media_begin = self.extra_tokens.media_begin # 273
        self.media_content = self.extra_tokens.media_content # 274
        self.media_end = self.extra_tokens.media_end # 275

        self.audio_tokenizer =  get_audio_tokenizer()
        self.audio_detokenizer = get_audio_detokenizer()
        model_path = "resources/text2semantic"
        self.model =  AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True, force_download=True).to(torch.cuda.current_device())
        self.generate_config = GenerationConfig(
            max_new_tokens=200 * 50, # no more than 200s per turn
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.8,
            eos_token_id=self.media_end,
        )
    
    def _clean_text(self, text):
        # you can add front-end processing here
        text = text.replace("“", "")
        text = text.replace("”", "")
        text = text.replace("...", " ")
        text = text.replace("…", " ")
        text = text.replace("*", "")
        text = text.replace(":", ",")
        text = text.replace("‘", "'")
        text = text.replace("’", "'")
        text = text.strip()
        return text

    @torch.inference_mode()
    def _process_text(self, js):

        if "role_mapping" in js:
            for role in js["role_mapping"].keys():
                js["role_mapping"][role]["ref_bpe_ids"] = self.tokenizer.encode(self._clean_text(js["role_mapping"][role]["ref_text"]))
                
        for turn in js["dialogue"]:
            turn["bpe_ids"] = self.tokenizer.encode(self._clean_text(turn["text"]))
        return js
        
    def inference(self, js, streaming=False):
        js = self._process_text(js)
        if "role_mapping" not in js:
            if streaming:
                return self.infer_without_prompt_streaming(js)
            else:
                return self.infer_without_prompt(js)
        else:
            if streaming:
                return self.infer_with_prompt_streaming(js)
            else:
                return self.infer_with_prompt(js)      
    
    @torch.inference_mode()
    def infer_with_prompt(self, js):
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids  + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids  + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]

        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]

        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(torch.cuda.current_device())
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(torch.cuda.current_device())
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(torch.cuda.current_device())
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(torch.cuda.current_device())
        

        prompt = []
        cur_role_dict = dict()
        for role, role_item in js["role_mapping"].items():
            waveform_24k = librosa.load(role_item["ref_audio"], sr=24000)[0]
            waveform_24k = torch.tensor(waveform_24k).unsqueeze(0).to(torch.cuda.current_device())

            waveform_16k = librosa.load(role_item["ref_audio"], sr=16000)[0]
            waveform_16k = torch.tensor(waveform_16k).unsqueeze(0).to(torch.cuda.current_device())

            semantic_tokens = self.audio_tokenizer.tokenize(waveform_16k)
            semantic_tokens = semantic_tokens.to(torch.cuda.current_device())
            prompt_ids = semantic_tokens + self.speech_token_offset

            cur_role_dict[role] = {
                "ref_bpe_ids": role_item["ref_bpe_ids"],
                "wav_24k": waveform_24k,
                "semantic_tokens": semantic_tokens,
                "prompt_ids": prompt_ids
            }
        
        prompt = prompt + user_role_0_ids + cur_role_dict["0"]["ref_bpe_ids"] + [self.msg_end]
        prompt = prompt + user_role_1_ids + cur_role_dict["1"]["ref_bpe_ids"] + [self.msg_end]
        
        for seg_id, turn in enumerate(js["dialogue"]):
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            cur_start_ids = cur_user_ids + turn["bpe_ids"] + [self.msg_end]
            prompt = prompt + cur_start_ids
        
        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())

        prompt = torch.cat([prompt, assistant_role_0_ids, media_start, cur_role_dict["0"]["prompt_ids"], media_end], dim=-1)
        prompt = torch.cat([prompt, assistant_role_1_ids, media_start, cur_role_dict["1"]["prompt_ids"], media_end], dim=-1)

        
        generation_config = self.generate_config
        # you can modify sampling strategy here

        wav_list = []
        for seg_id, turn in tqdm(enumerate(js["dialogue"])):
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids                
            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)
            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2
            # print(generation_config)
            # todo: add streaming support for generate function
            outputs = self.model.generate(prompt,
                                          generation_config=generation_config)
            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]
            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)            

            torch_token = output_token - self.speech_token_offset
            gen_speech_fm = detokenize(self.audio_detokenizer, torch_token, cur_role_dict[role_id]["wav_24k"], cur_role_dict[role_id]["semantic_tokens"])
            gen_speech_fm = gen_speech_fm.cpu()
            gen_speech_fm = gen_speech_fm / gen_speech_fm.abs().max()
            wav_list.append(gen_speech_fm)
            del torch_token
        
        concat_wav = torch.cat(wav_list, dim=-1).cpu()
        # print(concat_wav.shape)
        buffer = io.BytesIO()
        torchaudio.save(buffer, concat_wav, sample_rate=24000, format="mp3")
        audio_bytes = buffer.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_b64
    
    @torch.inference_mode()
    def infer_with_prompt_streaming(self, js):
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids  + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids  + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]

        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]

        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(torch.cuda.current_device())
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(torch.cuda.current_device())
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(torch.cuda.current_device())
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(torch.cuda.current_device())
        

        prompt = []
        cur_role_dict = dict()
        for role, role_item in js["role_mapping"].items():
            waveform_24k = librosa.load(role_item["ref_audio"], sr=24000)[0]
            waveform_24k = torch.tensor(waveform_24k).unsqueeze(0).to(torch.cuda.current_device())

            waveform_16k = librosa.load(role_item["ref_audio"], sr=16000)[0]
            waveform_16k = torch.tensor(waveform_16k).unsqueeze(0).to(torch.cuda.current_device())

            semantic_tokens = self.audio_tokenizer.tokenize(waveform_16k)
            semantic_tokens = semantic_tokens.to(torch.cuda.current_device())
            prompt_ids = semantic_tokens + self.speech_token_offset

            cur_role_dict[role] = {
                "ref_bpe_ids": role_item["ref_bpe_ids"],
                "wav_24k": waveform_24k,
                "semantic_tokens": semantic_tokens,
                "prompt_ids": prompt_ids
            }
        
        prompt = prompt + user_role_0_ids + cur_role_dict["0"]["ref_bpe_ids"] + [self.msg_end]
        prompt = prompt + user_role_1_ids + cur_role_dict["1"]["ref_bpe_ids"] + [self.msg_end]
        
        for seg_id, turn in enumerate(js["dialogue"]):
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            cur_start_ids = cur_user_ids + turn["bpe_ids"] + [self.msg_end]
            prompt = prompt + cur_start_ids
        
        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())

        prompt = torch.cat([prompt, assistant_role_0_ids, media_start, cur_role_dict["0"]["prompt_ids"], media_end], dim=-1)
        prompt = torch.cat([prompt, assistant_role_1_ids, media_start, cur_role_dict["1"]["prompt_ids"], media_end], dim=-1)

        
        generation_config = self.generate_config
        # you can modify sampling strategy here

        wav_list = []
        for seg_id, turn in tqdm(enumerate(js["dialogue"])):
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids                
            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)
            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2
            # print(generation_config)
            # todo: add streaming support for generate function
            outputs = self.model.generate(prompt,
                                          generation_config=generation_config)
            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]
            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)            

            torch_token = output_token - self.speech_token_offset
            for cur_chunk in detokenize_streaming(self.audio_detokenizer, torch_token, cur_role_dict[role_id]["wav_24k"], cur_role_dict[role_id]["semantic_tokens"]):
                cur_chunk = cur_chunk.cpu()
                cur_chunk = cur_chunk / cur_chunk.abs().max()
                cur_buffer = io.BytesIO()
                torchaudio.save(cur_buffer, cur_chunk, sample_rate=24000, format="mp3")
                audio_bytes = cur_buffer.getvalue()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                yield audio_b64
               
    @torch.inference_mode()
    def infer_without_prompt(self, js):
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids  + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids  + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]

        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]

        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(torch.cuda.current_device())
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(torch.cuda.current_device())
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(torch.cuda.current_device())
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(torch.cuda.current_device())
        

        prompt = []
        for seg_id, turn in enumerate(js["dialogue"]):
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            cur_start_ids = cur_user_ids + turn["bpe_ids"] + [self.msg_end]
            prompt = prompt + cur_start_ids

        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())
        generation_config = self.generate_config
        # you can modify sampling strategy here

        wav_list = []
        for seg_id, turn in tqdm(enumerate(js["dialogue"])):
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids                
            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)
            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2
            # todo: add streaming support for generate function
            outputs = self.model.generate(prompt,
                                          generation_config=generation_config)
            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]
            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)

            torch_token = output_token - self.speech_token_offset
            gen_speech_fm = detokenize_noref(self.audio_detokenizer, torch_token)
            gen_speech_fm = gen_speech_fm.cpu()
            gen_speech_fm = gen_speech_fm / gen_speech_fm.abs().max()
            wav_list.append(gen_speech_fm)
            del torch_token

        concat_wav = torch.cat(wav_list, dim=-1).cpu()
        # print(concat_wav.shape)
        buffer = io.BytesIO()
        torchaudio.save(buffer, concat_wav, sample_rate=24000, format="mp3")
        audio_bytes = buffer.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_b64
    
    @torch.inference_mode()
    def infer_without_prompt_streaming(self, js):
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids  + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids  + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]

        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]

        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(torch.cuda.current_device())
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(torch.cuda.current_device())
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(torch.cuda.current_device())
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(torch.cuda.current_device())
        

        prompt = []
        for seg_id, turn in enumerate(js["dialogue"]):
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            cur_start_ids = cur_user_ids + turn["bpe_ids"] + [self.msg_end]
            prompt = prompt + cur_start_ids

        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())
        generation_config = self.generate_config
        # you can modify sampling strategy here

        wav_list = []
        for seg_id, turn in tqdm(enumerate(js["dialogue"])):
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids                
            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)
            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2
            # print(generation_config)
            # todo: add streaming support for generate function
            outputs = self.model.generate(prompt,
                                          generation_config=generation_config)
            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]
            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)

            torch_token = output_token - self.speech_token_offset
            for cur_chunk in detokenize_noref_streaming(self.audio_detokenizer, torch_token):
                cur_chunk = cur_chunk.cpu()
                cur_chunk = cur_chunk / cur_chunk.abs().max()
                cur_buffer = io.BytesIO()
                torchaudio.save(cur_buffer, cur_chunk, sample_rate=24000, format="mp3")
                audio_bytes = cur_buffer.getvalue()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                yield audio_b64
           
        
if __name__ == "__main__":
    model = Model()
    
    # speaker should be interleaved
    zh_test_json = {
        "role_mapping": {
            "0": {
                "ref_audio": "./zh_prompt0.wav",
                "ref_text": "可以每天都骑并且可能会让你爱上骑车，然后通过爱上骑车的你省了很多很多钱。", #asr output
            },
            "1": {
                "ref_audio": "./zh_prompt1.wav",
                "ref_text": "他最后就能让同样食材炒出来的菜味道大大提升。" #asr output
            }
        },      
        "dialogue": [
            {
                "role": "0",
                "text": "嘿，大家好！欢迎收听我们的播客节目，今天我们要聊的主题是“终生成长”。我是主持人Kimi，今天非常荣幸地邀请到了我的好朋友，职业发展规划师Xihaohu。Xihaohu，要不你先和大家打个招呼，"
            },
            {
                "role": "1",
                "text": "大家好，我是Xihaohu，很高兴今天能和大家一起探讨这个非常有意义的话题，"
            },
            {
                "role": "0",
                "text": "今天我们要聊的这本书叫做《终身成长》，作者是卡罗尔·德韦克。这本书其实和我有点小故事。记得有一次我在机场等飞机，无聊翻书，结果就被这本书深深吸引了。书里提到的两种思维模式——固定型和成长型，真的让我大开眼界。我当时就想，哇，原来我们的思维模式可以这么影响我们的生活和成长，"
            },
            {
                "role": "1",
                "text": "对，这本书确实非常有启发性。它不仅解释了为什么人们在面对挑战和失败时的反应会如此不同，还提供了如何培养成长型思维的方法，"
            },
            {
                "role": "0",
                "text": "那我们先来聊聊第一个话题吧。书中提到的两种思维模式，固定型和成长型，它们到底是怎么影响我们的生活的呢？Xihaohu，你作为一个职业发展规划师，有没有遇到过因为思维模式不同而发展路径也不同的案例，"
            },
            {
                "role": "1",
                "text": "当然有。我记得有一个客户，他总是觉得自己不够聪明，每次遇到困难就觉得自己不行。这种固定型思维让他总是在同一个水平上徘徊，不敢挑战更高的目标。后来我们通过一些训练和指导，帮助他转变思维，开始相信通过努力可以提升自己。结果，他的工作表现和自信心都有了显著的提升，"
            },
            {
                "role": "0",
                "text": "哇，这听起来真的很神奇。那你觉得，对于普通人来说，如何才能更好地培养成长型思维呢，"
            },
            {
                "role": "1",
                "text": "我觉得关键是要接受挑战，不怕失败。每次失败其实都是一次学习的机会。还有就是，要学会给自己设定小目标，逐步提升。不要总是想着一步登天，而是要享受成长的过程，"
            },
            {
                "role": "0",
                "text": "说得对，成长其实就是一个不断学习和适应的过程。那我们再来聊聊第二个话题，书中提到的“成功的意义”。对于你来说，成功是什么？是学习的成果，还是某种证明，"
            },
            {
                "role": "1",
                "text": "对我来说，成功更多的是一种持续学习和成长的状态。不是说达到了某个目标就结束了，而是要不断地追求更好的自己。每次的进步和学习都是成功的一部分，"
            },
            {
                "role": "0",
                "text": "嗯，这让我想起了一句名言，“成功不是终点，失败也不是终结，重要的是继续前进的勇气。”那你觉得，我们在日常生活中，如何才能更好地理解和实践这种对成功的定义呢，"
            },
            {
                "role": "1",
                "text": "我觉得可以从日常的小事做起。比如，每天学习一点新知识，或者尝试解决一个小问题。这样不仅能积累知识，还能增强自信心，慢慢地就会形成一种持续成长的习惯，"
            },
            {
                "role": "0",
                "text": "听起来很实用啊。那我们今天的节目就先聊到这里，非常感谢Xihaohu的分享。希望大家都能从中获得一些启发，开始自己的成长之旅。我们下期再见，"
            },
            {
                "role": "1",
                "text": "谢谢大家，下期再见。"
            }
        ]
    }


    audio_bytes_gen = model.inference(zh_test_json, streaming=True)
    audio = AudioSegment.empty()
    for cur_chunk in audio_bytes_gen:
        cur_chunk = base64.b64decode(cur_chunk)
        audio_chunk = AudioSegment.from_file(io.BytesIO(cur_chunk), format="mp3")
        audio += audio_chunk
    audio.export("tmp_generated_zh_stream.mp3", format="mp3")
    print("zh stream done")
    

    audio_bytes = model.inference(zh_test_json)
    file_to_save = open(f"tmp_generated_zh.mp3", "wb")
    file_to_save.write(base64.b64decode(audio_bytes))
    print("zh done")

    # speaker should be interleaved
    en_test_json = {
        "role_mapping": {
            "0": {
                "ref_audio": "./en_prompt0.wav",
                "ref_text": "Yeah, no, this is my backyard. It's never ending So just the way I like it. So social distancing has never been a problem.", #asr output
            },
            "1": {
                "ref_audio": "./en_prompt1.wav",
                "ref_text": "I'm doing great And. Look, it couldn't be any better than having you at your set, which is the outdoors." #asr output
            }
        },      
        "dialogue": [
            {
                "role": "0",
                "text": "In an awesome time, And, we're even gonna do a second episode too So. This is part one part two, coming at some point in the future There. We are.",
            },
            {
                "role": "1",
                "text": "I love it. So grateful Thank you So I'm really excited. That's awesome. Yeah."
            },
            {
                "role": "0",
                "text": "All I was told, which is good because I don't want to really talk too much more is that you're really really into fitness and nutrition And overall holistic I love it Yes."
            },
            {
                "role": "1",
                "text": "Yeah So I started around thirteen Okay But my parents were fitness instructors as well. Awesome So I came from the beginning, and now it's this transition into this wholeness because I had to chart my. Own path and they weren't into nutrition at all So I had to learn that part."
            }
        ]
    }
    audio_bytes = model.inference(en_test_json)
    file_to_save = open(f"tmp_generated_en.mp3", "wb")
    file_to_save.write(base64.b64decode(audio_bytes))
    print("en done")


    # also support inference without prompt
    # speaker should be interleaved
    without_prompt_test_json = {
        "dialogue": [
            {
                "role": "0",
                "text": "嘿，大家好！欢迎收听我们的播客节目，今天我们要聊的主题是“终生成长”。我是主持人Kimi，今天非常荣幸地邀请到了我的好朋友，职业发展规划师Xihaohu。Xihaohu，要不你先和大家打个招呼，"
            },
            {
                "role": "1",
                "text": "大家好，我是Xihaohu，很高兴今天能和大家一起探讨这个非常有意义的话题，"
            },
            {
                "role": "0",
                "text": "今天我们要聊的这本书叫做《终身成长》，作者是卡罗尔·德韦克。这本书其实和我有点小故事。记得有一次我在机场等飞机，无聊翻书，结果就被这本书深深吸引了。书里提到的两种思维模式——固定型和成长型，真的让我大开眼界。我当时就想，哇，原来我们的思维模式可以这么影响我们的生活和成长，"
            },
            {
                "role": "1",
                "text": "对，这本书确实非常有启发性。它不仅解释了为什么人们在面对挑战和失败时的反应会如此不同，还提供了如何培养成长型思维的方法，"
            },
            {
                "role": "0",
                "text": "那我们先来聊聊第一个话题吧。书中提到的两种思维模式，固定型和成长型，它们到底是怎么影响我们的生活的呢？Xihaohu，你作为一个职业发展规划师，有没有遇到过因为思维模式不同而发展路径也不同的案例，"
            },
            {
                "role": "1",
                "text": "当然有。我记得有一个客户，他总是觉得自己不够聪明，每次遇到困难就觉得自己不行。这种固定型思维让他总是在同一个水平上徘徊，不敢挑战更高的目标。后来我们通过一些训练和指导，帮助他转变思维，开始相信通过努力可以提升自己。结果，他的工作表现和自信心都有了显著的提升，"
            },
            {
                "role": "0",
                "text": "哇，这听起来真的很神奇。那你觉得，对于普通人来说，如何才能更好地培养成长型思维呢，"
            },
            {
                "role": "1",
                "text": "我觉得关键是要接受挑战，不怕失败。每次失败其实都是一次学习的机会。还有就是，要学会给自己设定小目标，逐步提升。不要总是想着一步登天，而是要享受成长的过程，"
            },
            {
                "role": "0",
                "text": "说得对，成长其实就是一个不断学习和适应的过程。那我们再来聊聊第二个话题，书中提到的“成功的意义”。对于你来说，成功是什么？是学习的成果，还是某种证明，"
            },
            {
                "role": "1",
                "text": "对我来说，成功更多的是一种持续学习和成长的状态。不是说达到了某个目标就结束了，而是要不断地追求更好的自己。每次的进步和学习都是成功的一部分，"
            },
            {
                "role": "0",
                "text": "嗯，这让我想起了一句名言，“成功不是终点，失败也不是终结，重要的是继续前进的勇气。”那你觉得，我们在日常生活中，如何才能更好地理解和实践这种对成功的定义呢，"
            },
            {
                "role": "1",
                "text": "我觉得可以从日常的小事做起。比如，每天学习一点新知识，或者尝试解决一个小问题。这样不仅能积累知识，还能增强自信心，慢慢地就会形成一种持续成长的习惯，"
            },
            {
                "role": "0",
                "text": "听起来很实用啊。那我们今天的节目就先聊到这里，非常感谢Xihaohu的分享。希望大家都能从中获得一些启发，开始自己的成长之旅。我们下期再见，"
            },
            {
                "role": "1",
                "text": "谢谢大家，下期再见。"
            }
        ]
    }
    audio_bytes = model.inference(without_prompt_test_json)
    file_to_save = open(f"tmp_generated_woprompt.mp3", "wb")
    file_to_save.write(base64.b64decode(audio_bytes))
    print("without prompt done")