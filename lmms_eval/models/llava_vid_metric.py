import logging
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
import torch
from tqdm import tqdm
import time
import os

from decord import VideoReader, cpu
import numpy as np
import math
from datetime import timedelta
from transformers import AutoConfig
import copy
import types  # 新增


from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

eval_logger = logging.getLogger("lmms-eval")
import sys

sys.path.append("LLaVA-NeXT-inference")
try:
    from llavavid.model.language_model.llava_llama import LlavaConfig

    # from llavavid.model.language_model.llava_qwen import LlavaQwenConfig
    from llavavid.model.builder import load_pretrained_model
    from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llavavid.conversation import conv_templates, SeparatorStyle

    # AutoConfig.register("llava_qwen", LlavaQwenConfig)
    AutoConfig.register("llava_llama", LlavaConfig)

except ImportError:
    eval_logger.debug("LLaVA-Video is not installed. Please install LLaVA-Video to use this model.")

try:
    from llavavid.model.language_model.llava_qwen import LlavaQwenConfig

    AutoConfig.register("llava_qwen", LlavaQwenConfig)
except:
    eval_logger.debug("")

# ===== profiling helpers (新增) =====
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _bytes_to_gb(x: int) -> float:
    return x / float(1024 ** 3)

def _allreduce_max_scalar(x: float) -> float:
    # 多卡时取全局最大；单卡/未初始化分布式时原样返回
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([x], device=torch.cuda.current_device())
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return float(t.item())
    except Exception:
        pass
    return x
# ===== end helpers =====


# @register_model("llavavid")
@register_model("llava_vid")
class LlavaVid(lmms):
    """
    LlavaVid Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        max_frames_num: int = 3,
        mm_resampler_type: str = "spatial_pool",
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_out_channels: int = 1024,
        mm_spatial_pool_mode: str = "average",
        overwrite: bool = True,
        video_decode_backend: str = "pyav",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = overwrite
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(max_frames_num)
        if self.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = self.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_out_channels"] = self.mm_spatial_pool_out_channels
            overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
            overwrite_config["mm_resampler_location"] = "before"
            overwrite_config["patchify_video_feature"] = False
            overwrite_config["attn_implementation"] = attn_implementation
            overwrite_config["use_flash_attention_2"] = False

            cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

            if cfg_pretrained.architectures[0] == "LlavaLlamaForCausalLM":  # Ugly code, only used in  vicuna that needs ROPE
                if "224" in cfg_pretrained.mm_vision_tower:
                    least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            if "v1.5" in pretrained:  # A hardcode solution here to load v1.5 model, otherwise it will use LlavaConfig from hf transformers
                from transformers import AutoTokenizer
                from llavavid.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

                self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
                cfg_pretrained = LlavaConfig.from_pretrained(pretrained)
                if overwrite_config is not None:
                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                kwargs["torch_dtype"] = torch.float16
                self._model = LlavaLlamaForCausalLM.from_pretrained(pretrained, low_cpu_mem_usage=True, config=cfg_pretrained, device_map=self.device_map, **kwargs)
                vision_tower = self._model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model(device_map=self.device_map)
                if self.device_map != "auto":
                    vision_tower.to(device="cuda", dtype=torch.float16)
                self._image_processor = vision_tower.image_processor

                if hasattr(self._model.config, "max_sequence_length"):
                    self._max_length = self._model.config.max_sequence_length
                else:
                    self._max_length = 2048
            else:
                # self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map, overwrite_config=overwrite_config)
                self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                    pretrained,
                    None,
                    self.model_name,
                    device_map=self.device_map,
                    overwrite_config=overwrite_config,
                    attn_implementation=attn_implementation,
                )
        else:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                self.model_name,
                device_map=self.device_map,
            )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1


        # ---- 精确 Prefill 计时 hook（仅新增）----
        self._lm_timing = {"active": False, "prefill_s": 0.0, "decode_s": 0.0}

        def _patch_lm_timer(model_outer):
            # llava_vid 的语言模型对象尽量鲁棒获取
            lm = getattr(model_outer.model, "model", None) or getattr(model_outer.model, "language_model", None) or model_outer.model
            if getattr(lm, "_timing_patched", False):
                return
            orig_forward = lm.forward

            def wrapped_forward(*args, **kwargs):
                pkv    = kwargs.get("past_key_values", None)
                ids    = kwargs.get("input_ids", None)
                embeds = kwargs.get("inputs_embeds", None)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                # --- FIX: 避免 input_ids 既以位置又以关键字传入 ---
                if kwargs:  # 上游一般都会用关键字参数；有就只用 kwargs
                    out = orig_forward(**kwargs)
                else:       # 兜底：纯位置参数的少数路径
                    out = orig_forward(*args)
                # --- END FIX ---
                if torch.cuda.is_available(): torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                if model_outer._lm_timing.get("active", False):
                    seqlen = 0
                    if ids is not None: seqlen = ids.shape[1]
                    elif embeds is not None: seqlen = embeds.shape[1]
                    # "prefill" := 无 pkv 且输入长度>1 的那次 LM 前向
                    if (pkv is None) and (seqlen > 1):
                        model_outer._lm_timing["prefill_s"] += dt
                        model_outer._lm_timing["last_seqlen"] = int(seqlen)   # <--- 新增
                    else:
                        model_outer._lm_timing["decode_s"]  += dt
                return out

            lm.forward = types.MethodType(wrapped_forward, lm)
            lm._timing_patched = True

        _patch_lm_timer(self)
        # ---- end hook ----

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        # fps = round(vr.get_avg_fps())
        # frame_idx = [i for i in range(0, len(vr), fps)]
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video = self.load_video(visual, self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # --- TIMING (dataset-level accumulators) ---
        dataset_start = time.perf_counter()
        sum_gen = 0.0      # 仅 model.generate 阶段累计
        sum_total = 0.0    # 单样本总耗时累计（读视频->预处理->generate->decode）
        n = 0              # 成功处理的样本数
        # --- END TIMING ---

        prefill_exact_sum = 0.0
        prefill_exact_cnt = 0
        ctx_len_sum = 0  # 可选：统计进入 LM 的上下文长度

    

        # --- METRICS (新增) ---
        peak_mem_gb = 0.0
        sum_prefill = 0.0
        prefill_cnt = 0
        PROFILE_N = int(os.environ.get("PROFILE_N", "1"))      # 仅前 N 个样本做 prefill 拆分
        PROFILE_TOKENS = int(os.environ.get("PROFILE_TOKENS", "128"))  # N-token 估计基准
        # --- END METRICS ---


        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            req_start = time.perf_counter()
            # --- per-sample: reset peak mem & sync (新增) ---
            _sync()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            # --- end ---


            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            try:
                for visual in visuals:
                    if self.video_decode_backend == "decord":
                        video = self.load_video(visual, self.max_frames_num)
                    elif self.video_decode_backend == "pyav":
                        video = read_video_pyav(visual, num_frm=self.max_frames_num)
                    # video = self.load_video(visual, self.max_frames_num)
                    video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                    videos.append(video)
            except Exception as e:
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} can not load, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} can not load, check the source")
                pbar.update(1)
                continue

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if "llama_3" in self.conv_template:
                pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            # input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            # pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            # input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            # attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = contexts

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            # 开启 LM 计时
            self._lm_timing["prefill_s"] = 0.0
            self._lm_timing["decode_s"]  = 0.0
            self._lm_timing["active"]    = True

            _sync()
            gen_start = time.perf_counter()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=videos,
                    attention_mask=attention_masks,
                    modalities="video",
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            _sync()
            self._lm_timing["active"] = False
            prefill_exact_sum += self._lm_timing["prefill_s"]
            prefill_exact_cnt += 1
            # 可选：记录上下文长度（进入 generate 的 attention_masks 已构造）
            ctx_len_sum += int(self._lm_timing.get("last_seqlen", 0))  # <--- 用真实 prefill 序列长

                # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, use_cache=True, stopping_criteria=[stopping_criteria])
            gen_end = time.perf_counter()
            sum_gen += (gen_end - gen_start)
            

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            res.append(outputs)
            pbar.update(1)
            sum_total += (time.perf_counter() - req_start)
            n += 1
            # --- update peak GPU (新增) ---
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                mx = max(torch.cuda.max_memory_allocated(dev), torch.cuda.max_memory_reserved(dev))
                peak_mem_gb = max(peak_mem_gb, _bytes_to_gb(mx))

            # --- prefill estimation: 仅前 PROFILE_N 个样本，避免拖慢整体（新增） ---
            if prefill_cnt < PROFILE_N:
                # 复制一份生成参数，关采样避免抖动
                _gkw = {
                    "inputs": input_ids,
                    "images": videos,
                    "attention_mask": attention_masks,
                    "modalities": "video",
                    "use_cache": self.use_cache,
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                # 1 token（≈ prefill + 首 token decode）
                _sync(); _t1s = time.perf_counter()
                with torch.inference_mode():
                    _ = self.model.generate(max_new_tokens=1, **_gkw)
                _sync(); _t1 = time.perf_counter() - _t1s

                # N token（≈ prefill + N 次 decode）
                _N = max(8, PROFILE_TOKENS)
                _sync(); _tNs = time.perf_counter()
                with torch.inference_mode():
                    _ = self.model.generate(max_new_tokens=_N, **_gkw)
                _sync(); _tN = time.perf_counter() - _tNs

                # 平均每 token 解码；prefill ≈ t1 - decode_avg
                _decode_avg = max(0.0, (_tN - _t1) / max(1, _N - 1))
                _prefill = max(0.0, _t1 - _decode_avg)
                sum_prefill += _prefill
                prefill_cnt += 1
            # --- end prefill estimation ---

            # --- TIMING (finalize & log) ---
            total_dataset_time = time.perf_counter() - dataset_start
            avg_generate_s = (sum_gen / n) if n else 0.0
            avg_total_s = (sum_total / n) if n else 0.0

            # 仅主进程打印，避免多卡重复
            if (hasattr(self, "accelerator") and self.accelerator.is_local_main_process) or (getattr(self, "_rank", 0) == 0):
                eval_logger.info(
                    "Timing | total_dataset_time=%.3f s  avg_generate_s=%.3f s  avg_total_s=%.3f s  (n=%d)",
                    total_dataset_time, avg_generate_s, avg_total_s, n
                )

        # ====== 汇总并打印 Metrics（新增） ======
        # ====== 结束 ======
        avg_e2e_s = (sum_total / n) if n else 0.0
        avg_prefill_exact = (prefill_exact_sum / prefill_exact_cnt) if prefill_exact_cnt else 0.0
        avg_ctx_len = (ctx_len_sum / n) if n else 0

        # 多卡取全局最大显存（你已加过 _allreduce_max_scalar 的话就用之；没有就直接打印 peak_mem_gb）
        peak_mem_gb = _allreduce_max_scalar(peak_mem_gb)


        if (hasattr(self, "accelerator") and self.accelerator.is_local_main_process) or (getattr(self, "_rank", 0) == 0):
            eval_logger.info("Metrics | MaxGPU=%.2f GB  Prefill=%.3f s  E2E=%.3f s  (ctx_len≈%d)",
                            peak_mem_gb, avg_prefill_exact, avg_e2e_s, avg_ctx_len)

        return res


#llava_vid.py from llms_eval/models/llava_vid.py
# import glob
# import math
# import os
# from datetime import timedelta
# from typing import List, Optional, Tuple, Union

# import numpy as np
# import torch
# from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
# from accelerate.state import AcceleratorState
# from decord import VideoReader, cpu
# from llava.constants import (
#     DEFAULT_IM_END_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IMAGE_TOKEN,
#     IGNORE_INDEX,
#     IMAGE_TOKEN_INDEX,
# )
# from llava.conversation import SeparatorStyle, conv_templates
# from llava.mm_utils import (
#     KeywordsStoppingCriteria,
#     get_model_name_from_path,
#     tokenizer_image_token,
# )
# from llava.model.builder import load_pretrained_model
# from llava.model.language_model.llava_llama import LlavaConfig
# from llava.model.language_model.llava_qwen import LlavaQwenConfig

# # eval_logger = logging.getLogger("lmms-eval")
# # import sys;sys.path.append("llava-video")
# from loguru import logger as eval_logger
# from PIL import Image
# from tqdm import tqdm
# from transformers import AutoConfig, AutoModelForCausalLM

# from lmms_eval.api.instance import Instance
# from lmms_eval.api.model import lmms
# from lmms_eval.api.registry import register_model
# from lmms_eval.models.model_utils.load_video import read_video_pyav

# try:
#     from llavavid.model.builder import load_pretrained_model
#     from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
#     from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
#     from llavavid.conversation import conv_templates, SeparatorStyle
#     from llavavid.mm_utils import tokenizer_image_token_qwen_merge, preprocess_qwen, preprocess_llama3
# except ImportError:
#     import llava
#     import pdb;pdb.set_trace()
#     if "llava-video-old" in llava.__file__:
#         from llava.model.language_model.llava_llama import LlavaConfig
#         from llava.model.language_model.llava_qwen import LlavaQwenConfig
#         from llava.model.builder import load_pretrained_model
#         from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
#         from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
#         from llava.conversation import conv_templates, SeparatorStyle

#         AutoConfig.register("llava_llama", LlavaConfig)
#         AutoConfig.register("llava_qwen", LlavaQwenConfig)
#     else:
#         eval_logger.debug("LLaVA-Video is not installed. Please install LLaVA-Video to use this model.")

# from llavavid.model.language_model.llava_qwen import LlavaQwenConfig
# from llavavid.model.language_model.llava_llama import LlavaConfig

# AutoConfig.register("llava_qwen", LlavaQwenConfig)
# AutoConfig.register("llava_llama", LlavaConfig)


# AutoConfig.register("llava_llama", LlavaConfig)
# AutoConfig.register("llava_qwen", LlavaQwenConfig)


# @register_model("llava_vid")
# class LlavaVid(lmms):
#     """
#     LlavaVid Model
#     """

#     def __init__(
#         self,
#         pretrained: str = "liuhaotian/llava-v1.5-7b",
#         truncation: Optional[bool] = True,
#         torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
#         device: Optional[str] = "cuda:0",
#         batch_size: Optional[Union[int, str]] = 1,
#         attn_implementation=(
#             "sdpa" if torch.__version__ >= "2.1.2" else "eager"
#         ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
#         device_map="cuda:0",
#         conv_template="vicuna_v1",
#         use_cache=True,
#         truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
#         max_frames_num: int = 20,
#         video_fps: int = 1,
#         mm_resampler_type: str = "spatial_pool",
#         mm_spatial_pool_stride: int = 2,
#         mm_spatial_pool_out_channels: int = 1024,
#         mm_spatial_pool_mode: str = "average",
#         mm_resampler_location: str = "before",
#         mm_newline_position: str = "grid",
#         overwrite: bool = True,
#         video_decode_backend: str = "decord",
#         delay_load: bool = False,
#         tie_weights: bool = True,
#         force_sample: bool = False,
#         add_time_instruction: bool = False,
#         add_faster_video: bool = False,
#         faster_token_stride: int = 10,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

#         accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
#         accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
#         if accelerator.num_processes > 1:
#             self._device = torch.device(f"cuda:{accelerator.local_process_index}")
#             self.device_map = f"cuda:{accelerator.local_process_index}"
#         elif accelerator.num_processes == 1 and (device_map == "auto" or device_map == "balanced_low_0"):
#             self._device = torch.device(device)
#             self.device_map = device_map
#         else:
#             self._device = torch.device(f"cuda:{accelerator.local_process_index}")
#             self.device_map = f"cuda:{accelerator.local_process_index}"

#         self.pretrained = pretrained
#         self.model_name = get_model_name_from_path(pretrained)
#         self.video_decode_backend = video_decode_backend
#         # self._config = AutoConfig.from_pretrained(self.pretrained)
#         self.overwrite = overwrite
#         self.mm_resampler_type = mm_resampler_type
#         self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
#         self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
#         self.mm_spatial_pool_mode = mm_spatial_pool_mode
#         self.max_frames_num = int(max_frames_num)
#         self.fps = int(video_fps)
#         self.mm_resampler_location = mm_resampler_location
#         self.delay_load = delay_load
#         self.force_sample = force_sample
#         self.add_time_instruction = add_time_instruction
#         print("force sample:", self.force_sample)
#         # self.add_faster_video = add_faster_video
#         # self.faster_token_stride = faster_token_stride
#         self.torch_dtype = torch_dtype
#         if self.overwrite == True:
#             overwrite_config = {}
#             # overwrite_config["mm_resampler_type"] = self.mm_resampler_type
#             overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
#             overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
#             overwrite_config["mm_pooling_position"] = self.mm_resampler_location
#             overwrite_config["mm_newline_position"] = mm_newline_position
#             overwrite_config["delay_load"] = self.delay_load
#             # overwrite_config["attn_implementation"] = attn_implementation

#             if "vicuna" in self.pretrained.lower() or "yi" in self.pretrained.lower():
#                 cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

#                 if "224" in cfg_pretrained.mm_vision_tower:
#                     # suppose the length of text tokens is around 1000, from bo's report
#                     least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
#                 else:
#                     least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

#                 scaling_factor = math.ceil(least_token_number / 4096)
#                 if scaling_factor >= 2:
#                     eval_logger.info(f"Scaling factor: {scaling_factor}")
#                     # print(float(scaling_factor))
#                     overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
#                     overwrite_config["max_sequence_length"] = 4096 * scaling_factor
#                     overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

#             self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
#                 pretrained, None, self.model_name, device_map=self.device_map, torch_dtype=self.torch_dtype, overwrite_config=overwrite_config, attn_implementation=attn_implementation
#             )
#         else:
#             self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map, torch_dtype=self.torch_dtype, attn_implementation=attn_implementation)

#         self._config = self._model.config

#         # import pdb;pdb.set_trace()

#         if self._tokenizer.pad_token_id is None:
#             if "qwen" in self._tokenizer.name_or_path.lower():
#                 print("Setting pad token to bos token for qwen model.")
#                 self._tokenizer.pad_token_id = 151643

#         self.model.eval()
#         if tie_weights:
#             self.model.tie_weights()
#         self.truncation = truncation
#         self.batch_size_per_gpu = int(batch_size)
#         self.conv_template = conv_template
#         self.use_cache = use_cache
#         self.truncate_context = truncate_context
#         # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
#         if accelerator.num_processes > 1:
#             assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
#             # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
#             # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
#             # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
#             if accelerator.distributed_type == DistributedType.DEEPSPEED:
#                 kwargs = {
#                     "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
#                     "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
#                 }
#                 AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
#                 eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
#             if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
#                 self._model = accelerator.prepare(self.model)
#             else:
#                 self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
#             self.accelerator = accelerator
#             if self.accelerator.is_local_main_process:
#                 eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
#             self._rank = self.accelerator.local_process_index
#             self._world_size = self.accelerator.num_processes
#         elif accelerator.num_processes == 1 and device_map == "auto":
#             eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
#             self._rank = 0
#             self._world_size = 1
#         else:
#             eval_logger.info(f"Using single device: {self._device}")
#             self.model.to(self._device)
#             self._rank = 0
#             self._world_size = 1

#     @property
#     def config(self):
#         # return the associated transformers.AutoConfig for the given pretrained model.
#         return self._config

#     @property
#     def tokenizer(self):
#         return self._tokenizer

#     @property
#     def model(self):
#         # returns the model, unwrapping it if using Accelerate
#         if hasattr(self, "accelerator"):
#             return self.accelerator.unwrap_model(self._model)
#         else:
#             return self._model

#     @property
#     def eot_token_id(self):
#         # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
#         return self.tokenizer.eos_token_id

#     @property
#     def max_length(self):
#         return self._max_length

#     def pad_sequence(self, input_ids, batch_first, padding_value):
#         if self.tokenizer.padding_side == "left":
#             input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
#         input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
#         if self.tokenizer.padding_side == "left":
#             input_ids = torch.flip(input_ids, [1])
#         return input_ids

#     @property
#     def batch_size(self):
#         return self.batch_size_per_gpu

#     @property
#     def device(self):
#         return self._device

#     @property
#     def rank(self):
#         return self._rank

#     @property
#     def world_size(self):
#         return self._world_size

#     def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
#         """ """
#         add_special_tokens = False if add_special_tokens is None else add_special_tokens
#         encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
#         # left-truncate the encoded context to be at most `left_truncate_len` tokens long
#         if left_truncate_len:
#             encoding = encoding[-left_truncate_len:]
#         return encoding

#     def load_image(self, image_path):
#         frame_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
#         frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

#         # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
#         num_frames_to_sample = 10

#         total_frames = len(frame_files)

#         sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

#         # Read and store the sampled frames
#         video = []
#         for idx in sampled_indices:
#             frame_path = frame_files[idx]
#             try:
#                 with Image.open(frame_path) as img:
#                     # Convert the PIL image to a numpy array if needed
#                     # frame = np.array(img.convert('RGB'))
#                     frame = img.convert("RGB")
#                     video.append(frame)
#             except IOError:
#                 print(f"Failed to read frame at path: {frame_path}")
#         return video

#     def load_video(self, video_path, max_frames_num, fps, force_sample=False):
#         if max_frames_num == 0:
#             return np.zeros((1, 336, 336, 3))
#         vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#         total_frame_num = len(vr)
#         video_time = total_frame_num / vr.get_avg_fps()
#         fps = round(vr.get_avg_fps() / fps)
#         frame_idx = [i for i in range(0, len(vr), fps)]
#         frame_time = [i / fps for i in frame_idx]
#         if len(frame_idx) > max_frames_num or force_sample:
#             sample_fps = max_frames_num
#             uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
#             frame_idx = uniform_sampled_frames.tolist()
#             frame_time = [i / vr.get_avg_fps() for i in frame_idx]
#         frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
#         spare_frames = vr.get_batch(frame_idx).asnumpy()
#         # import pdb;pdb.set_trace()

#         return spare_frames, frame_time, video_time

#     def tok_decode(self, tokens):
#         return self.tokenizer.decode(tokens)

#     def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
#         res = []
#         pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

#         for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
#             # encode, pad, and truncate contexts for this batch
#             if type(doc_to_target) == str:
#                 continuation = doc_to_target
#             else:
#                 continuation = doc_to_target(self.task_dict[task][split][doc_id])
#             visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
#             visuals = self.flatten(visuals)
#             videos = []
#             for visual in visuals:
#                 video, frame_time, video_time = self.load_video(visual, self.max_frames_num, self.fps, force_sample=self.force_sample)
#                 video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
#                 if self.torch_dtype == "bfloat16":
#                     video = video.bfloat16()
#                 else:
#                     video = video.half()
#                 videos.append(video)

#             qs = contexts
#             if self.model.config.mm_use_im_start_end:
#                 qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
#             else:
#                 qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#             conv = conv_templates[self.conv_template].copy()
#             conv.append_message(conv.roles[0], qs)
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()

#             contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

#             conv = conv_templates[self.conv_template].copy()
#             conv.append_message(conv.roles[0], qs)
#             conv.append_message(conv.roles[1], continuation)
#             prompt = conv.get_prompt()

#             input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
#             attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

#             labels = input_ids.clone()
#             # Context part no need to calculate for loss
#             labels[0, : contxt_id.shape[1]] = -100

#             with torch.inference_mode():
#                 outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

#             loss = outputs["loss"]
#             # loss = torch.exp(loss)
#             logits = outputs["logits"]
#             greedy_tokens = logits.argmax(dim=-1)
#             cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
#             greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
#             max_equal = (greedy_tokens == cont_toks).all()
#             res.append((float(loss.item()), bool(max_equal)))
#             pbar.update(1)
#         pbar.close()
#         return res

#     def flatten(self, input):
#         new_list = []
#         for i in input:
#             for j in i:
#                 new_list.append(j)
#         return new_list

#     def generate_until(self, requests) -> List[str]:
#         res = []
#         pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

#         for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
#             # if self.task_dict[task][split][doc_id]["duration"] != "short":
#             # # if doc_id != 112:
#             #     # import pdb;pdb.set_trace()
#             #     res.append("A")
#             #     pbar.update(1)
#             #     continue
#             # encode, pad, and truncate contexts for this batch
#             # import pdb;pdb.set_trace()
#             visuals = doc_to_visual(self.task_dict[task][split][doc_id])
#             # visuals = [visuals]
#             # visuals = self.flatten(visuals)
#             if os.path.isdir(visuals[0]):
#                 visuals = glob.glob(visuals[0] + "/*")
#             videos = []
#             try:
#                 # for visual in visuals:
#                 if len(visuals) == 1:
#                     if self.video_decode_backend == "decord":
#                         video, frame_time, video_time = self.load_video(visuals[0], self.max_frames_num, self.fps, force_sample=self.force_sample)
#                     elif self.video_decode_backend == "pyav":
#                         video, frame_time, video_time = read_video_pyav(visuals[0], self.max_frames_num, self.fps, force_sample=self.force_sample)
#                     elif self.video_decode_backend == "image":
#                         video = self.load_image(visuals[0])
#                 else:
#                     if task == "seedbench":
#                         video = visuals
#                         frame_time = "1.00s"
#                         video_time = 1
#                     elif "mvbench" in task:
#                         # video = visuals
#                         # Reference: https://github.com/jayleicn/TVQA/blob/dfb0e5fe4582efca574dfddfeafd1008db3b33ef/data/README.md?plain=1#L50C34-L50C60
#                         fps = 3
#                         video_time = len(visuals) / fps
#                         sampled_indices = np.linspace(0, len(visuals) - 1, self.max_frames_num, dtype=int)
#                         frame_idx = sampled_indices.tolist()
#                         frame_time = [i / fps for i in frame_idx]
#                         frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
#                         # video = [visuals[i] for i in frame_idx]
#                         video = np.stack([np.array(Image.open(visuals[i])) for i in frame_idx], axis=0)

#                 video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
#                 if self.torch_dtype == "bfloat16":
#                     video = video.bfloat16()
#                 else:
#                     video = video.half()
#                 videos.append(video)
#             except Exception as e:
#                 # import pdb;pdb.set_trace()
#                 eval_logger.info(f"{e}")
#                 eval_logger.info(f"Video {visuals} can not load, check the source")
#                 video_path = "\n".join(visuals)
#                 res.append(f"Video {video_path} can not load, check the source")
#                 pbar.update(1)
#                 continue

#             qs = contexts
#             # import pdb;pdb.set_trace()
#             if self.add_time_instruction:
#                 time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
#                 qs = f"{time_instruciton}\n{qs}"
#             if self.model.config.mm_use_im_start_end:
#                 qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
#             else:
#                 qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs

#             # This is much safer for llama3, as we now have some object type in it
#             if "llama_3" in self.conv_template:
#                 conv = copy.deepcopy(conv_templates[self.conv_template])
#             else:
#                 conv = conv_templates[self.conv_template].copy()

#             conv.append_message(conv.roles[0], qs)
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()

#             input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
#             pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
#             if "llama_3" in self.conv_template:
#                 pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
#             attention_masks = input_ids.ne(pad_token_ids).long().cuda()

#             stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#             keywords = [stop_str]

#             stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

#             cur_prompt = qs

#             if "max_new_tokens" not in gen_kwargs:
#                 gen_kwargs["max_new_tokens"] = 1024
#             if "temperature" not in gen_kwargs:
#                 gen_kwargs["temperature"] = 0
#             if "top_p" not in gen_kwargs:
#                 gen_kwargs["top_p"] = None
#             if "num_beams" not in gen_kwargs:
#                 gen_kwargs["num_beams"] = 1

#             # import pdb;pdb.set_trace()
#             with torch.inference_mode():
#                 output_ids = self.model.generate(
#                     inputs=input_ids,
#                     images=videos,
#                     attention_mask=attention_masks,
#                     modalities="video",
#                     use_cache=self.use_cache,
#                     stopping_criteria=[stopping_criteria],
#                     do_sample=True if gen_kwargs["temperature"] > 0 else False,
#                     temperature=gen_kwargs["temperature"],
#                     top_p=gen_kwargs["top_p"],
#                     num_beams=gen_kwargs["num_beams"],
#                     max_new_tokens=gen_kwargs["max_new_tokens"],
#                 )
#                 # output_ids_2 = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=False, max_new_tokens=50,stopping_criteria=[stopping_criteria])
#                 # output_ids = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=50,use_cache=True)

#             outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#             eval_logger.debug(f"Question: {cur_prompt}")
#             eval_logger.debug(f"Answer: {outputs}")
#             # import pdb;pdb.set_trace()
#             res.append(outputs)
#             pbar.update(1)
#         return res

#     def generate_until_multi_round(self, requests) -> List[str]:
#         raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")