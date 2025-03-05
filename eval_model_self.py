# coding=gb2312
import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import *

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        #���õ�ģ���������
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}_test_{args.extra}.pth'

        model = MiniMindLM(LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        checkpoint = torch.load(ckp, map_location=args.device)
        state_dict = checkpoint['model']
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    print(f'MiniMindģ�Ͳ�����: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrainģ�͵Ľ����������޷��Ի���
        prompt_datas = [
            '���˼�������ԭ��',
            '������Ե���Ҫ����',
            '��������ԭ����',
            '��������ߵ�ɽ����',
            '������̼�ڿ�����',
            '���������Ķ�����',
            '�����е���ʳ��'
        ]
    else:
        if args.lora_name == 'None':
            # ͨ�öԻ�����
            prompt_datas = [
                '�����һ���Լ���',
                '����ó���һ��ѧ�ƣ�',
                '³Ѹ�ġ������ռǡ���������з⽨��̵ģ�',
                '�ҿ����Ѿ����������ܣ���ҪȥҽԺ�����',
                '��ϸ�Ľ��ܹ��ٵ�������',
                '�Ƽ�һЩ���ݵ���ɫ��ʳ�ɡ�',
                '��Ϊ�ҽ��⡰������ģ�͡�������',
                '������ChatGPT��',
                'Introduce the history of the United States, please.'
            ]
        else:
            # �ض���������
            lora_prompt_datas = {
                'lora_identity': [
                    "����ChatGPT�ɡ�",
                    "���ʲô���֣�",
                    "���openai��ʲô��ϵ��"
                ],
                'lora_medical': [
                    '����������е�ͷ�Σ�������ʲôԭ��',
                    '�ҿ����Ѿ����������ܣ���ҪȥҽԺ�����',
                    '���ÿ�����ʱ��Ҫע����Щ���',
                    '��챨������ʾ���̴�ƫ�ߣ��Ҹ���ô�죿',
                    '�и�����ʳ����Ҫע��ʲô��',
                    '���������Ԥ���������ɣ�',
                    '��������Ǹе����ǣ�Ӧ����ô���⣿',
                    '�������ͻȻ�ε���Ӧ����μ��ȣ�'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# ���ÿɸ��ֵ��������
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # �˴�max_seq_len������������볤�ȣ�������ζģ�;��ж�Ӧ�ĳ��ı������ܣ�����ֹQA���ֱ��ضϵ�����
    # MiniMind2-moe (145M)��(dim=640, n_layers=8, use_moe=True)
    # MiniMind2-Small (26M)��(dim=512, n_layers=8)
    # MiniMind2 (104M)��(dim=768, n_layers=16)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # Я����ʷ�Ի�����������
    # history_cnt��Ҫ��Ϊż���������û�����, ģ�ͻش�Ϊ1�飻����Ϊ0ʱ������ǰquery��Я����ʷ����
    # ģ��δ��������΢��ʱ���ڸ����������ĵ�chat_templateʱ����������ܵ������˻��������Ҫע��˴�����
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: ԭ��torchȨ�أ�1: transformers����")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: Ԥѵ��ģ�ͣ�1: SFT-Chatģ�ͣ�2: RLHF-Chatģ�ͣ�3: Reasonģ��")
    parser.add_argument("--extra", type=str, default='', help = 'extra information')
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input('[0] �Զ�����\n[1] �ֶ�����\n'))
    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('?: '), '')):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # ����̶�ÿ������򻻳ɡ��̶������������
        if test_mode == 0: print(f'?: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        answer = new_prompt
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('??: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '?') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
