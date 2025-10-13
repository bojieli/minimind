#!/usr/bin/env python3
"""
Tokenizer 翻译工具

将 ByteLevel BPE tokenizer.json 文件中的所有"乱码"token翻译为原始文本。
这会修改 tokenizer.json 中的 vocab、merges 和 added_tokens 部分。
"""

import json
import argparse
from pathlib import Path
import re


def bytes_to_unicode():
    """
    生成字节到Unicode字符的正向映射表
    这个映射来自 GPT-2 的 tokenizer 实现
    
    返回字典：{byte_value: unicode_char}
    """
    # 保留的可打印字符范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) +      # ASCII可打印字符（33-126）
        list(range(ord("¡"), ord("¬") + 1)) +      # 扩展字符（161-172）
        list(range(ord("®"), ord("ÿ") + 1))        # 扩展字符（174-255）
    )
    
    cs = bs.copy()
    n = 0
    
    # 将未被保留的字节映射到更高的Unicode码位
    for b in range(2**8):  # 遍历所有字节（0-255）
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)  # 映射到256+n的Unicode码位
            n += 1
    
    # 转换为Unicode字符
    cs = [chr(code) for code in cs]
    
    return dict(zip(bs, cs))


def get_reverse_mapping(forward_map):
    """
    根据正向映射生成反向映射
    
    返回字典：{unicode_char: byte_value}
    """
    return {v: k for k, v in forward_map.items()}


def unicode_str_to_bytes(unicode_str, reverse_map):
    """
    将映射后的Unicode字符串转换回原始字节序列
    
    Args:
        unicode_str: 映射后的Unicode字符串（例如："ä½łå¥½"）
        reverse_map: 反向映射表
    
    Returns:
        bytes: 原始字节序列
    """
    return bytes([reverse_map[c] for c in unicode_str])


def decode_token(token_str, reverse_map):
    """
    解码单个token字符串为原始文本
    
    Args:
        token_str: token字符串（可能是乱码）
        reverse_map: 反向映射表
    
    Returns:
        str: 解码后的原始文本（如果无法解码或解码为空则返回原字符串）
    """
    try:
        # 将Unicode字符串转回字节
        byte_sequence = unicode_str_to_bytes(token_str, reverse_map)
        # 尝试解码为UTF-8文本
        decoded = byte_sequence.decode('utf-8', errors='ignore')
        # 如果解码结果为空，返回原字符串
        if not decoded:
            return token_str
        return decoded
    except Exception:
        # 如果解码失败，返回原字符串
        return token_str


def translate_vocab(vocab, reverse_map):
    """
    翻译vocab字典
    
    Args:
        vocab: {token: id} 映射
        reverse_map: 反向映射表
    
    Returns:
        dict: 翻译后的vocab
    """
    translated = {}
    for token, token_id in vocab.items():
        decoded = decode_token(token, reverse_map)
        translated[decoded] = token_id
    return translated


def translate_merges(merges, reverse_map):
    """
    翻译merges列表
    
    Args:
        merges: [[token1, token2], ...] 格式的merge规则列表
        reverse_map: 反向映射表
    
    Returns:
        list: 翻译后的merges
    """
    translated = []
    for merge in merges:
        # merges 格式可能是 [token1, token2] 或 "token1 token2"
        if isinstance(merge, list) and len(merge) == 2:
            token1, token2 = merge
            decoded1 = decode_token(token1, reverse_map)
            decoded2 = decode_token(token2, reverse_map)
            translated.append([decoded1, decoded2])
        elif isinstance(merge, str):
            parts = merge.split(' ')
            if len(parts) == 2:
                token1, token2 = parts
                decoded1 = decode_token(token1, reverse_map)
                decoded2 = decode_token(token2, reverse_map)
                translated.append(f"{decoded1} {decoded2}")
            else:
                # 保持原样
                translated.append(merge)
        else:
            # 保持原样（不应该发生）
            translated.append(merge)
    return translated


def translate_added_tokens(added_tokens, reverse_map):
    """
    翻译added_tokens列表
    
    Args:
        added_tokens: [{"content": token, ...}, ...] 格式的列表
        reverse_map: 反向映射表
    
    Returns:
        list: 翻译后的added_tokens
    """
    translated = []
    for token_info in added_tokens:
        token_info_copy = token_info.copy()
        if 'content' in token_info_copy:
            content = token_info_copy['content']
            decoded = decode_token(content, reverse_map)
            token_info_copy['content'] = decoded
        translated.append(token_info_copy)
    return translated


def translate_tokenizer(input_path, output_path=None):
    """
    翻译整个tokenizer.json文件
    
    Args:
        input_path: 输入的tokenizer.json路径
        output_path: 输出路径（如果为None，则覆盖原文件）
    """
    print("🔄 加载映射表...")
    forward_map = bytes_to_unicode()
    reverse_map = get_reverse_mapping(forward_map)
    
    print(f"📖 加载tokenizer: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    print("✨ 翻译tokenizer...")
    
    # 翻译 model.vocab
    if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
        print("  - 翻译 vocab...")
        original_vocab_size = len(tokenizer_data['model']['vocab'])
        tokenizer_data['model']['vocab'] = translate_vocab(
            tokenizer_data['model']['vocab'], 
            reverse_map
        )
        print(f"    ✓ 已翻译 {original_vocab_size} 个tokens")
    
    # 翻译 model.merges
    if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
        print("  - 翻译 merges...")
        original_merges_count = len(tokenizer_data['model']['merges'])
        tokenizer_data['model']['merges'] = translate_merges(
            tokenizer_data['model']['merges'], 
            reverse_map
        )
        print(f"    ✓ 已翻译 {original_merges_count} 个merge规则")
    
    # 翻译 added_tokens
    if 'added_tokens' in tokenizer_data:
        print("  - 翻译 added_tokens...")
        original_added_count = len(tokenizer_data['added_tokens'])
        tokenizer_data['added_tokens'] = translate_added_tokens(
            tokenizer_data['added_tokens'], 
            reverse_map
        )
        print(f"    ✓ 已翻译 {original_added_count} 个特殊tokens")
    
    # 保存结果
    if output_path is None:
        output_path = input_path
    
    print(f"\n💾 保存翻译后的tokenizer到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 翻译完成！")
    
    # 显示一些统计信息
    if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
        vocab = tokenizer_data['model']['vocab']
        print(f"\n📊 统计信息:")
        print(f"   总词汇量: {len(vocab)}")
        
        # 统计一些示例
        print(f"\n📝 翻译示例 (前10个):")
        for i, (token, token_id) in enumerate(sorted(vocab.items(), key=lambda x: x[1])):
            if i >= 10:
                break
            print(f"   [{token_id}] {repr(token)}")


def main():
    parser = argparse.ArgumentParser(
        description='Tokenizer 翻译工具 - 将ByteLevel编码的token翻译为原始文本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译tokenizer.json并保存到新文件
  python3 translate_tokenizer.py ../model/tokenizer.json -o ../model/tokenizer_translated.json
  
  # 翻译并覆盖原文件（谨慎使用！建议先备份）
  python3 translate_tokenizer.py ../model/tokenizer.json
  
注意:
  - 这会修改tokenizer的内部表示，使其更易读
  - 翻译后的tokenizer功能上应该保持一致
  - 建议在翻译前备份原文件
        """
    )
    
    parser.add_argument('input_path', help='输入的tokenizer.json文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（默认覆盖原文件）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在: {args.input_path}")
        return 1
    
    # 如果没有指定输出路径，提示用户
    if args.output is None:
        print("⚠️  警告: 未指定输出路径，将覆盖原文件！")
        try:
            response = input("是否继续？(y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ 操作已取消")
                return 0
        except KeyboardInterrupt:
            print("\n❌ 操作已取消")
            return 0
    
    try:
        translate_tokenizer(args.input_path, args.output)
        return 0
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

