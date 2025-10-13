#!/usr/bin/env python3
"""
BPE ByteLevel Tokenizer Vocab 解码工具

用于将 ByteLevel BPE tokenizer 的 vocab.json 中的"乱码"token 还原为原始文本。
适用于 Qwen、GPT-2 等使用 ByteLevel 编码的模型。
"""

import json
import argparse
from pathlib import Path


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


def load_vocab(vocab_path):
    """
    加载vocab.json文件
    
    Args:
        vocab_path: vocab.json的路径
    
    Returns:
        dict: token到ID的映射
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def decode_vocab(vocab_path, output_path=None, show_samples=10):
    """
    解码整个vocab文件
    
    Args:
        vocab_path: vocab.json的路径
        output_path: 输出文件路径（可选）
        show_samples: 显示的样本数量
    """
    print("🔄 加载映射表...")
    forward_map = bytes_to_unicode()
    reverse_map = get_reverse_mapping(forward_map)
    
    print(f"📖 加载词汇表: {vocab_path}")
    vocab = load_vocab(vocab_path)
    
    print(f"✨ 解码 {len(vocab)} 个tokens...")
    decoded_vocab = {}
    
    for token_str, token_id in vocab.items():
        decoded = decode_token(token_str, reverse_map)
        decoded_vocab[token_str] = decoded
    
    # 显示示例
    print(f"\n📊 显示前 {show_samples} 个解码示例：")
    print("-" * 80)
    print(f"{'ID':<8} {'原始Token':<30} {'解码后':<30}")
    print("-" * 80)
    
    # 按ID排序显示
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    for i, (token_str, token_id) in enumerate(sorted_items):
        if i >= show_samples:
            break
        decoded = decoded_vocab[token_str]
        original = token_str[:28]  # 截断过长的字符串
        decoded_display = decoded[:28]
        print(f"{token_id:<8} {original:<30} {decoded_display:<30}")
    
    # 保存到文件
    if output_path:
        print(f"\n💾 保存解码结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(decoded_vocab, f, ensure_ascii=False, indent=2)
        print("✅ 保存完成！")
    
    return decoded_vocab


def query_token(vocab_path, query):
    """
    查询特定token的解码结果
    
    Args:
        vocab_path: vocab.json的路径
        query: 要查询的token字符串或ID
    """
    forward_map = bytes_to_unicode()
    reverse_map = get_reverse_mapping(forward_map)
    vocab = load_vocab(vocab_path)
    
    # 如果query是数字，按ID查找
    if query.isdigit():
        token_id = int(query)
        for token_str, tid in vocab.items():
            if tid == token_id:
                decoded = decode_token(token_str, reverse_map)
                print(f"\n📌 Token ID: {token_id}")
                print(f"   原始: {token_str}")
                print(f"   解码: {decoded}")
                return
        print(f"❌ 未找到 ID: {token_id}")
    else:
        # 按token字符串查找
        if query in vocab:
            decoded = decode_token(query, reverse_map)
            print(f"\n📌 Token: {query}")
            print(f"   ID: {vocab[query]}")
            print(f"   解码: {decoded}")
        else:
            print(f"❌ 未找到 token: {query}")


def interactive_mode(vocab_path):
    """
    交互式查询模式
    
    Args:
        vocab_path: vocab.json的路径
    """
    forward_map = bytes_to_unicode()
    reverse_map = get_reverse_mapping(forward_map)
    vocab = load_vocab(vocab_path)
    
    print("\n🎮 进入交互模式 (输入 'quit' 退出)")
    print("提示：可以输入token字符串或ID进行查询")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n🔍 请输入查询 > ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not query:
                continue
            
            # 按ID查询
            if query.isdigit():
                token_id = int(query)
                found = False
                for token_str, tid in vocab.items():
                    if tid == token_id:
                        decoded = decode_token(token_str, reverse_map)
                        print(f"  📌 ID: {token_id}")
                        print(f"     原始: {repr(token_str)}")
                        print(f"     解码: {decoded}")
                        found = True
                        break
                if not found:
                    print(f"  ❌ 未找到 ID: {token_id}")
            else:
                # 按token字符串查询
                if query in vocab:
                    decoded = decode_token(query, reverse_map)
                    print(f"  📌 Token: {repr(query)}")
                    print(f"     ID: {vocab[query]}")
                    print(f"     解码: {decoded}")
                else:
                    print(f"  ❌ 未找到 token: {query}")
        
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"  ⚠️  错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='BPE ByteLevel Tokenizer Vocab 解码工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 解码整个vocab文件并显示前20个样本
  python3 decode_vocab.py ../model/vocab.json -n 20
  
  # 解码并保存结果到文件
  python3 decode_vocab.py ../model/vocab.json -o ../model/vocab_decoded.json
  
  # 查询特定token (按ID)
  python3 decode_vocab.py ../model/vocab.json -q 5892
  
  # 查询特定token (按字符串)
  python3 decode_vocab.py ../model/vocab.json -q "çļĦä¸įåĲĮ"
  
  # 交互式查询模式
  python3 decode_vocab.py ../model/vocab.json -i
        """
    )
    
    parser.add_argument('vocab_path', help='vocab.json文件路径')
    parser.add_argument('-o', '--output', help='输出解码结果到文件')
    parser.add_argument('-n', '--samples', type=int, default=10, 
                        help='显示的样本数量（默认：10）')
    parser.add_argument('-q', '--query', help='查询特定token（ID或字符串）')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='进入交互式查询模式')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.vocab_path).exists():
        print(f"❌ 错误: 文件不存在: {args.vocab_path}")
        return 1
    
    try:
        if args.interactive:
            # 交互模式
            interactive_mode(args.vocab_path)
        elif args.query:
            # 查询模式
            query_token(args.vocab_path, args.query)
        else:
            # 解码模式
            decode_vocab(args.vocab_path, args.output, args.samples)
        
        return 0
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
