#!/usr/bin/env python3
"""Convert GML to *_nodes.txt and *_edges.txt (Python 3, no dependencies).

Usage: python3 gml_to_txt.py [input.gml] [-o output_directory]
Without an input filename, converts all GML files beside this script.
Nodes: id power mass gamma. Edges: source target weight.
Preserves GML record order; writes spaces, 15 significant digits, LF, no header.
Ignores unused attributes, including label, mask_gen, and repeated pos values.
"""

import argparse
import math
from pathlib import Path
import re


TOKEN = re.compile(
    r'\s+|\#[^\n]*|"[^"]*"|[A-Za-z_][A-Za-z_0-9]*|'
    r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[\[\]]'
)


def parse_gml(text):
    """Parse nested GML key/value lists without losing repeated keys."""
    tokens = []
    position = 0
    while position < len(text):
        match = TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"Invalid GML near character {position}: {text[position:position+30]!r}")
        token = match.group()
        if not token.isspace() and not token.startswith('#'):
            tokens.append(token)
        position = match.end()
    index = 0

    def pairs(nested=False):
        nonlocal index
        result = []
        while index < len(tokens):
            key = tokens[index]
            index += 1
            if key == ']':
                if not nested:
                    raise ValueError('Unexpected closing bracket')
                return result
            if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
                raise ValueError(f'Expected attribute name, got {key!r}')
            if index == len(tokens):
                raise ValueError(f'Missing value for {key!r}')
            value = tokens[index]
            index += 1
            if value == '[':
                value = pairs(nested=True)
            elif value.startswith('"'):
                value = value[1:-1]
            elif re.fullmatch(r'[+-]?\d+', value):
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f'Invalid value for {key!r}: {value!r}') from None
            result.append((key, value))
        if nested:
            raise ValueError('Missing closing bracket')
        return result

    return pairs()


def required(record, name):
    if not isinstance(record, list):
        raise ValueError('Expected a GML attribute list')
    values = [v for k, v in record if k == name]
    if len(values) != 1:
        raise ValueError(f'Expected exactly one {name!r} attribute, found {len(values)}')
    return values[0]


def integer(record, name):
    value = required(record, name)
    if not isinstance(value, int):
        raise ValueError(f'{name!r} must be an integer, got {value!r}')
    return value


def number(record, name):
    value = required(record, name)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name!r} must be a finite number, got {value!r}')
    return format(value, '.15g')


def convert_gml(input_path, output_dir=None):
    input_path = Path(input_path)
    graph = required(parse_gml(input_path.read_text(encoding='utf-8-sig')), 'graph')
    if not isinstance(graph, list):
        raise ValueError('graph must contain a GML attribute list')
    node_lines, edge_lines, node_ids = [], [], set()
    for key, record in graph:
        if key == 'node':
            node_id = integer(record, 'id')
            if node_id in node_ids:
                raise ValueError(f'Duplicate node id: {node_id}')
            node_ids.add(node_id)
            node_lines.append(' '.join([str(node_id)] + [
                number(record, name) for name in ('power', 'mass', 'gamma')
            ]))
    for key, record in graph:
        if key == 'edge':
            source = integer(record, 'source')
            target = integer(record, 'target')
            if source not in node_ids or target not in node_ids:
                raise ValueError(f'Edge references an undefined node: {source} -> {target}')
            edge_lines.append(f"{source} {target} {number(record, 'weight')}")

    output_dir = Path(output_dir) if output_dir is not None else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, lines in [('nodes', node_lines), ('edges', edge_lines)]:
        path = output_dir / f'{input_path.stem}_{suffix}.txt'
        path.write_bytes(''.join(line + '\n' for line in lines).encode('utf-8'))
        paths.append(path)
    return tuple(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gml', nargs='?', type=Path,
                        help='Optional: defaults to all GML files beside this script')
    parser.add_argument('-o', '--output-dir', type=Path, help='Default: input file directory')
    args = parser.parse_args(argv)
    script_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    inputs = [args.gml] if args.gml is not None else sorted(
        path for path in script_dir.iterdir()
        if path.is_file() and path.suffix.lower() == '.gml'
    )
    if not inputs:
        parser.exit(1, f'GML 파일이 없습니다. 이 스크립트와 같은 폴더에 넣어 주세요: {script_dir}\n')
    failed = False
    for input_path in inputs:
        try:
            paths = convert_gml(input_path, args.output_dir)
        except (OSError, ValueError) as exc:
            print(f'Error ({input_path.name}): {exc}')
            failed = True
            continue
        for path in paths:
            print(f'생성 완료: {path}')
    if failed:
        parser.exit(1)


if __name__ == '__main__':
    if '__file__' in globals():
        main()
    else:
        print('Jupyter: convert_gml("54_sample0.gml") 또는 main(["54_sample0.gml"])을 실행하세요.')
