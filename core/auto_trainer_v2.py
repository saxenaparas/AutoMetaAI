"""
AutoMetaAI - auto_trainer_v2.py

Updated version: provides safe defaults and will create a skeleton
pattern_knowledge.json if the file is missing. This lets you run the
script with no arguments during development.

Run examples:
  # default predict (no args)
  python core/auto_trainer_v2.py

  # thinking mode with defaults (only provide these flags if you want to change behavior)
  python core/auto_trainer_v2.py --mode think --max_iter 5 --min_occ 5 --min_conf 0.85

Note: the file will update the patterns JSON in-place during THINKING mode.
Backup your patterns file if it is important.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd


def tokenize_text(s):
    if pd.isna(s):
        return []
    s = str(s).lower()
    s = re.sub(r"[_/\\\-\.\(\)\[\],:;@#%]", ' ', s)
    tokens = re.findall(r"[a-z0-9%°]+", s)
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit() or t in ['%', '°']]
    return tokens


def load_patterns(path):
    path = Path(path)
    if not path.exists():
        print(f"Patterns file not found at {path}. Creating an empty skeleton patterns file.")
        skeleton = {
            'meta': {
                'created_at': pd.Timestamp.now().isoformat(),
                'notes': 'Auto-created skeleton by auto_trainer_v2.py'
            },
            'patterns': [],
            'field_relations': [],
            'normalizations': [],
            'examples': []
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(skeleton, fh, indent=2, ensure_ascii=False)
        return skeleton, []

    with open(path, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    patterns = doc.get('patterns', [])
    norm = []
    for p in patterns:
        token = p.get('pattern_token') or p.get('pattern_regex')
        norm.append({
            'pattern_token': (token.lower() if isinstance(token, str) else None),
            'pattern_regex': p.get('pattern_regex'),
            'target_column': p.get('target_column'),
            'predicted_value': p.get('predicted_value'),
            'occurrence': int(p.get('occurrence', 0)),
            'confidence': float(p.get('confidence', 0.0)),
            'sources': p.get('sources', [])
        })
    return doc, norm


def save_patterns_doc(doc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)


def apply_patterns_to_df(df, patterns, text_fields=('description', 'dataTagId')):
    out = df.copy(deep=True)
    row_tokens = []
    for _, r in out.iterrows():
        toks = []
        for f in text_fields:
            if f in out.columns:
                toks += tokenize_text(r.get(f, ''))
        row_tokens.append(set(toks))

    patterns_sorted = sorted(patterns, key=lambda p: (p.get('occurrence', 0) * p.get('confidence', 0.0)), reverse=True)

    for idx, toks in enumerate(row_tokens):
        for p in patterns_sorted:
            token = p.get('pattern_token')
            if not token:
                continue
            matched = False
            if p.get('pattern_regex'):
                joined = ' '.join([str(out.at[idx, f]) for f in text_fields if f in out.columns])
                try:
                    if re.search(p['pattern_regex'], joined, flags=re.IGNORECASE):
                        matched = True
                except re.error:
                    if token in toks:
                        matched = True
            else:
                if token in toks:
                    matched = True
            if not matched:
                continue
            target = p.get('target_column')
            val = p.get('predicted_value')
            if target in out.columns:
                cur = out.at[idx, target]
                if pd.isna(cur) or str(cur).strip() == '':
                    out.at[idx, target] = val
            else:
                out.loc[idx, target] = val
    return out


def evaluate_predictions(pred_df, truth_df, text_fields=('description','dataTagId')):
    common_cols = [c for c in pred_df.columns if c in truth_df.columns and c not in text_fields]
    stats = {}
    for c in common_cols:
        pred = pred_df[c].fillna('').astype(str).str.strip()
        truth = truth_df[c].fillna('').astype(str).str.strip()
        mask = (truth != '')
        if mask.sum() == 0:
            acc = None
        else:
            same = (pred[mask] == truth[mask]).sum()
            acc = same / mask.sum()
        stats[c] = {'accuracy': acc, 'n_truth': int(mask.sum())}
    total = sum(v['n_truth'] for v in stats.values() if v['accuracy'] is not None)
    overall = None
    if total > 0:
        overall = sum(v['accuracy'] * v['n_truth'] for v in stats.values() if v['accuracy'] is not None) / total
    return stats, overall


def discover_new_patterns(truth_df, pred_df, text_fields=('description','dataTagId'), min_occurrence=5, min_confidence=0.85, exclude_existing_tokens=None):
    if exclude_existing_tokens is None:
        exclude_existing_tokens = set()
    token_rows = defaultdict(list)
    for i, r in truth_df.iterrows():
        toks = []
        for f in text_fields:
            if f in truth_df.columns:
                toks += tokenize_text(r.get(f, ''))
        toks = list(dict.fromkeys(toks))
        for t in toks:
            token_rows[t].append(i)
    new_patterns = []
    target_cols = [c for c in truth_df.columns if c in pred_df.columns and c not in text_fields]
    diff_mask = []
    for i in range(len(truth_df)):
        row_diff = False
        for c in target_cols:
            tr = str(truth_df.at[i,c]) if not pd.isna(truth_df.at[i,c]) else ''
            pr = str(pred_df.at[i,c]) if not pd.isna(pred_df.at[i,c]) else ''
            if tr.strip() != '' and tr.strip() != pr.strip():
                row_diff = True
                break
        diff_mask.append(row_diff)
    for token, rows in token_rows.items():
        if token in exclude_existing_tokens:
            continue
        occ = len(rows)
        if occ < min_occurrence:
            continue
        mis_rows = [r for r in rows if diff_mask[r]]
        if len(mis_rows) < max(3, int(min_occurrence/2)):
            continue
        for col in target_cols:
            vals = []
            for r in rows:
                v = truth_df.at[r,col]
                if pd.isna(v) or str(v).strip() == '':
                    continue
                vals.append(str(v).strip())
            if not vals:
                continue
            cnt = Counter(vals)
            top_val, top_count = cnt.most_common(1)[0]
            confidence = top_count / len(vals)
            if confidence >= min_confidence and len(vals) >= min_occurrence:
                new_patterns.append({
                    'pattern_token': token,
                    'pattern_regex': rf"\\b{re.escape(token)}\\b",
                    'target_column': col,
                    'predicted_value': top_val,
                    'occurrence': occ,
                    'confidence': round(confidence,3),
                    'source': 'discovered_in_thinking'
                })
    return new_patterns


def merge_new_patterns_into_doc(doc, new_patterns):
    existing = {(p.get('pattern_token'), p.get('target_column'), str(p.get('predicted_value'))): p for p in doc.get('patterns', [])}
    added = 0
    updated = 0
    for npat in new_patterns:
        key = (npat['pattern_token'], npat['target_column'], str(npat['predicted_value']))
        if key in existing:
            p = existing[key]
            p['occurrence'] = int(p.get('occurrence',0)) + int(npat.get('occurrence',0))
            p['confidence'] = round((float(p.get('confidence',0)) + float(npat.get('confidence',0))) / 2, 3)
            if 'sources' not in p:
                p['sources'] = []
            if npat.get('source') not in p['sources']:
                p['sources'].append(npat.get('source'))
            updated += 1
        else:
            doc.setdefault('patterns', []).append({
                'pattern_token': npat['pattern_token'],
                'pattern_regex': npat['pattern_regex'],
                'target_column': npat['target_column'],
                'predicted_value': npat['predicted_value'],
                'occurrence': npat['occurrence'],
                'confidence': npat['confidence'],
                'sources': [npat.get('source')]
            })
            added += 1
    return added, updated


def main():
    p = argparse.ArgumentParser()
    default_patterns = r'D:\Projects\AutoMetaAI\pattern_knowledge.json'
    default_input = r'D:\Projects\AutoMetaAI\data\input\sample_data.xlsx'
    default_output = r'D:\Projects\AutoMetaAI\data\output\PREDICTED_sample_data.xlsx'
    default_verif = r'D:\Projects\AutoMetaAI\data\verification\Filled_Sample_Data_For_Verification.xlsx'
    p.add_argument('--patterns', default=default_patterns, help='path to pattern_knowledge.json (default project path)')
    p.add_argument('--input', default=default_input, help='input xlsx with tag_id + description to predict (default sample)')
    p.add_argument('--output', default=default_output, help='path to save predicted xlsx (default output path)')
    p.add_argument('--verification', default=default_verif, help='ground truth xlsx for thinking mode (optional)')
    p.add_argument('--mode', choices=['predict','think'], default='predict')
    p.add_argument('--max_iter', type=int, default=5, help='max thinking iterations')
    p.add_argument('--min_occ', type=int, default=5, help='min occurrence to accept discovered pattern')
    p.add_argument('--min_conf', type=float, default=0.85, help='min confidence to accept discovered pattern')
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    doc, patterns = load_patterns(args.patterns)
    print(f"Loaded {len(patterns)} patterns from {args.patterns}")
    if not Path(args.input).exists():
        raise SystemExit(f'Input file not found: {args.input}')
    in_df = pd.read_excel(args.input)
    print(f"Input rows: {len(in_df)} - columns: {list(in_df.columns)}")
    if args.mode == 'predict':
        pred = apply_patterns_to_df(in_df, patterns)
        pred.to_excel(args.output, index=False)
        print(f"Saved predictions to {args.output}")
        return
    if args.mode == 'think':
        if not Path(args.verification).exists():
            raise SystemExit(f'Thinking mode requires a verification file. Not found: {args.verification}')
        truth_df = pd.read_excel(args.verification)
        best_overall = -1.0
        for it in range(1, args.max_iter+1):
            print('\n' + '='*80)
            print(f'Iteration {it}/{args.max_iter} — applying patterns and evaluating...')
            pred_df = apply_patterns_to_df(in_df, patterns)
            stats, overall = evaluate_predictions(pred_df, truth_df)
            print(f'Iteration {it} — overall weighted accuracy: {overall}')
            for col, v in stats.items():
                print(f"  {col:30s} => accuracy: {v['accuracy']}, n={v['n_truth']}")
            iter_out = str(Path(args.output).with_suffix('')) + f'_iter{it}.xlsx'
            pred_df.to_excel(iter_out, index=False)
            print(f'Wrote iteration predictions to {iter_out}')
            existing_tokens = set(p.get('pattern_token') for p in patterns if p.get('pattern_token'))
            new_pats = discover_new_patterns(truth_df, pred_df, min_occurrence=args.min_occ, min_confidence=args.min_conf, exclude_existing_tokens=existing_tokens)
            print(f'Found {len(new_pats)} candidate new patterns (min_occ={args.min_occ}, min_conf={args.min_conf})')
            added, updated = merge_new_patterns_into_doc(doc, new_pats)
            if added > 0 or updated > 0:
                save_patterns_doc(doc, args.patterns)
                doc, patterns = load_patterns(args.patterns)
                print(f'Patterns updated: +{added} added, {updated} updated — total patterns now {len(patterns)}')
            else:
                print('No pattern updates merged.')
            if overall is not None and overall > best_overall:
                print(f'Improved overall accuracy: {best_overall} -> {overall}')
                best_overall = overall
            else:
                print('No improvement this iteration (or unable to compute overall). Stopping early.')
                break
        pred_final = apply_patterns_to_df(in_df, patterns)
        pred_final.to_excel(args.output, index=False)
        print(f'Final predictions (after thinking) saved to {args.output}')
        return


if __name__ == '__main__':
    main()

