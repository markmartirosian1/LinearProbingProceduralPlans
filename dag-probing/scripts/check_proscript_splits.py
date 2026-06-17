"""
check_proscript_splits.py
-------------------------
Counts single-ordering vs multi-ordering plans in the ProScript dataset.
A plan is multi-ordering if it has at least one pair of steps that are
genuinely incomparable (no directed path between them in either direction).

Handles three common dataset layouts:
  1. ZIP of individual JSON files (e.g. the 622-plan training subset)
  2. Folder of individual JSON files
  3. Combined split files: train.json / dev.json / test.json
     where each is either a list of plan dicts or a dict keyed by scenario name

Usage:
  python check_proscript_splits.py --path /path/to/proScript_data.zip
  python check_proscript_splits.py --path /path/to/proScript_data/
  python check_proscript_splits.py --path /path/to/data/ --splits train dev test
"""

import argparse, json, zipfile, os
from collections import defaultdict
from pathlib import Path


# ── DAG utilities ──────────────────────────────────────────────────────────────

def parse_plan(plan_dict):
    """
    Extract real steps and edges from a ProScript plan dict.
    Filters out START and END sentinel nodes.
    Returns: (step_ids: list[int], adj: dict[int -> list[int]])
    
    Supports two formats:
    1. Original: {'steps': {...}, 'edges': [[a, b], ...]}
    2. JSONL: {'events': {...}, 'gold_edges_for_prediction': ["a->b", ...]}
    """
    # Handle both 'steps' and 'events' keys
    if 'events' in plan_dict:
        steps = {int(k): v.strip() for k, v in plan_dict['events'].items()}
    else:
        steps = {int(k): v.strip() for k, v in plan_dict['steps'].items()}
    
    start_id = next((k for k, v in steps.items() if v.upper() == 'START'), None)
    end_id   = next((k for k, v in steps.items() if v.upper() == 'END'),   None)
    real_ids = [k for k in steps if k not in (start_id, end_id)]

    adj = defaultdict(list)
    
    # Handle both edge formats
    if 'gold_edges_for_prediction' in plan_dict:
        # Format: ["0->1", "1->2", ...]
        for edge_str in plan_dict['gold_edges_for_prediction']:
            a, b = edge_str.split('->')
            a, b = int(a), int(b)
            if a in (start_id, end_id) or b in (start_id, end_id):
                continue
            if a in steps and b in steps:
                adj[a].append(b)
    else:
        # Format: [[0, 1], [1, 2], ...]
        for a, b in plan_dict['edges']:
            a, b = int(a), int(b)
            if a in (start_id, end_id) or b in (start_id, end_id):
                continue
            if a in steps and b in steps:
                adj[a].append(b)

    return real_ids, adj


def reachable(start, adj):
    """BFS reachability from start node."""
    visited, queue = set(), [start]
    while queue:
        node = queue.pop()
        for nb in adj.get(node, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def has_incomparable_pairs(plan_dict):
    """
    Returns True if the plan has at least one pair of steps (i, j) where
    neither i can reach j nor j can reach i — i.e., genuinely parallel steps.
    """
    real_ids, adj = parse_plan(plan_dict)
    if len(real_ids) < 2:
        return False
    reach = {n: reachable(n, adj) for n in real_ids}
    for i, ni in enumerate(real_ids):
        for nj in real_ids[i + 1:]:
            if nj not in reach[ni] and ni not in reach[nj]:
                return True
    return False


def count_incomparable_pairs(plan_dict):
    """Returns the count of incomparable step pairs."""
    real_ids, adj = parse_plan(plan_dict)
    reach = {n: reachable(n, adj) for n in real_ids}
    count = 0
    for i, ni in enumerate(real_ids):
        for nj in real_ids[i + 1:]:
            if nj not in reach[ni] and ni not in reach[nj]:
                count += 1
    return count


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_from_zip(zip_path):
    """Load individual JSON plan files from a ZIP archive."""
    plans = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('.json') and not name.endswith('/'):
                try:
                    d = json.loads(z.read(name))
                    goal = Path(name).stem.replace('_', ' ')
                    plans[goal] = d
                except Exception:
                    pass
    return plans


def load_from_folder(folder_path):
    """Load individual JSON plan files from a folder."""
    plans = {}
    for p in Path(folder_path).rglob('*.json'):
        try:
            d = json.loads(p.read_text(encoding='utf-8-sig'))
            goal = p.stem.replace('_', ' ')
            plans[goal] = d
        except Exception:
            pass
    return plans


def load_split_file(json_path):
    """
    Load a combined split file (train.json / dev.json / test.json).
    Handles two formats:
      - List:  [ {scenario, steps, edges, ...}, ... ]
      - Dict:  { scenario_name: {steps, edges, ...}, ... }
    """
    data = json.loads(Path(json_path).read_text(encoding='utf-8-sig'))
    if isinstance(data, list):
        return {d.get('scenario', d.get('goal', f'plan_{i}')): d
                for i, d in enumerate(data)}
    elif isinstance(data, dict):
        return data
    return {}


def load_jsonl_file(jsonl_path):
    """
    Load a .jsonl file where each line is a separate JSON object.
    Returns a dict mapping scenario names to plan dictionaries.
    """
    plans = {}
    with open(jsonl_path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                scenario = d.get('scenario', d.get('goal', f'plan_{i}'))
                plans[scenario] = d
            except Exception as e:
                print(f'  Warning: could not parse line {i+1}: {e}')
    return plans


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyse(plans, split_name='all'):
    """Print ordering statistics for a dict of {goal: plan_dict}."""
    single, multi, total = 0, 0, 0
    incompat_counts = []

    for goal, plan in plans.items():
        try:
            n = count_incomparable_pairs(plan)
            incompat_counts.append(n)
            if n == 0:
                single += 1
            else:
                multi += 1
            total += 1
        except Exception as e:
            print(f'  Warning: could not parse plan "{goal}": {e}')

    if total == 0:
        print(f'[{split_name}] No plans found.')
        return

    avg_ic = sum(incompat_counts) / total
    max_ic = max(incompat_counts)
    multi_ic = [c for c in incompat_counts if c > 0]
    avg_ic_multi = sum(multi_ic) / len(multi_ic) if multi_ic else 0

    print(f'\n{"=" * 55}')
    print(f'  Split: {split_name}  ({total} plans)')
    print(f'{"=" * 55}')
    print(f'  Single-ordering (total order):   {single:5d}  ({single/total*100:5.1f}%)')
    print(f'  Multi-ordering  (partial order): {multi:5d}  ({multi/total*100:5.1f}%)')
    print(f'  ─────────────────────────────────────────────')
    print(f'  Avg incomparable pairs (all):    {avg_ic:.2f}')
    print(f'  Avg incomparable pairs (multi):  {avg_ic_multi:.2f}')
    print(f'  Max incomparable pairs:          {max_ic}')
    print(f'{"=" * 55}')

    return {'split': split_name, 'total': total, 'single': single,
            'multi': multi, 'pct_multi': round(multi / total * 100, 1),
            'avg_incompat': round(avg_ic, 2), 'max_incompat': max_ic}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Count multi/single-ordering ProScript plans')
    parser.add_argument('--path', required=True,
                        help='Path to ZIP file, folder of JSONs, or folder containing split files')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'],
                        help='Split names to look for (default: train dev test)')
    args = parser.parse_args()

    path = Path(args.path)
    summaries = []

    if path.suffix == '.zip':
        # Single ZIP of individual plan JSON files
        print(f'Loading from ZIP: {path}')
        plans = load_from_zip(path)
        summaries.append(analyse(plans, split_name=path.stem))

    elif path.is_dir():
        # Check for .jsonl split files first
        jsonl_files = {s: path / f'{s}.jsonl' for s in args.splits
                       if (path / f'{s}.jsonl').exists()}
        # Also check val.jsonl as alias for dev
        if 'dev' in args.splits and not (path / 'dev.jsonl').exists():
            if (path / 'val.jsonl').exists():
                jsonl_files['dev'] = path / 'val.jsonl'
        
        # Check for .json split files
        json_files = {s: path / f'{s}.json' for s in args.splits
                      if (path / f'{s}.json').exists()}
        # Also check val.json as alias for dev
        if 'dev' in args.splits and not (path / 'dev.json').exists():
            if (path / 'val.json').exists():
                json_files['dev'] = path / 'val.json'

        if jsonl_files:
            print(f'Found .jsonl split files: {list(jsonl_files.keys())}')
            for split_name, split_path in jsonl_files.items():
                plans = load_jsonl_file(split_path)
                summaries.append(analyse(plans, split_name=split_name))
        elif json_files:
            print(f'Found .json split files: {list(json_files.keys())}')
            for split_name, split_path in json_files.items():
                plans = load_split_file(split_path)
                summaries.append(analyse(plans, split_name=split_name))
        else:
            # Fall back to loading all JSON files from folder
            print(f'Loading all JSON files from folder: {path}')
            plans = load_from_folder(path)
            summaries.append(analyse(plans, split_name='all'))
    else:
        print(f'Error: {path} is not a ZIP file or directory.')
        return

    # Overall summary if multiple splits
    if len(summaries) > 1:
        total_all = sum(s['total'] for s in summaries)
        multi_all = sum(s['multi'] for s in summaries)
        single_all = sum(s['single'] for s in summaries)
        print(f'\n{"=" * 55}')
        print(f'  OVERALL ({total_all} plans across all splits)')
        print(f'{"=" * 55}')
        print(f'  Single-ordering: {single_all:5d}  ({single_all/total_all*100:5.1f}%)')
        print(f'  Multi-ordering:  {multi_all:5d}  ({multi_all/total_all*100:5.1f}%)')
        print(f'{"=" * 55}')


if __name__ == '__main__':
    main()