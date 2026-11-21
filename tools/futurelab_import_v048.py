#!/usr/bin/env python3
"""Import the original FutureLab Git history without overwriting remote main.
Requires Python 3, Git and authenticated push access. No credential is stored.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

EXPECTED_BUNDLE = '7c57553b2d92908be315a5659ee15658980e628f9693e82bc2e1877be7400284'
EXPECTED_HEAD = '64b7006f1322ef18b911cf2dba898c5e2e6b1b84'
EXPECTED_TREE = '7e60602b0ff44d4d1abb205fe30f32accc73959a'
DEST_BRANCH = 'refs/heads/import/futurelab-v0.48'
REMOTE = 'https://github.com/tortletortle/FutureLab.git'
TAGS = {
    'refs/tags/v0.36': '543c33e3bb96ccb08e5c258d604015e16af4e828',
    'refs/tags/v0.37': 'd231b1e157d654e8d9736debadbc3653b998d7a5',
    'refs/tags/v0.41': '96aed15178a9fa00fb49e0a49e5d0eb1b990c37d',
    'refs/tags/v0.42': 'e30c7fa6a34b3e0a2daaf5889de1109c92f27dfa',
    'refs/tags/v0.43': '7411cd290cd0d6cb068407c0b77af06778591cc6',
    'refs/tags/v0.48': '50a51fcf326150f6d6d8ea75715654222f5d0dee',
}

def git(*args, cwd=None):
    return subprocess.check_output(['git', *map(str,args)], cwd=cwd).decode('utf-8').strip()

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def refs(remote):
    return {ref:sha for sha,ref in (line.split('\t') for line in git('ls-remote','--refs',remote).splitlines() if line)}

def check(ok, msg):
    if not ok:
        raise SystemExit('STOP: '+msg)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bundle', nargs='?', default=str(Path(__file__).with_name('futurelab_v48.git.bundle')))
    ap.add_argument('--remote',default=REMOTE,help='Use default for production; override only for an intentional test mirror.')
    ap.add_argument('--check-only', action='store_true', help='Validate bundle and remote refs without writing.')
    args=ap.parse_args()
    bundle=Path(args.bundle).resolve()
    check(bundle.is_file(),'bundle not found')
    check(sha256(bundle)==EXPECTED_BUNDLE,'bundle SHA-256 mismatch')
    with tempfile.TemporaryDirectory(prefix='futurelab-import-') as d:
        git('init','--bare',d)
        git('bundle','verify',bundle,cwd=d)
        git('fetch','--no-tags',bundle,'refs/heads/main:'+DEST_BRANCH,*[f'{tag}:{tag}' for tag in TAGS],cwd=d)
        check(git('rev-parse',DEST_BRANCH,cwd=d)==EXPECTED_HEAD,'source commit mismatch')
        check(git('rev-parse',DEST_BRANCH+'^{tree}',cwd=d)==EXPECTED_TREE,'source tree mismatch')
        check(int(git('rev-list','--count',DEST_BRANCH,cwd=d))==15,'history count mismatch')
        check(len(git('ls-tree','-r','--name-only',DEST_BRANCH,cwd=d).splitlines())==158,'file count mismatch')
        for tag,sha in TAGS.items():
            check(git('rev-parse',tag,cwd=d)==sha,f'tag mismatch: {tag}')
        git('fsck','--full',cwd=d)
        desired={DEST_BRANCH:EXPECTED_HEAD,**TAGS}
        before=refs(args.remote)
        for ref,sha in desired.items():
            check(ref not in before or before[ref]==sha, f'remote ref differs, refusing to overwrite: {ref}')
        print('Verified local source tree: 158 files, 15 commits, 6 version tags.',flush=True)
        if args.check_only:
            print('CHECK ONLY: no remote changes.')
            return
        pending=[ref for ref,sha in desired.items() if before.get(ref)!=sha]
        if pending:
            git('push','--atomic',args.remote,*[f'{ref}:{ref}' for ref in pending],cwd=d)
        after=refs(args.remote)
        for ref,sha in desired.items():
            check(after.get(ref)==sha,f'remote verification failed: {ref}')
        check(after.get('refs/heads/main')==before.get('refs/heads/main'),'remote main changed concurrently; investigate separately')
        print('SUCCESS: original commit, tree and 6 tags verified on remote; main unchanged.')
        print('Source branch: import/futurelab-v0.48')

if __name__=='__main__':
    main()