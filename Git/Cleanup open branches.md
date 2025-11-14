# Handy commands to cleanup branches which are merged with master

## Show the last commiters for branches which are already merged with main

``` bash
for branch in `git branch -r --merged main | grep -v HEAD`; do echo -e `git show --format="%ci %cr %an" $branch | head -n 1` \\t$branch; done | sort -r
```

## Show which branches can be removed

```bash
git branch -r --merged master | egrep -v "(^\*|main|master|develop)" | sed 's/origin\///' | xargs -n 1 echo
```

## Delete all remote branches which are merged with main

```bash
git branch -r --merged master | egrep -v "(^\*|main|master|develop)" | sed 's/origin\///' | xargs -n 1 git push --delete origin
```

## Delete all locale branches which are merged with main

```bash
git branch --merged master | egrep -v "(^\*|main|master|develop)" | xargs git branch -d
