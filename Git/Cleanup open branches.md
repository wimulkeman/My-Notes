# Handy commands to cleanup branches which are merged with master

## Show the last commiters for branches which are already merged with master

for branch in `git branch -r --merged master | grep -v HEAD`; do echo -e `git show --format="%ci %cr %an" $branch | head -n 1` \\t$branch; done | sort -r


## Show which branches can be removed

git branch -r --merged master | egrep -v "(^\*|master|develop)" | sed 's/origin\///' | xargs -n 1 echo


## Delete all remote branches which are merged with master

git branch -r --merged master | egrep -v "(^\*|master|develop)" | sed 's/origin\///' | xargs -n 1 git push --delete origin


## Delete all locale branches which are merged with master

git branch --merged master | egrep -v "(^\*|master|develop)" | xargs git branch -d
