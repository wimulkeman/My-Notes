Get the size of files
=====================
The following command can be used to retrieve the size of the files and directories (accumulated) in the directory where the command is run.

```bash
alias duf='nice -n 19 du -sk * | nice -n 19 sort -n | while read size fname; do for unit in k M G T P E Z Y; do if [ $size -lt 1024 ]; then echo -e "${size}${unit}\t${fname}"; break; fi; size=$((size/1024)); done; done'
```
