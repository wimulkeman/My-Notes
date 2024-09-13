Loading Vim without plugins (usefull when working with large files)

```sh
vim -u NONE file-to-open.txt
```

Removing all rows in a file which don't match the pattern (removing the ! after `g` will remove all rows that match the pattern)

```
:g!/Matching Text/d
```
