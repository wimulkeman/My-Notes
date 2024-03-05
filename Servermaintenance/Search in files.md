Search in files
===============

To search in multiple files in a (sub)directory using `grep`, you can use the following command:

```shell
grep -Rnw ./FILE_20230* -e 'foobar'
```

Where the following params are used:

* -e = The search string
* -R = Search recursive in the path provided
* -n = Alias for --line-number. Show the line mumber for the match in the file
* -w = Search for word expression

To search in all gz files in the current directory, you can use the following command:

```shell
find . -name \*.gz -print0 | xargs -0 zgrep "STRING"
```
