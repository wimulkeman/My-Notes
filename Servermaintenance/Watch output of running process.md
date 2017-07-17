Watch the output of a running process
====================================

This commando can be used to watch the output of a already
running process.

First get the process id using top or ps aux.

Then run the following command:

```bash
strace -p[process id] -s9999 -e write
```

-p = contains the process id of the process of which the output
is required.
-s = the number of characters to output. Default is 32.
-e write = outputs everything of the process. There are multiple
levels available.