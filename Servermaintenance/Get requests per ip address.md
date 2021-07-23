Get requests per ip address
===========================
Command to retrieve the number of requests made per IP address from the access log of a website

```bash
awk '{ print $1}' access_ssl.log | sort | uniq -c | sort -nr | head -n 10
```
