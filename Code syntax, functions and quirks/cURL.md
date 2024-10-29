cURL
====

Retrieving the headers of a request

```bash
curl -I http://localhost
```

Send another hostname with the request to a server specific IP address

```bash
curl --header "Host: my-host.test" http://127.0.0.1/path/to/request
```
