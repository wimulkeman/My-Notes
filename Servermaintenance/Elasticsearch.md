Elasticsearch
=============

Get the configuration settings used for
the nodes:

```bash
curl http://localhost:9200/_nodes/settings?pretty
```

These settings include the loaded configuration
and the paths which are used for retrieving
the configuration.

Get the settings for a specified index.

```bash
# Get settings for the index tweets
curl http://localhost:9200/tweets/_settings?pretty
```
