Bamboo
======

When the following error appears when you try to 
load a list with available repositories:

```text
Failed to load data from Bitbucket. [403 Forbidden]
```

and the following appears within the Bamboo logs:

```text
Additional XSRF checks failed for request: https://[domain-name]...
```

check if your also on the same protocol within
Bamboo as the protocol provided for the request
which was made. In this case, you should be in on
the https environment.

## Resources
- Atlassian - [Cross Site Request Forgery (CSRF) protection changes in Atlassian REST](https://confluence.atlassian.com/kb/cross-site-request-forgery-csrf-protection-changes-in-atlassian-rest-779294918.html)