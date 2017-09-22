SSH
===

To retrieve the fingerprint of a local key

```bash
ssh-keygen -lf /path/to/ssh/key
```

To retrieve the fingerprint in the old method (MD5:aa:02:aa:52..), using the new SSH keygen

```bash
ssh-keygen -E md5 -lf <fileName>
```
