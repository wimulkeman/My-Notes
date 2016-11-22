Speeding up SourceTree for Windows
==================================

Since version 1.5, SourceTree is slow in executing Git actions.
This is mostly noticeable when the user has a high number of
repositories added in SourceTree.

## Changing the default Git options

By adding the following to the default Git options, the
actions are performed faster.

```bash
git config --global core.preloadindex true
git config --global core.fscache true
git config --global gc.auto 256
```

These options are used for the following:

* **core.preload** Makes some of the actions multi threaded.
* **core.fscache** Enables another style of caching which
speeds up the caching mechanism which is used on Windows.
* **core.auto** Makes git to do fewer garbage collection.
That way Git is less intensive for the drive.

## Resources

* [Atlassian forums - SourceTree slow in Windows 7 x64...](https://answers.atlassian.com/questions/10413451/answers/12283864)