Create a archive with recently modified files
=============================================

## Git diff

To create a archive with the files of the last modified files
in a repository, we can use git diff.

The following command will give a list of the files changes in between these commits.

```bash
git diff --name-only <revision A> <revision B>
```

## Git archive

This list can be used within the git archive command to create a
archive file with the modified files.

```bash
git archive -o update.tar.gz --format=tar.gz HEAD $(git diff --name-only HEAD^)
```

## References

- http://tosbourn.com/using-git-to-create-an-archive-of-changed-files/
- https://git-scm.com/docs/git-diff
- https://git-scm.com/docs/git-archive