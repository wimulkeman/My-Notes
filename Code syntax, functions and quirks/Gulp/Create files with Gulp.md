Create files with Gulp
======================

This practise is used to create the distributed files required
for the websites which are created at work.

## Base directory

The base directory to navigate to in order to use the following
commands is the place where the package.json and gulpfile.js
are stored.

In Cake projects, those are resident in app/webroot-sources/website.

## NPM

When the file package.json is available, then run the command

```bash
npm install
```

This will in stall the packages required.

## Gulp

Whem the file gulpfile.js is present, then run the command

```bash
gulp
```

This will create the compressed and optimized files in the
ditribution directory defined in the gulpfile.js.