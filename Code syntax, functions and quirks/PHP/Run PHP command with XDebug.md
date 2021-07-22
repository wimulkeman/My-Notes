Running a command with XDebug on CLI (in Docker)
================================================
Xdebug 2
--------
```bash
/usr/local/bin/php -d memory_limit=-1 -d xdebug.remote_autostart=on -d xdebug.idekey=PHPSTORM -d xdebug.remote_enable=1 [php-command]
```

Xdebug 3
--------
```bash
/usr/local/bin/php -d memory_limit=-1 -d xdebug.client_host=${XDEBUG_CLIENT_HOST} -d xdebug.mode=${XDEBUG_MODE} -d xdebug.start_with_request=yes -d xdebug.idekey=PHPSTORM [php-command]
```
