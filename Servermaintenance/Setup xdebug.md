1. Add the xdebug extension to PHP
2. Add the following configuration to a seperate xdebug ini file or in php.ini:
[XDebug]
zend_extension="xdebug.so"
xdebug.var_display_max_children = 256
xdebug.var_display_max_depth = 8
xdebug.max_nesting_level = 256
xdebug.cli_color = 2
xdebug.profiler_enable_trigger = 1
xdebug.profiler_output_dir = /tmp/xdebug
xdebug.remote_port = 10000
xdebug.remote_handler = dbgp
xdebug.remote_autostart = 1
xdebug.remote_enable = 1
xdebug.remote_connect_back = 0
xdebug.remote_host = localhost
3. Change the following config in php.ini: `implicit_flush = Off` > `implicit_flush = On`

Bron https://docs.joomla.org/Edit_PHP.INI_File_for_XDebug
