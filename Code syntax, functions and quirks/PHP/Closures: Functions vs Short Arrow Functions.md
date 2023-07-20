# Closures: Functions vs Short Arrow Functions

For closures you can use either functions, or the short arrow functions.
These functions differ from eachother by the scope they operate in.

Normal functions have a own scope which is separate from the scope they are implemented in
(except for $this). If you require access to a variable from the implementing scope you
need to pass it into the function using `use`.

Arrow functions have full access to the scope they are implemented in. Therefore dropping the
`use` requirement.

The thought behind this is that normal functions have a larger body which could make it harder
to keep track which variable name is used in which scope when both would be mixed by default.
The arrow functions are normally one-lines which reduces the cognitive load for the developer.
-- Source: PHP Internals News Podcast - Episode 4: Short Arrow Functions

```php
<?php
$foo = 'foo';
$bar = 'bar';

$arrayFunction = [
    'frist-',
    'second-',
];

$arrayFn = $arrayFunction;

// When using the function callback, you need to provide the variables inserted into its scope.
// The $this scope is always available in the callback
array_walk($arrayFunction, function (&$value) use ($foo) {
    $value .= $foo;
});

print_r($arrayFunction);
/*
Array
(
    [0] => frist-foo
    [1] => second-foo
)
*/

// Short arrow functions have full access to the variables in the scope where they are implemented
array_walk($arrayFn, fn (&$value) => $value .= $bar);

print_r($arrayFn);
/*
Array
(
    [0] => frist-bar
    [1] => second-bar
)
*/
```
