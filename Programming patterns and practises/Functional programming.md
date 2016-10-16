Functional programming
======================

## Fixed variable content

Once you have defined a variable, its content should
not be modified. When the content needs to be modified,
the variable should be copied into a new variable.

~~~php
// Wrong
$definedVar = 'test';
$definedVar += 1;

// Good
$definedVar = 'test';
$newVar = $definedVar + 1;
~~~

## References
- Medium - [So you want to be a functional programmer (part 1)](https://medium.com/@cscalfani/so-you-want-to-be-a-functional-programmer-part-1-1f15e387e536#.3xr5qsnex)