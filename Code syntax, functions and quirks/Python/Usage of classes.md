Usage of classes
================

## Init function

Classes in Python contain a init function which is like the __construct function within PHP.

# Instances
 
## Instance variable

A variable which is assigned to the self key word.

```python
class Foo:
    def __init__(self, bar):
        # Instance variable
        self.bar
        
    # Instance method defined by the self argument
    def foo_bar(self):
        return self.bar
        
    # Non-instace method
    @staticmethod
    def bar_foo():
        return 'bar'
```