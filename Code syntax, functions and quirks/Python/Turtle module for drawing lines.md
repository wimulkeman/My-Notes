Turtle module for drawing lines
===============================

The turtle module can be used for drawing lines in Python.

```python
import turtle

drawer = turtle.Turtle()
```

After initiating the drawer, the lines can be created for a n of steps.

```python
drawer.forward(100)

# Making a turn
drawer.right(90)
drawer.formward(100)
drawer.left(45)
```

For the line to be shown, there needs to be a canvas

```python
window = turtle.Screen
window.bgcolor("red")

# Make the line move

window.exitonclick()
```

The drawer which is shown, can be modified.

```python
brad.shape('turtle')  # Default is arrow
brad.color('yellow')  # Default is color
brad.speed(2)  # Default is 1
```

## References
- [Python docs for the Turtle module](https://docs.python.org/2/library/turtle.html)