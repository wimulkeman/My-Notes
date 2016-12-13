Twig
====

# For loop with inline if with and

```twig
<div class="row">
    {% for product in products %}
        <div class="span4">
            {# ... #}
        </div>

        {% if loop.index is divisibleby(3) and not loop.last %}
            </div><div class="row">
        {% endif %}
    {% endfor %}
</div>
```

# Check if argument is a even number

```twig
{% if products|length is even %}
    <div class="row">
        <div class="span12">
            There is an even number of products! OMG!
        </div>
    </div>
{% endif %}
```

# A else check for when the for loop is not run

```twig
{% for product in products %}
    {# ... #}
{% else %}
    <div class="alert alert-error span12">
        It looks like we're out of really awesome-looking penguin clothes :/.
    </div>
{% endfor %}
```

# Inline if syntax

```twig
<div class="well" style="background-color: {{ backgroundColor is defined ? backgroundColor : 'lightBlue' }};">
    {# ... #}
</div>
```

# Resources
- Knp University - [The for “loop” and inline “if” Syntax](https://knpuniversity.com/screencast/twig/for-loop-inline-if)