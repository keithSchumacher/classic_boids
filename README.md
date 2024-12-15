For development, navigate to root directory, activate virtual environment and run:
pip install -e .

Useful info for pytest:
https://github.com/wbolster/emacs-python-pytest?tab=readme-ov-file
coverage run -m pytest tests/
coverage report



## Equations

[Truncate](./src/classic_boids/core/vector.py#L78) vector $\mathbf{v}$ by maximal size $v$:

$$
\lfloor \mathbf{v} \rceil^v =
\begin{cases}
\mathbf{v} & \text{if } \|\mathbf{v}\| \le v \\
\hat{\mathbf{v}} \cdot v & \text{if } \|\mathbf{v}\| > v
\end{cases}
$$
