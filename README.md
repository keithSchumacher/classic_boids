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



[Action Selection Protocol](./src/classic_boids/core/protocols.py#L112) defines:

$$
S_{ws}: A \times Q \to Q
$$

Given a tuple of actions $ a \in A^3 $ and an internal state $ q \in Q $, the [Action Selection Function](./src/classic_boids/core/action_selection.py#L6) produces a new internal state:

$$
S_{ws}(a, q) = \langle p', v', \ldots \rangle
$$

$$
\mathbf{v}' = \left\lfloor \mathbf{v} + \frac{\lfloor w_{\text{s}} \cdot a_{\text{s}} + w_{\text{a}} \cdot a_{\text{a}} + w_{\text{c}} \cdot a_{\text{c}} \rceil^{f_{\max}}}{m} \right\rceil^{v_{\max}}
$$


$$
\mathbf{p}' = \mathbf{p} + \mathbf{v}'
$$
