For development, navigate to root directory, activate virtual environment and run: pip install -e .  Useful info for pytest: https://github.com/wbolster/emacs-python-pytest?tab=readme-ov-file coverage run -m pytest tests/ coverage report

## Boid Formalism

This project implements the Boid formalism from [*The computational beauty of flocking: Boids revisited* (Bajec, Zimic, and Mraz, 2007)](https://www.researchgate.net/publication/243041154_The_computational_beauty_of_flocking_Boids_revisited). Below is a brief overview of the formalism.

Let $V = \mathbb{R}^d, (d = 2,3)$ be a Euclidean [vector](./src/classic_boids/core/vector.py#L9) space.

A boid's [Internal State](./src/classic_boids/core/protocols.py#L60) $q \in Q$ is: $`Q = \{q \mid q = \langle p, v, r, fov, m, v_m, f_m, w \rangle \}`$

where $p \in V$ is position, $v \in V$ is velocity, $r = \langle r_s, r_a, r_c \rangle$ are perception distances, $fov = \langle fov_s, fov_a, fov_c \rangle$ are fields of view, $m$ is mass, $v_m$ is maximal speed, $f_m$ is maximal force, and $w = \langle w_s, w_a, w_c \rangle$ are action weights.

A boid's output alphabet $Y = V \times V$ is its position and velocity. The [input alphabet](./src/classic_boids/core/input_alphabet.py) of a boid is the current state of all boids in the system, $X = Y^n$ for an $n$-boid system.

A [perception](./src/classic_boids/core/protocols.py#L77) function for a characteristic $c$ is defined as $P: X \times Q \to P^c$. Here, $p \in P^c$ is the perceived neighborhood information.

[Perception Function](./src/classic_boids/core/perception.py#L20):

$$
P_c(x,q) = \langle N_c, x \rangle,\quad N_c = \\{i \mid i \in N_c, i \neq j, \Delta(\lambda_i, q_i) \leq r_c, \phi(\lambda_i, q_i) \leq fov_c \\}
$$

[Distance](./src/classic_boids/core/vector.py#L42):

```math
\Delta(\lambda_i, q_i) = \| p_i - p_j \|
```

[Angular offset](./src/classic_boids/core/vector.py#L49):

```math
\phi(\lambda_i, q_i) = \arccos \left[ \frac{v_j \cdot (p_i - p_j)}{\|v_j\| \cdot \|p_i - p_j\|} \right]
```

A [drive](./src/classic_boids/core/protocols.py#L86) function for a characteristic $c$ is defined as $D: P \times Q \to A^c$. Here, $a_c \in A^c$ is a force vector.

[Separation Drive Function](./src/classic_boids/core/drive.py#L6):

```math
D_s(p,q) =\left[\sum_{i \in N_s} \frac{p - p_i}{\|p_i - p\|^2}\right]^o
```

[Alignment Drive Function](./src/classic_boids/core/drive.py#L31):

$$D_a(p,q) =\left[ \left( \frac{1}{|N_a|} \sum_{i \in N_a} v_i \right) - v \right]^o $$

[Cohesion Drive Function](./src/classic_boids/core/drive.py#L56):

$$ D_c(p,q) = \left[ \left( \frac{1}{|N_c|} \sum_{i \in N_c} p_i \right) - p \right]^o $$

[Normalization](./src/classic_boids/core/vector.py#L66):

```math
v^o = \frac{v}{\|v\|}
```

[Truncate](./src/classic_boids/core/vector.py#L78) vector $v$ by maximal size $v$:

```math
\lfloor v \rceil^v = \begin{cases} v & \text{if } \|v\| \le v \\ v^o \cdot v & \text{if } \|v\| > v \end{cases}
```

[Action Selection Protocol](./src/classic_boids/core/protocols.py#L112) defines $S_{ws}: A \times Q \to Q$. Given $a \in A^3$ and $q \in Q$, the [Action Selection Function](./src/classic_boids/core/action_selection.py#L6) produces:

$$ S_{ws}(a, q) = \langle p', v', r, fov, m, v_m, f_m, w \rangle $$

$$ v' = \left\lfloor v + \frac{\lfloor w_{s} a_{s} + w_{a} a_{a} + w_{c} a_{c} \rceil^{f_{\max}}}{m} \right\rceil^{v_{\max}} $$

$$ p' = p + v' $$



## Installation & Setup


Follow these steps to install and run the **Classic Boids** simulation on a new machine.

## **1. Clone the Repository**
First, clone the project from GitHub:
```sh
git clone git@github.com:keithSchumacher/classic_boids.git
cd classic_boids
```

## **2. Create and Activate a Virtual Environment**
Create a Python virtual environment to isolate dependencies:
```sh
python -m venv .venv
source .venv/bin/activate
```

## **3. Install Dependencies**
Install the necessary Python packages:
```sh
pip install -r requirements.txt
```

## **4. Install the Project in Editable Mode**
To allow local imports during development, install the package in editable mode:
```sh
pip install -e .
```

## **5. Run Tests**
Run tests using `pytest`:
```sh
python -m pytest tests/
```
