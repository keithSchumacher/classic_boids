# Mathematical Model of Boid Formalism

This document details the mathematical model implemented in the Classic Boids project, based on [*The computational beauty of flocking: Boids revisited* (Bajec, Zimic, and Mraz, 2007)](https://www.researchgate.net/publication/243041154_The_computational_beauty_of_flocking_Boids_revisited).

## Vector Space

Let $V = \mathbb{R}^d, (d = 2,3)$ be a Euclidean vector space.

## Internal State

A boid's Internal State $q \in Q$ is: $`Q = \{q \mid q = \langle p, v, r, fov, m, v_m, f_m, w \rangle \}`$

where:
- $p \in V$ is position
- $v \in V$ is velocity
- $r = \langle r_s, r_a, r_c \rangle$ are perception distances
- $fov = \langle fov_s, fov_a, fov_c \rangle$ are fields of view
- $m$ is mass
- $v_m$ is maximal speed
- $f_m$ is maximal force
- $w = \langle w_s, w_a, w_c \rangle$ are action weights

## Input and Output Alphabets

- A boid's output alphabet $Y = V \times V$ is its position and velocity
- The input alphabet of a boid is the current state of all boids in the system, $X = Y^n$ for an $n$-boid system

## Perception Function

A perception function for a characteristic $c$ is defined as $P: X \times Q \to P^c$. Here, $p \in P^c$ is the perceived neighborhood information.

$$
P_c(x,q) = \langle N_c, x \rangle,\quad N_c = \\{i \mid i \in N_c, i \neq j, \Delta(\lambda_i, q_i) \leq r_c, \phi(\lambda_i, q_i) \leq fov_c \\}
$$

### Distance

```math
\Delta(\lambda_i, q_i) = \| p_i - p_j \|
```

### Angular Offset

```math
\phi(\lambda_i, q_i) = \arccos \left[ \frac{v_j \cdot (p_i - p_j)}{\|v_j\| \cdot \|p_i - p_j\|} \right]
```

## Drive Functions

A drive function for a characteristic $c$ is defined as $D: P \times Q \to A^c$. Here, $a_c \in A^c$ is a force vector.

### Separation Drive Function

```math
D_s(p,q) =\left[\sum_{i \in N_s} \frac{p - p_i}{\|p_i - p\|^2}\right]^o
```

### Alignment Drive Function

$$D_a(p,q) =\left[ \left( \frac{1}{|N_a|} \sum_{i \in N_a} v_i \right) - v \right]^o $$

### Cohesion Drive Function

$$ D_c(p,q) = \left[ \left( \frac{1}{|N_c|} \sum_{i \in N_c} p_i \right) - p \right]^o $$

## Vector Operations

### Normalization

```math
v^o = \frac{v}{\|v\|}
```

### Truncation

Truncate vector $v$ by maximal size $v$:

```math
\lfloor v \rceil^v = \begin{cases} v & \text{if } \|v\| \le v \\ v^o \cdot v & \text{if } \|v\| > v \end{cases}
```

## Action Selection

Action Selection Protocol defines $S_{ws}: A \times Q \to Q$. Given $a \in A^3$ and $q \in Q$, the Action Selection Function produces:

$$ S_{ws}(a, q) = \langle p', v', r, fov, m, v_m, f_m, w \rangle $$

$$ v' = \left\lfloor v + \frac{\lfloor w_{s} a_{s} + w_{a} a_{a} + w_{c} a_{c} \rceil^{f_{\max}}}{m} \right\rceil^{v_{\max}} $$

$$ p' = p + v' $$

## Implementation Details

The mathematical model is implemented in the following core components:

- Vector operations: `src/classic_boids/core/vector.py`
- Internal state: `src/classic_boids/core/internal_state.py`
- Perception: `src/classic_boids/core/perception.py`
- Drive functions: `src/classic_boids/core/drive.py`
- Action selection: `src/classic_boids/core/action_selection.py`

For more details on the implementation, refer to the source code and the original paper. 