---
date: 2025-07-04
title: Independence Is Not All You Need
categories:
  - Statistics
  - Python
  - Markov Chains
description: When do probabilities of independent events not capture the full story?
---
A couple days ago, the other Alex challenged me with a deceptively simple brainteaser: 

Suppose you're rolling a fair, six-sided die. You can choose one of two halting conditions:

1. You roll two $4$'s in a row.
2. You roll a $4$ followed immediately by a $5$.

Which of these conditions, if either, would maximize the expected number of times you'd roll the die?

Because rolls are independent, once you’ve rolled a $4$, the next roll is equally likely to be a $4$ or a $5$. This means that the halting conditions are equivalent... right?

As you've probably already guessed, there's something very subtly wrong with this logic! To find out what exactly is wrong, let's talk through the problem a little bit more carefully.

For both conditions, the first roll has a $\frac{1}{6}$ probability of being a $4$. When we consider the $44$ condition, the second roll has a $\frac{1}{6}$ probability of being another $4$, but you also have a $\frac{5}{6}$ probability of resetting your progress toward halting. If you reset, you'll need to roll another $4$ followed by *another* $4$!

Now let's consider the $45$ condition. Again, your second roll has a $\frac{1}{6}$ probability of being a $5$, but this time, there's also a $\frac{1}{6}$ probability of it being a $4$, which would *maintain* your progress toward halting! Now there's only a $\frac{4}{6}$ probability of completely resetting your progress toward halting.

Given this, it makes sense that the $44$ condition yields a higher expected number of dice rolls than the $45$ condition, but as we've just seen, intuition isn't always correct -- although we *are* right this time -- so let's start formalizing!

This problem is an example of a **Markov chain**, which is a stochastic process $X$ with the property that the probability of the next event $P(X_{n+1})$ depends only on the probability of the current event $P(X_{n})$:
$$
P(X_{n+1} = x_{n+1} \mid X_n=x_n, \, ... , X_1=x_1) = P(X_{n+1} \mid X_n=x_n)
$$
This aptly-named Markov property is an incredibly powerful idea because it creates a symmetry that is often extremely helpful when computing quantities of interest! In our die-rolling problem, for example, the symmetry lies in the fact that we can always calculate the expected number of rolls starting from some state $i$, denoted $E_i$, with the same three steps:

1. Increment our number of steps.
2. Jump from $i$ to $j$ with some transition probability denoted $p_{ij}$
3. Perform these *same* steps on $j$!

{{< sidenote "This logic never changes shape" >}}This is called "conditioning on the first step". In essence, we find a symmetry between a starting state and a next state, then we apply this symmetry to the third, fourth, etc. state from there.{{< /sidenote >}}; only the probabilities of transition from one state to another differ. This can be formalized easily as shown:

$$
E_i = 1 + \sum_{j}{p_{ij}E_j}; \, E_{end} = 0
$$

We can visualize our transition probabilities in a matrix $\textrm{P}$, where the value in the $i$-th row and $j$-th column is the probability of jumping from $i$ to $j$. For our $44$ halting condition, this might be:

$$
\textrm{P}_{44} = \begin{pmatrix}
\frac{5}{6} & \frac{1}{6} & 0 \\[4pt]
\frac{5}{6} & 0 & \frac{1}{6} \\[4pt]
\end{pmatrix}
$$

Here, the columns are the non-$4$ state, the $4$ state, and the ending state. Notice that the non-$4$ state encapsulates the "we haven't rolled yet" state. Let's unwrap our summation now!

$$
E_1 = 1 + p_{11}E_1 + p_{12}E_2 + p_{13}E_3
\\[4pt]
E_2 = 1 + p_{21}E_1 + p_{22}E_2 + p_{23}E_3
\\[4pt]
E_3 = 0
$$

We essentially {{< sidenote "copy-paste" >}}Alternatively, we can compute $\textrm{A} = \textrm{P} - I$ with the appropriately sized identity, then compute $\textrm{rref}\left(\begin{array}{ccccc|c} \textrm{A}_{11} & \textrm{A}_{12} & \textrm{A}_{13} & \cdots & \textrm{A}_{1n} & -1 \\ \textrm{A}_{21} & \textrm{A}_{22} & \textrm{A}_{23} & \cdots & \textrm{A}_{2n} & -1 \\ \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ \textrm{A}_{m1} & \textrm{A}_{m2} & \textrm{A}_{m3} & \cdots & \textrm{A}_{mn} & -1 \\ 0 & 0 & \cdots & 0 & 1 & 0\end{array} \right)${{< /sidenote >}}  $\textrm{P}_{44}$ into this system of equations.
$$
\\
E_1 = 1 + \frac{5}{6}E_1 + \frac{1}{6}E_2 + 0
\\[4pt]
E_2 = 1 + \frac{5}{6}E_1 + 0 + 0
\\[4pt]
E_3 = 0
$$
Solving yields $E_1 = 42$, $E_2 = 36$, $E_3 = 0$! 

Let's do the same for our $45$ halting condition. Our $\textrm{P}$ matrix might be:

$$
\textrm{P}_{45} = \begin{pmatrix}
\frac{5}{6} & \frac{1}{6} & 0 \\[4pt]
\frac{4}{6} & \frac{1}{6} & \frac{1}{6} \\[4pt]
\end{pmatrix}
$$
Here, the columns are still the non-$4$ state, the $4$ state, and the ending state. However, notice that in *this* matrix, we have a $\frac{1}{6}$ probability of staying in our $4$ state! This makes a pretty big difference when we compute our expected roll counts: $E_1 = 36$, $E_2 = 30$, $E_3 = 0$.

As a final sanity check, let's run some code to verify our results empirically.

```python
import collections
import random

def rolls_until_halt_condition(end_seq: list):
    total, n = 0, 0
    halt = collections.deque(end_seq)

    for _ in range(100000):
        count = 0
        last_seen = collections.deque([], maxlen=len(end_seq))

        while True:
            count += 1
            last_seen.append(random.randint(1, 6))
            if last_seen == halt:
                break

        total += count
        n += 1

    return total / n

print(f"44: {rolls_until_halt_condition([4, 4])}") # 41.91218
print(f"45: {rolls_until_halt_condition([4, 5])}") # 36.01862
```

Happy Independence Day!