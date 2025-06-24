---
title: All Finite DAGs May be Topologically Sorted
date: 2025-06-23
categories:
  - Algorithms
  - DAGs
  - Proofs
description: All Directed Acyclic Graphs may be topologically sorted
---
# Proof Sketch
By contradiction, show that every finite DAG has at least one {{< sidenote "source vertex" >}}A vertex that has in-degree ($\text{deg}^{-}$) 0{{< /sidenote >}}, and then show by induction that a topological sort may be formed.
## Lemma 1: Every Finite DAG has at least one source vertex
Assume for the sake of contradiction that a DAG $G$ does not have any source vertices. That is, all vertices in $G$ have in-degree of at least 1. 

Select an arbitrary vertex $v_0$ , since $deg^{-}(v_0)\geq 1$ , there exists some vertex $v_1$ such that the edge $v_1\to v_0$ exists.  Likewise, for $v_1$, there exists some vertex $v_2$ such that $v_2\to v_1$ exists. By induction, this process may be repeated infinitely to create an infinite sequence of neighboring vertices. However, since $G$ is finite, $|V|$ is finite, so there must exist some vertex that is repeated in the sequence, indicating a directed cycle.

This is a contradiction to the definition of a DAG. Hence, all finite DAGs have at least one source vertex $v$ with $\text{deg}^{-}(v) = 0$.

## Proof: All finite DAGs May Be Topologically Sorted
We prove with induction on the number of edges $V$.
* **Base Case:** An empty DAG is trivial.
* **Induction Hypothesis:** A finite DAG $G$ with $n$ vertices has a valid topological sort.
* **Inductive Step:** Consider a DAG $G$ with $n+1$ vertices. We know by Lemma 1 that there exists some source vertex $v$ with $\text{deg}^{-}(v) = 0$. Create DAG $G'$ by removing $v$ and all of it's edges. We know that $G'$ must also be a DAG as $G$ was acyclic. By the Induction hypothesis, $G'$ has some topological sort $v_1, v_2, \ldots, v_n$ . Now add back vertex $v$. Since $v$ is a source edge, it has no pre-requisites. Concatenating $v$ onto the topological sort of $G'$ yields a valid topological sort for $G$ . That is:
\[
	v, v_1, v_2, \ldots, v_n
\]

$\boxed{ }$