# KNN query point placement

## Motivating example
Say you have a dataset of restaurants from OpenStreetMap. You have their locations but you are not sure how good they are, so you want to gather the reviews from the Google Places API (it's a very important dinner - you are taking out a github user who has kindly done lots of work on KNN query point placement). 

The problem is you are a bit strapped for cash, and you want to save as much as possible to wine and dine said github user. Therefore you want to minimise the number of queries to the Google PLaces API, as each one costs money and before long you'll be saying 'no starter or dessert'.

In querying the Google Places API, you provide a latitude and longitude and it will provide the $k$ closest entries to that point (which we will call the 'query point').

The question is: 'how do you place the query points to minimise the number of queries you have to make in order to return reviews on all the restaurants?'

For now we will ignore the complexity that Google may have more, less, or just different restaurants, and assume OpenStreetMap and Google perfectly agree in your area. 

Let's say you had 9 restaurants arranged in space like so:

[IMAGE]

Now, let's say that $k=3$, so each query to the Google Places API will return the three closest restaurants to the query point. Looking at the map of restaurants, you, as an intelligent github user (who btw makes good decisions about who to take out for dinner) can easily choose where to place the points, for example, like so:

[IMAGE2]

Where we have linked the query points with the returned restaurants with a dashed line.

You are done, choose the best restaurant and treat that user to their well-deserved grub!

## Extending

However, life is not always so tasty. What about if you had 1000 restuarants, like so:

[IMAGE]

And let's say Google return not three but sixty closest restuarants. Now how do you place your query points? The purpose of this project is to provide a method to do so algorithmically.

## Mathematical formulation

Given a set of data points $S = \\{ s_1, s_2, ..., s_n \\}$ where $s_i \in \mathbb{R}^2$, an arbitrary point $x \in \mathbb{R}^2$, and a norm on $\mathbb{R}^2$ define a new ordering of $S$ such that 

$$\Vert s_{x, 1} - x \Vert \leq \Vert s_{x, 2} - x \Vert \leq ... \leq \Vert s_{x, n} - x \Vert$$

Now define the set $T_{x, k, S} = \\{s_{x, 1}, ..., s_{x, k} \\}$, i.e. the $k$ nearest neighbours of the point $x$ in the set $S$.

Define the concept of knn coverage for a set $T_{X, k, S} = \cup_{x \in X} T_{x, k, S}$

The objective is to minimise $t$ and find $X = \\{x_1, ..., x_t\\}$ such that $=T_{X, k, S} = S$.

We next try to prove the existance of an optimal solution, against which to meausure our algorithm. For that we will need the concept of a Voronoi cell.

## Voronoi cells

[Voronoi diagrams](https://en.wikipedia.org/wiki/Voronoi_diagram) are a way to parition the space into areas such that the KNN algorithm will return the same answer for every point in the cell. They are most usually seen for $k=1$ i.e. each point in a cell has the same nearest neighbour. They can however be extended to $k$ nearest neighbours. Now each point in the cell has the same $k$ nearest neighbours. In our above notation a Voronoi cell of order $k$ for a subset of points $P \subset S $ (where $|P|=k$) is defined as $V_{S, P} = \\{x : T_{x, k, S} = P \\}$


## Existence of an optimal solution

_Theorem_: Given a set $S$ and an integer $k$ where $|S| \text{ mod } k = 0$, $\exists$ a set $X = \\{x_1, ..., x_t\\}$ such that $\cup_{i} T_{x_i, k, S} = S$ and $|X| = \frac{|S|}{k}$.

_Proof_: By induction, start with $|S|=k$. This is trivial as any point will have all points in $S$ as its $k$ nearest neighbours. 

Now take $|S| = l$ as given, where $l=m \cdot k$. We are interested in the case of $|S| = l+k $. 

Consider each subset of $S$ of size $l$, of which there are $l+k\choose l$, call this number $h$. Label these sets $S_1,...,S_h$. For each of these there exists (by induction) a set $X_i$ such that $|X_i|=m$ and $\cup_{i,j} T_{X_i, k, S_i} = S_i$ for all $i$. 

Consider one of these $X_i = \\{x_{i,1},...,x_{i,m}\\}$ want to find a point $x_{i, m+1}$ such that $T_{x_{i,j}, k, S_i} = T_{x_{i, j}, k, S}$ for all $j=1,...,m+1$. For $j=1,...m$ this condition states that we do not change the $k$ nearest neighbours of any point $x_{i,1},...,x_{i,m}$ by adding the points in $S-S_{i}$. The case of $j=m+1$ states that the $k$ closest neighbours of our new point are exactly those in $S-S_{i}$. 

For each $j=1,..,m$ define $U_j = \max_{y \in T_{x_{i, j}, k, S_i}}(\Vert x_{i,j} - y \Vert)$, i.e. each query point's furtherst nearest neighbour. 

We need to find a point such that $\frac{\Vert x_{i, m+1} - x_{i,j} \Vert}{2} \geq U_j \quad \forall j \leq m$ and $\Vert p - x_{i, m+1} \Vert \leq \Vert q  - x_{i, m+1} \Vert \quad \forall p \in S-S_{i}$ and $\forall q \in S$.


## Finding an optimal solution




