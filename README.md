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

Now define the set $T_{x, k, S} = \\{s_{x, 1}, ..., s_{x, k} \\}$

The objective is to minimise $t$ and find $X = \\{x_1, ..., x_t\\}$ such that $\cup_{i} T_{x_i, k, S} = S$.

## Existance of an optimal solution

Lets first try and prove the existance of an optimal solution, against which to meausure our algorithm. 

_Theorem_: Given a set $S$ and an integer $k$ where $|S| \text{ mod } k = 0$, $\exists$ a set $X = \\{x_1, ..., x_t\\}$ such that $\cup_{i} T_{x_i, k, S} = S$ and $|X| = \frac{|S|}{k}$.

_Proof_: By induction, start with $|S|=k$. This is trivial as any point will have all points in $S$ as its $k$ nearest neighbours. 

Now take $|S| = l$ as given, where $l=m \cdot k$. We are interested in the case of $|S| = l+k $. 

Consider each subset of $S$ of size $k$, of which there are $l+k\choose k$, call this number $h$. Label these sets $S_1,...,S_h$. For each of these there exists (by induction) a set $X_i$ such that $|X_i|=m$ and $\cup_{i} T_{x_i, k, S_i} = S_i$ for all $i$. 

We want to find a point $x_{m+1}$ such that $T_{x_i, k, S_i} = T_{x_i, k, S}$ for all $i=1,...,m+1$. For $i=1,...m$ this condition states that we do not change the $k$ nearest neighbours of any point $x_1,...,x_m$ by adding the points in $S_{m+1}$. The case of $i=m+1$ states that the $k$ closest neighbours of our new point are exactly those missing from $\cup_{i \leq m} S_i$. 


## Voronoi cells

[Voronoi diagrams](https://en.wikipedia.org/wiki/Voronoi_diagram) are a way to parition the space into areas such that the KNN algorithm will return the same answer for every point in the cell. They are most usually seen for $k=1$ i.e. each point in a cell has the same nearest neighbour. They can however be extended to $k$ nearest neighbours. Now each point in the cell has the same $k$ nearest neighbours. An implementation of this can be found in `knn_query_point_placement/knn_query_point_placement/query_point_algorithms/nth_degree_voronoi.py`.

The question then becomes what can we do with this partition? We have transformed the problem from an infinite one (where a query point could be placed anywhere in 2d space, to a finite one, where for every point in each cell we will get an identical answer if we run KNN.

An obvious first approach is to simply check all the combinations. The problem with this is it could be very time consuming. Say we ended up with $V$ voronoi cells of degree $k$. If we can find a placement of $m=\lceil n/k \rceil$ (which would be the theoretical minimum number of query points if each query returned a distinct set of solutions) then this would involve checking $V \choose m$ combinations, which will be prohibitively large in most settings. 



