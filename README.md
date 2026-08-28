# Markowitz_MPT_Python

An implementation of Harry Markowitz's Modern Portfolio Theory's mean-variance optimization, relying on the underlying linear algebra to compute covariance matrix, lagrange multipliers and the efficient frontier.

## Why Mean-Variance Optimization

Just about any YouTube video or Article about MPT will tell you within the first 5 seconds that it changed the field forever, that it was a rupture in investment portfolio management. What a lot of these ressources fail to highlight however is how satisfying the financial and mathematical intuitions and reasonings behind this theory is, and that is precisely what motivated to dive deeper into this affair.

What these ressources definitely succeed in highlighting is that the actual math behind it, much like anything else in the field of mathematics that makes it so interesting, is how daunting and confusing it can get (or perhaps it's because I haven't been in school for long enough...)

That said, this project has allowed me to further develop not just my mathematical understandings of the tools I manipulated, but also a fundamental application of it through code.

## Things I've learned/understood better thanks to this project

1. Covariance matrix and their structure - through building a function that allows me to compute the covariance matrix of our list of assets from scratch, I've come to develop a sharper understanding of covariance matrix and their applications.

2. Deriving the closed-form solutions of Markowitz' problem via Lagrangian optimization - I was able to put to application the sometimes abstract rules of the game that is Linear Algebra, all while developing a newfound appreciation for this powerful tool (Lagrange multipliers), and not to mention explore the way a library like numpy handles its different function and why one solution not only is cleaner but also more optimized computationally.

3. The efficient frontier - other than the satisfaction from seeing a pyplot graph display on my screen with this curve that I've been chasing after for the past few days, building it, researching it and pondering on it allowed me to more deeply understand this curve, especially what it can say about a person. What makes MPT so interesting, and this applies to all of trading, is that no matter how concrete and grounded the mathematical theory behind it is, there is truly no 'correct' portfolio to choose from amongst the infinite amounts of portfolios above the suboptimal line. Markowitz's work revolutionzed the world of trading but in a way, it simply shrunk and moved a trader's playground, grounding it more on reason.

## Methodology

You can find the full derivation, from the equation for the standard deviation squared of a portfolio to the Lagrangian multipliers and its closed-form solution in : 

Choices of implementations : 
1. np.linalg.solve instead of np.linalg.inv - this allows us to avoid the numerical instability due to the increased number of floating point operations in explicitly inversing a matrix (especially as it gets larger and larger).

2. Covariance from scratch - this was more of a personal choice if anything as using pandas' pd.cov() just seemed to undermine this tool far too much.

3. Sweep range for the efficient frontier - on its lower-bound we have the closed-form minimum-variance return (derived analytically), and on its upper-bound we have a multiple of the highest individual asset return with the multiple being selected through empirical observations of the clarity of the final curve.

## Known limiations

1. Sample covariance can be susceptible to noise, thus requiring much data and observations to have something concrete. As well, and quite ironically, it  fails to capture industry changes that make 2 assets more or less intertwined (especially in the very fast-paced era of AI) 

2. Without limitations on these weights, negative results may occur thus suggesting holding an asset in a shorting position (example for r = 0.0015, return of MSFT < 0). For very high budgets to work with, I'd imagine that this might not always be possible because of supply/demand price fluctuation in large volumes.

### Be sure to check out requirements.txt