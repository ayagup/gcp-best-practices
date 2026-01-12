For most high-traffic models, you don't need to analyze every prediction to get a statistically significant view of the data distribution.

Think of it like a political poll: you don't need to ask every single person in a country how they will vote to get an accurate prediction. A well-chosen sample of a few thousand people is enough.

Similarly, a lower sampling rate (e.g., 5% or 10%) on a high-volume endpoint will often provide more than enough data to accurately detect feature skew or drift. Sampling 100% of the data provides little to no additional accuracy in detecting drift but comes at 10-20x the cost.