# Diamonds

I came across this dataset a while back while looking through I think the MBA website for either the University of Virginia or Harvard website. It was titled "Sarah Gets a Diamond" and immediately two things came to mind. The first was that it would be good for a comparison of regression models.

The second is harder for me to pin down. In part it was how very traditional it was. The other part was how odd picking out an engagement ring would be based strictly off the numbers. I understand that not everyone has the means or the high earning potential HBS grads have that they can bank on and so are price sensitive. But I can't imagine anyone just basing the purchase of an engagement ring off a metric like caret per dollar. There's a lot of hopes and emotions tied up in that. And if there's not, that probably very telling. Also, a lot of couples shop together.

Still, it's a good dataset though, in hindsight, I'm pretty sure this ins't the same dataset. It's certainly not the one marked on HBS's site as that one has a paltry 6000 diamonds whereas this has just under 220,000 diamonds.

<br>

## Notes
 1. This is far too long for a single notebook, so the project has been broken into three different notebooks:I
    - Data Wrangling
    - EDA
    - Models
    
<br>

##  Key observations so far:
 * 72% of diamonds are round cut
 * Oval, Pear, and Emerald are the next most popular, ranging between 3%-6% each
 * 95% of the stones are 2 carats or less
 * The remain 5% are fairly evenly distributed for the balance (over 19 carats)
 * 2% of the stones can be classified as `melee diamonds` - .2 carats or less
 * The vast majority of stones have 'very, very slight', 'very slight', or 'slight' inclusions
 * There are clear distinction for stones that are cut at (or very close to) integer carat weights
 * For one carat diamonds, round-cut stones are sold a premium when you condsider supply and demand.

<br>

### Outstanding questions:
 * Amongst fancy diamonds, what colors are more valued? What trends are there in clarity?

<br>

 ## Current Work
 * Create models - using pipelines.
