from probo.marketdata import MarketData
from probo.payoff import VanillaPayoff, call_payoff, put_payoff
from probo.engine import MonteCarloEngine, NaiveMonteCarloPricer, barrierPricer, AntitheticMonteCarloPricer
from probo.facade import OptionFacade

## Set up the market data
spot = 100
rate = 0.06
volatility = 0.20
dividend = 0.03
barrier = 99
thedata = MarketData(rate, spot, volatility, dividend, barrier)

## Set up the option
expiry = 1.0
strike = 100
thecall = VanillaPayoff(expiry, strike, call_payoff)
theput = VanillaPayoff(expiry, strike, put_payoff)

## Set up Naive Monte Carlo
nreps = 100
steps = 10
pricer = barrierPricer
mcengine = MonteCarloEngine(nreps, steps, pricer)

## Calculate the price
option1 = OptionFacade(thecall, mcengine, thedata)
price1 = option1.price()
print("The call price via Barrier is: {0:.3f}".format(price1))

option2 = OptionFacade(theput, mcengine, thedata)
price2 = option2.price()
print("The put price via Barrier is: {0:.3f}".format(price2))
