### Options trading simulator

I made this after finishing my other market simulator which I've built:

https://github.com/tobiasocula/market-simulator

The idea is to use a Hawkes process, which is a self-exciting process used in the financial sector to model occuring events in markets:
https://en.wikipedia.org/wiki/Hawkes_process

The idea is to generate orders for option contracts using this function. The activity of an order occuring within a contract depends on some of its statistics, like whether the contract is ITM or OTM, the expiry date etc. The function I use to model this is:

