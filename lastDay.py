import argparse
import random
import pandas as pd
import numpy as np
import time
from client.exchange_service.client import BaseExchangeServerClient
from protos.order_book_pb2 import Order
from protos.service_pb2 import PlaceOrderResponse

#Price Arrays. This is where we store all the ask and bid data for the different assets
priceK = []
priceM = []
priceN = []
priceQ = []
priceU = []
priceV = []
bidK = []
bidM = []
bidN = []
bidQ = []
bidU = []
bidV = []
askK = []
askM = []
askN = []
askQ = []
askU = []
askV = []

#Holding array. This is where we store data on our current holdings
holding01 = []

#Spread Array. This is where we store data on the spread between the different assets
spreads = []

#Predicted Spreads Array. This is where we store where we think the spread should be based on our Kalman Filtering
predSpreads = []

total = 0
firstTime = True
t=0

#Variable for keeping track of the time it takes us to run through a loop
global programStart
programStart = 0



class MarketMaker(BaseExchangeServerClient):
    """Market Making Bot"""

    def __init__(self, *args, **kwargs):
        BaseExchangeServerClient.__init__(self, *args, **kwargs)
        self._orderids = set([])
        
	#Function to make an order at a given price
    def _make_order(self, asset_code, quantity, base_price, spread, bid=True):
        return Order(asset_code = asset_code, quantity=quantity if bid else -1*quantity,
                     order_type = Order.ORDER_LMT,
                     price = base_price-spread/2 if bid else base_price+spread/2,
                     competitor_identifier = self._comp_id)
    
    '''Function to make a market order for a given asset. Base price is arbitrary.'''
    def _make_order_mkt(self, asset_code, quantity, base_price, spread, bid=True):
        return Order(asset_code = asset_code, quantity=quantity if bid else -1*quantity,
                     order_type = Order.ORDER_MKT,
                     price = None,
                     competitor_identifier = self._comp_id)

	#Creates dataframe to store price data
    def createPriceDF(self, exchange_update_response):
        priceK.append(exchange_update_response.market_updates[0].mid_market_price)
        priceM.append(exchange_update_response.market_updates[1].mid_market_price)
        priceN.append(exchange_update_response.market_updates[2].mid_market_price)
        priceQ.append(exchange_update_response.market_updates[3].mid_market_price)
        priceU.append(exchange_update_response.market_updates[4].mid_market_price)
        priceV.append(exchange_update_response.market_updates[5].mid_market_price)
        pricedf = pd.DataFrame(np.array([priceK, priceM,priceN, priceQ,priceU,priceV]))
        return pricedf
    #Creates dataframe to store bide data
    def createBidDF(self, exchange_update_response):
        bidK.append(exchange_update_response.market_updates[0].bids[0].price)
        bidM.append(exchange_update_response.market_updates[1].bids[0].price)
        bidN.append(exchange_update_response.market_updates[2].bids[0].price)
        bidQ.append(exchange_update_response.market_updates[3].bids[0].price)
        bidU.append(exchange_update_response.market_updates[4].bids[0].price)
        bidV.append(exchange_update_response.market_updates[5].bids[0].price)
        biddf = pd.DataFrame(np.array([bidK, bidM,bidN, bidQ,bidU,bidV]))
        return biddf
	#Creates dataframe to store ask data
    def createAskDF(self, exchange_update_response):
        askK.append(exchange_update_response.market_updates[0].asks[0].price)
        askM.append(exchange_update_response.market_updates[1].asks[0].price)
        askN.append(exchange_update_response.market_updates[2].asks[0].price)
        askQ.append(exchange_update_response.market_updates[3].asks[0].price)
        askU.append(exchange_update_response.market_updates[4].asks[0].price)
        askV.append(exchange_update_response.market_updates[5].asks[0].price)
        askdf = pd.DataFrame(np.array([askK, askM,askN, askQ,askU,askV]))
        return askdf
    
    #Initializes the Kalman filter
    def initializeKalmanVariables(self, a1, a2):
        a1 = np.array(a1)
        a2 = np.array(a2)
        a1Offset = np.ones(len(a1[0]))
        a2Offset = np.ones(len(a2[0]))
        a1 = np.vstack((a1, a1Offset))
        a2 = np.vstack((a2, a2Offset))
        delta=0.0001
        yhat=np.zeros(len(a1[0])) #measurement prediction
        e=np.zeros(len(a1[0])) #measurement prediction error 
        Q=np.zeros(len(a1[0])) #measurement prediction error variance
        P=np.zeros(2)
        beta=np.zeros((2, len(a1[0])))
        Vw=delta/(1-delta)*np.diag(np.ones((2, 1))) #Vw is covariance of Gaussian noise in state transition
        Ve=0.001  #Variance of noise in measurement equation
        n=0
        spent = 0
        spentShort = 0
        R=0
        qPower = 200
        yhat[t]=np.matmul(a2[:,t],beta[:,t].T) #measurement prediction.
        Q[t]=np.matmul(a2[:,t]*R,a2[:, t].T)+Ve; #measurement variance prediction.
        e[t]=a1[0][t]-yhat[t] #Measurement prediction error
        K=R*a2[:, t].T/Q[t] #Kalman Gain
        beta[:, t] = a1[t]/a2[t]
        beta[:, t]=beta[:, t]+K*e[t] #State Update
        P=R-K*a2[:, t]*R  #State Covariance Update
        return beta, R, P, Vw, yhat, Q, Ve, e
        
   	#Runs the Kalman Filter 
    def kalmanFilterUpdate(self, index1, index2, a1, a2, t, beta, R, P, Vw, yhat, Q, Ve, e, sitting):
        global total
        a1 = np.array(a1)
        a2 = np.array(a2)
        a1Offset = np.ones(len(a1[0]))
        a2Offset = np.ones(len(a2[0]))
        a1 = np.vstack((a1, a1Offset))
        a2 = np.vstack((a2, a2Offset))
        row1 = (np.append(beta[0], beta[0][t]))
        row2 = (np.append(beta[1], beta[1][t]))
        beta = np.array([row1, row2]) #State Prediction
        t = t+1
        R = P+Vw
        yhat = np.append(yhat, (np.matmul(a2[:,t],beta[:,t].T))) #measurement prediction.
        Q = np.append(Q, np.matmul(a2[:,t]*R,a2[:, t].T)+Ve); #measurement variance prediction.
        e = np.append(e, a1[0][t]-yhat[t]) #Measurement prediction error
        K=R*a2[:, t].T/Q[t] #Kalman Gain
        
        if t < 10:
            beta[:, t] = beta[:, t-1]+(K)*(e[t]**((t//2)+1))
        else:
            if e[t] > 1:
                beta[:, t]=beta[:, t-1]+(K)*(e[t]) 
            elif sitting > 50:
                beta[:, t]=beta[:, t-1]+(K)*(e[t])
            else:
                #e[t] exponent is hardcoded
                beta[:, t]=beta[:, t-1]+(K)*(e[t]**3) #State Update
                
        P=R-K*a2[:, t]*R  #State Covariance Update
        if index1 == 3 and index2 == 4:
            spread = a1[0][t]-a2[0][t]
            print("Spread: ", spread)
            predSpread = yhat[t]-a2[0][t]
            print("Pred Spread: ", predSpread)
            spreads.append(spread)
            predSpreads.append(predSpread)
        return beta, R, P, Vw, yhat, Q, Ve, e
        
    '''Function to test if there are both asks and bids for a given asset. This prevents
    	us from getting stuck trying to fill a market order'''  	       
    def bothSides(self,index, exchange_update_response):
        return len(exchange_update_response.market_updates[index].asks)>0 and  len(exchange_update_response.market_updates[index].bids)>0 
    
    #Perform a market buy          
    def buyMarket(self, index, quantity, exchange_update_response):
        asset = exchange_update_response.market_updates[index].asset.asset_code
        bid = (exchange_update_response.market_updates[index].bids[0].price)
        bid = round(bid, 2)+.01
        return self.place_order(self._make_order_mkt(asset, quantity, 0, 0, True))
            
	#Perform a market sell
    def sellMarket(self, index, quantity, exchange_update_response):
        asset = exchange_update_response.market_updates[index].asset.asset_code
        ask = (exchange_update_response.market_updates[index].asks[0].price)
        ask = round(ask, 2)+.01
        return self.place_order(self._make_order_mkt(asset, quantity, 0, 0, False))          
    
    #Limit Buy
    def buyIndex(self, index, quantity, exchange_update_response):
        asset = exchange_update_response.market_updates[index].asset.asset_code
        spread = (exchange_update_response.market_updates[index].asks[0].price)-(exchange_update_response.market_updates[index].bids[0].price)
        bid = (exchange_update_response.market_updates[index].asks[0].price)+.02
        bid = (exchange_update_response.market_updates[index].mid_market_price)
        bid = (exchange_update_response.market_updates[index].asks[0].price)
        bid = round(bid, 2)+5
        #print("Buy Options ", asset, exchange_update_response.market_updates[index].asks)
        #print("Sell Options ", asset, exchange_update_response.market_updates[index].bids)
        #print("Midpoint ", exchange_update_response.market_updates[index].mid_market_price)
        #print("BUYING ", asset, "at: ", bid)
        return self.place_order(self._make_order(asset, quantity, bid, 0, True))
    
    #Limit Sell                            
    def sellIndex(self, index, quantity, exchange_update_response):
        asset = exchange_update_response.market_updates[index].asset.asset_code
        spread = (exchange_update_response.market_updates[index].asks[0].price)-(exchange_update_response.market_updates[index].bids[0].price)
        ask = (exchange_update_response.market_updates[index].bids[0].price)-.05
        ask = (exchange_update_response.market_updates[index].mid_market_price)
        ask = (exchange_update_response.market_updates[index].bids[0].price)
        ask = round(ask, 2)-5
        return self.place_order(self._make_order(asset, quantity, ask, 0, False)

    '''Code that's run every time we receive an update from the exchange'''    
    def handle_exchange_update(self, exchange_update_response):
        start = time.time()
        global t, total, firstTime, kalmanStored, Exposure, fillActual, indices, fillTarget, asset2index, contractExposure, pnl, sitting
        global beta, R, P, Vw, yhat, Q, Ve, e

        buys = 0
        sells = 0
        
        #Updates all the dataframes that hold information from exchange updates
        pricedf = self.createPriceDF(exchange_update_response)
        biddf = self.createBidDF(exchange_update_response)
        askdf = self.createAskDF(exchange_update_response)
        
        #Initializes a bunch of stuff the first time we go through
        if firstTime:
            pnl = []
            print(firstTime)
            indices = {
                            0: "K",
                            1: "M", 
                            2: "N", 
                            3: "Q", 
                            4: "U", 
                            5: "V"
            }

            asset2index = {
                            "K": 0,
                            "M": 1, 
                            "N": 2, 
                            "Q": 3, 
                            "U": 4, 
                            "V": 5
            }

            fillActual = {"K": 0,
                          "M": 0,
                          "N": 0,
                          "Q": 0,
                          "U": 0,
                          "V": 0}

            fillTarget= {"K": 0,
                          "M": 0,
                          "N": 0,
                          "Q": 0,
                          "U": 0,
                          "V": 0}
            
            #Exposure of positions
            Exposure = {"01": 0,
                            "02": 0,
                            "03": 0,
                            "04": 0,
                            "05": 0,
                            "12": 0,
                            "13": 0,
                            "14": 0,
                            "15": 0,
                            "23": 0,
                            "24": 0,
                            "25": 0,
                            "34": 0,
                            "35": 0,
                            "45": 0}

            #Exposure of positions
            sitting = {"01": 0,
                            "02": 0,
                            "03": 0,
                            "04": 0,
                            "05": 0,
                            "12": 0,
                            "13": 0,
                            "14": 0,
                            "15": 0,
                            "23": 0,
                            "24": 0,
                            "25": 0,
                            "34": 0,
                            "35": 0,
                            "45": 0}            
            kalmanStored = {"01": self.initializeKalmanVariables(pricedf.loc[[0]], pricedf.loc[[1]]),
                            "02": self.initializeKalmanVariables(pricedf.loc[[0]], pricedf.loc[[2]]),
                            "03": self.initializeKalmanVariables(pricedf.loc[[0]], pricedf.loc[[3]]),
                            "04": self.initializeKalmanVariables(pricedf.loc[[0]], pricedf.loc[[4]]),
                            "05": self.initializeKalmanVariables(pricedf.loc[[0]], pricedf.loc[[5]]),
                            "12": self.initializeKalmanVariables(pricedf.loc[[1]], pricedf.loc[[2]]),
                            "13": self.initializeKalmanVariables(pricedf.loc[[1]], pricedf.loc[[3]]),
                            "14": self.initializeKalmanVariables(pricedf.loc[[1]], pricedf.loc[[4]]),
                            "15": self.initializeKalmanVariables(pricedf.loc[[1]], pricedf.loc[[5]]),
                            "23": self.initializeKalmanVariables(pricedf.loc[[2]], pricedf.loc[[3]]),
                            "24": self.initializeKalmanVariables(pricedf.loc[[2]], pricedf.loc[[4]]),
                            "25": self.initializeKalmanVariables(pricedf.loc[[2]], pricedf.loc[[5]]),
                            "34": self.initializeKalmanVariables(pricedf.loc[[3]], pricedf.loc[[4]]),
                            "35": self.initializeKalmanVariables(pricedf.loc[[3]], pricedf.loc[[5]]),
                            "45": self.initializeKalmanVariables(pricedf.loc[[4]], pricedf.loc[[5]])}
            
            beta, R, P, Vw, yhat, Q, Ve, e= self.initializeKalmanVariables(pricedf.loc[[0]], pricedf.loc[[2]])
            print("Done")
            firstTime = False
            return



        print("Outstanding Orders: ", len(self._orderids))
        print("Filled Orders: ", len(exchange_update_response.fills))
        print("Sitting: ", sitting)
        print("Holdings: ", fillActual)
        print("Positions: ", Exposure)
        for fill in exchange_update_response.fills:
            fillActual[str(fill.order.asset_code)] += fill.order.quantity
            
		'''Because exceeding long/short expusure limits resulted in disqualification, we paid
		extra care to keeping our number of filled buy orders close to our number of filled sell orders.
		Most of our effort went into making sure all of our orders were filled, but we included failsafes
		to protect against tail risks.'''
        
        netExposure = sum(fillActual.values())
        
        '''Whenever we had 15 more buy orders filled than sell order filled, we put in an
        order to sell one of every asset. Not the most elegant solution, but because we were
        trading such a high quantity at such a high rate and because we rarely exceeded a
        difference of 15, it didn't have a sizeable affect on our performance.''' 
        if netExposure >= 15:
            for index1 in range(0, len(pricedf)):
                ask_resp = self.sellIndex(index1, 1, exchange_update_response)                
                if type(ask_resp) != PlaceOrderResponse:
                    print(ask_resp)
                else:
                    self._orderids.add(ask_resp.order_id)

        if netExposure <= -15:
            for index1 in range(0, len(pricedf)):
                bid_resp = self.buyIndex(index1, 1, exchange_update_response)                
                if type(bid_resp) != PlaceOrderResponse:
                    print(bid_resp)
                else:
                    self._orderids.add(bid_resp.order_id)                        
      
        if len(self._orderids) > 0:
            for i in range(0, len(self._orderids)):
                self.cancel_order(self._orderids.pop())
        #Iterates through every combination of two assets        
        for index1 in range(len(pricedf)-2, len(pricedf)-1):    
            for index2 in range(index1+1, len(pricedf)):
                dictPosition = str(index1)+str(index2)
                #Runs Kalman Filter for given pair
                beta, R, P, Vw, yhat, Q, Ve, e = kalmanStored[dictPosition]
                kalmanStored[dictPosition] = self.kalmanFilterUpdate(index1, index2, pricedf.loc[[index1]], pricedf.loc[[index2]], t, beta, R, P, Vw, yhat, Q, Ve, e, sitting[dictPosition])
                beta, R, P, Vw, yhat, Q, Ve, e = kalmanStored[dictPosition]                
                if start - programStart < 10:
                    continue
                    
				#Isolates error terms from Kalman Filtering. This is what we will use to make buy/sell decision
                errorT = e[t]
                errorQ = Q[t]
                
                #If statement to protect against some extreme cases we encountered in testing.
                if errorT != None and (-1000< errorT < 1000) and self.bothSides(index1, exchange_update_response) and self.bothSides(index2, exchange_update_response) and abs(netExposure) <30:

                    if abs(Exposure[dictPosition]) >= 1000:
                        sitting[dictPosition]+=1
                    
                    else:
                        sitting[dictPosition]=0

                    '''Enter positions. Not that we almost always make a decision to either buy or sell.
                    Because we managed to achieve very high predictive power of a spreads movement, we were more concerned 
                    with being able to enter positions then we were about being picky about when to enter'''
                    if errorT < -0.00 and Exposure[dictPosition]<1000:
                        buyQuantity = 7
                        if Exposure[dictPosition] < -15:
                            buyQuantity = 14

                        fillTarget[indices[index1]] += buyQuantity
                        fillTarget[indices[index2]] -= buyQuantity
                        bid_resp = self.buyMarket(index1, buyQuantity, exchange_update_response)
                        ask_resp = self.sellMarket(index2, buyQuantity, exchange_update_response)
                        Exposure[dictPosition]+=buyQuantity
                        if type(bid_resp) != PlaceOrderResponse:
                            print(bid_resp)
                        else:
                            self._orderids.add(bid_resp.order_id)

                        if type(ask_resp) != PlaceOrderResponse:
                            print(ask_resp)
                        else:
                            self._orderids.add(ask_resp.order_id)
     
                    if errorT > 0.00 and Exposure[dictPosition]>-1000:
                        buyQuantity = 7
                        if Exposure[dictPosition] > 15:
                            buyQuantity = 14                  
                        fillTarget[indices[index1]] -= buyQuantity
                        fillTarget[indices[index2]] += buyQuantity
                        bid_resp = self.buyMarket(index2, buyQuantity, exchange_update_response)
                        ask_resp = self.sellMarket(index1, buyQuantity, exchange_update_response)
                        Exposure[dictPosition]-=buyQuantity
                        if type(bid_resp) != PlaceOrderResponse:
                            print(bid_resp)
                        else:
                            self._orderids.add(bid_resp.order_id)

                        if type(ask_resp) != PlaceOrderResponse:
                            print(ask_resp)
                        else:
                            self._orderids.add(ask_resp.order_id)

        print("PNL: ", exchange_update_response.competitor_metadata.pnl)
        holding01.append(Exposure["01"])
        if abs(exchange_update_response.competitor_metadata.pnl) < 25000:
            pnl.append(exchange_update_response.competitor_metadata.pnl)
        if (total > 825):
            np.savetxt("simulatedKalmanPrices.csv", pricedf, delimiter=",") 
            np.savetxt("simulatedKalmanBids.csv", biddf, delimiter=",")
            np.savetxt("simulatedKalmanAsks.csv", askdf, delimiter=",")
            np.savetxt("simulatedHolding01.csv", holding01, delimiter=",")
        if (total > 825):
            np.savetxt("simulatedPnl.csv", pnl)
            
        print("NET EXPOSURE:", netExposure)            
        t = t+1  
        total = time.time()-programStart
        print("TOTAL: ", total)
        loopTime = time.time()-start
        print("Loop Time: ", loopTime)

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the exchange client')
    parser.add_argument("--server_host", type=str, default="localhost")
    parser.add_argument("--server_port", type=str, default="0")
    parser.add_argument("--client_id", type=str)
    parser.add_argument("--client_private_key", type=str)
    parser.add_argument("--websocket_port", type=int, default=0)

    args = parser.parse_args()
    host, port, client_id, client_pk, websocket_port = (args.server_host, args.server_port,
                                        args.client_id, args.client_private_key,
                                        args.websocket_port)
    programStart = time.time()
    client = MarketMaker(host, port, client_id, client_pk, websocket_port)
    client.start_updates()
