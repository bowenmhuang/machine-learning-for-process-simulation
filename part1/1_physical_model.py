'''
This script contains the physical model of an ethylene oxide plug flow reactor

inlet -> reactor model -> outlet

The inlet stream is defined by temperature, pressure and molar flowrates
The reactor model is defined by reaction kinetic equations, plus the amount of catalyst and heat added
The outlet is solved for a given inlet by solving the model using solve_ivp numerical differential equation solver

Note that the object-oriented programming of Stream and UnitOp was just for personal learning, not used extensively here.
The benefit would be that it would be more intuitive and scalable for sequences of different unit operations in plant-wide models.
'''

import numpy as np
from scipy.integrate import solve_ivp

#lower and upper bounds for inlet T,P and molar flows [T,P,C2H4,O2,EO,CO2,H2O]
slope=np.array([180,12*1e5,10,10,10,10,10])
intcp=np.array([350,10*1e5, 0, 0, 0, 0, 0])

#amount of heat and catalyst added [Q/MJ,cat/kg], also known as the "fluxes"
slope2=np.array([2,15])
intcp2=np.array([-1,0])

# molecular properties of [C2H4,O2,EO,CO2,H2O]
C_pT=[0.088, -0.0019, 8.0473, 0.0373, 0.0092]
C_p0=[42.9, 27.1, 43.71, 37.6, 75.4]
Mr = [28.05, 32, 44.05, 44.01, 18.015]

#Stream class has attributes T,P and molar flows [T,P,C2H4,O2,EO,CO2,H2O]
class Stream():
    def __init__(self,T,P,*v_mol):
        self.T=T
        self.P=P
        self.v_mol=v_mol
    @property
    def list_prop(self):
        return [self.T,self.P,*self.v_mol]  

#UnitOp class takes in inlet (a Stream) and fluxes. Here, we only model a plug flow reactor.
class UnitOp():
    def __init__(self,inlet,fluxes):
        self.inlet_prop=np.array(inlet.list_prop)*slope+intcp
        self.fluxes=np.array(fluxes)*slope2+intcp2
        self.outlet_prop=np.array(inlet.list_prop)*slope+intcp #initialize outlet as the inlet
    #EO plug flow reactor model using reaction kinetic equations
    def model(self,dx,model_inlet_prop):
        T,P,nC2H4,nO2,nEO,nCO2,nH2O = model_inlet_prop[0:7]
        v_mol = [nC2H4,nO2,nEO,nCO2,nH2O]
        totmol = np.sum(np.array(v_mol))
        molfrac = np.array(v_mol/totmol)
        p = P*np.array(molfrac)
        #for safety, temperature should not go above 530 K at any point. If so, the simulation is stopped.
        if T<530 and T>350:
            # reactor modelling and reaction kinetics: heat capacity, rate constants, equilibrium constants, rates (mol/(kg cat s)), molar enthalpy of reaction
            Cp = np.sum((np.array(C_pT)*T+np.array(C_p0))*np.array(v_mol))/(totmol+1e-5)
            k1 = np.exp(-4.087-43585.7/(8.3145*T))
            k2 = np.exp(3.503-77763.2/(8.3145*T))
            K1 = np.exp(-16.644+18321/(8.3145*T))
            K2 = np.exp(-14.823+34660.6/(8.3145*T))
            r1 = self.fluxes[1]*k1*p[0]*p[1]/(1+K1*p[1]+K2*p[1]**(1/2)*p[2])
            r2 = self.fluxes[1]*k2*p[0]*p[1]**(1/2)/(1+K1*p[1]+K2*p[1]**(1/2)*p[2])
            deltaH1 = 106.7*1e3
            deltaH2 = 1323*1e3
            # the step change in stream properties [T,P,C2H4,O2,EO,CO2,H2O]. molar flows are affected by the reaction
            # C2H4 + 0.5*O2 -> EO 
            # C2H4 + 3*O2 -> 2*CO2 + 2*H2O
            prop_step_change = np.array([(deltaH1*r1+deltaH2*r2+self.fluxes[0]*1e6)/(Cp*totmol), 0, -r1-r2, -0.5*r1-3*r2, r1, 2*r2, 2*r2])
            return prop_step_change
        else:
            return np.array([0, 0, 0, 0, 0, 0, 0])
    #solve the reaction kinetic equations using solve_ivp to find the outlet
    def calc_outlet(self):
        dx = (0,1)
        sol = solve_ivp(self.model,dx,self.inlet_prop)
        #outlet
        self.outlet_prop = sol.y[:,-1]
        return self.outlet_prop
    #results are [T,C2H4,O2,EO,CO2,H2O]. Pressure is omitted because the reactor is assumed isobaric
    def results(self):
        self.calc_outlet()
        return [(self.outlet_prop[0]-intcp[0])/slope[0],*(self.outlet_prop[2:7]-intcp[2:7])/slope[2:7]]
