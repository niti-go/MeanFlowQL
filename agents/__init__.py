from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.meanflowql import MeanFlowQL_Agent
from agents.meanflowql_beta import MeanFlowQL_Agent_BETA
from agents.meanflow_bc import MeanFlowBC_Agent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    meanflowql = MeanFlowQL_Agent,
    meanflowql_beta = MeanFlowQL_Agent_BETA,
    meanflow_bc = MeanFlowBC_Agent,
)
