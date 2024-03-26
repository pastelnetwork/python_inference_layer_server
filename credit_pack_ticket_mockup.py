from typing import List
from datetime import datetime

class InferenceCreditPackMockup:
    def __init__(self, credit_pack_identifier: str, authorized_pastelids: List[str], psl_cost_per_credit: float, total_psl_cost_for_pack: float, initial_credit_balance: float, credit_usage_tracking_psl_address: str):
        self.credit_pack_identifier = credit_pack_identifier
        self.authorized_pastelids = authorized_pastelids
        self.psl_cost_per_credit = psl_cost_per_credit
        self.total_psl_cost_for_pack = total_psl_cost_for_pack
        self.current_credit_balance = initial_credit_balance
        self.initial_credit_balance = initial_credit_balance
        self.credit_usage_tracking_psl_address = credit_usage_tracking_psl_address
        self.version = 1
        self.purchase_height = 0
        self.timestamp = datetime.utcnow()

    def is_authorized(self, pastelid: str) -> bool:
        return pastelid in self.authorized_pastelids

    def has_sufficient_credits(self, requested_credits: float) -> bool:
        return self.current_credit_balance >= requested_credits

    def deduct_credits(self, credits_to_deduct: float):
        if self.has_sufficient_credits(credits_to_deduct):
            self.current_credit_balance -= credits_to_deduct
        else:
            raise ValueError("Insufficient credits in the pack.")

    def to_dict(self):
        return {
            "credit_pack_identifier": self.credit_pack_identifier,
            "authorized_pastelids": self.authorized_pastelids,
            "psl_cost_per_credit": self.psl_cost_per_credit,
            "total_psl_cost_for_pack": self.total_psl_cost_for_pack,
            "initial_credit_balance": self.initial_credit_balance,
            "current_credit_balance": self.current_credit_balance,
            "credit_usage_tracking_psl_address": self.credit_usage_tracking_psl_address,
            "version": self.version,
            "purchase_height": self.purchase_height,
            "timestamp": self.timestamp.isoformat()
        }