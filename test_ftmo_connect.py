"""Test FTMO MT5 connection."""
import os
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5

login = int(os.getenv("MT5_LOGIN"))
password = os.getenv("MT5_PASSWORD")
server = os.getenv("MT5_SERVER")
path = os.getenv("MT5_PATH")

# Try 1: with path
print(f"Trying with path: {path}")
r = mt5.initialize(path=path, login=login, password=password, server=server)
print(f"  Result: {r}")
if not r:
    print(f"  Error: {mt5.last_error()}")
    mt5.shutdown()

    # Try 2: without path
    print(f"\nTrying without path...")
    r = mt5.initialize(login=login, password=password, server=server)
    print(f"  Result: {r}")
    if not r:
        print(f"  Error: {mt5.last_error()}")
        print("\n>>> You need to add FTMO-Demo server to your MT5 terminal.")
        print(">>> Open MT5 > File > Open an Account > search 'FTMO' > select FTMO-Demo")
        print(">>> Or click 'Open' on the FTMO credentials page to install their MT5.")
    else:
        info = mt5.account_info()
        print(f"  Connected! Account: {info.login} | Balance: {info.balance} {info.currency}")
        mt5.shutdown()
else:
    info = mt5.account_info()
    print(f"  Connected! Account: {info.login} | Balance: {info.balance} {info.currency} | Server: {info.server}")
    mt5.shutdown()
