from web3 import Web3, HTTPProvider
from flask import Flask, request, jsonify, render_template
import solcx
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# connect to a local Ethereum node
w3 = Web3(HTTPProvider('HTTP://127.0.0.1:7545'))

# compile the smart contract
contract_source_code = '''
pragma solidity ^0.8.0;

contract Chatbot {
    struct Message {
        address sender;
        string content;
    }
    Message[] public messages;

    function addMessage(string memory _content) public {
        messages.push(Message(msg.sender, _content));
    }

    function getMessages() public view returns (Message[] memory) {
        return messages;
    }
}
'''

solcx.install_solc('0.8.0')  # install Solidity compiler version 0.8.0
compiled_sol = solcx.compile_source(contract_source_code)
contract_interface = compiled_sol['<stdin>:Chatbot']

# deploy the smart contract

# ...

# deploy the smart contract
contract = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
tx_hash = contract.constructor().transact({'from': w3.eth.accounts[0], 'gas': 1000000})
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# Obtain the contract address from the transaction receipt
contract_address = tx_receipt['contractAddress']
contract = w3.eth.contract(address=contract_address, abi=contract_interface['abi'])

# ...


# Load GPT-2 model and tokenizer
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl' for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# initialize the Flask app
app = Flask(__name__)

# define the chat and messages routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    content = request.get_json()['content']
    contract.functions.addMessage(content).transact({'from': w3.eth.accounts[0]})

    # Generate response using GPT-2
    input_ids = tokenizer.encode(content, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify(success=True, response=response)

@app.route('/messages', methods=['GET'])
def messages():
    messages = contract.functions.getMessages().call()
    return render_template('messages.html', messages=messages)

# run the app
if __name__ == '__main__':
    app.run(debug=True)
