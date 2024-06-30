// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CityLedger {
    // Struct to store transaction details
    struct Transaction {
        uint timestamp;
        address from;
        address to;
        uint amount;
        string description;
    }

    // Mapping to store transactions
    mapping(uint => Transaction) public transactions;
    uint public transactionCount;

    // Event to log transactions
    event TransactionAdded(uint indexed id, uint timestamp, address indexed from, address indexed to, uint amount, string description);

    // Function to add a transaction
    function addTransaction(address _to, uint _amount, string memory _description) public {
        transactionCount++;
        transactions[transactionCount] = Transaction(block.timestamp, msg.sender, _to, _amount, _description);
        emit TransactionAdded(transactionCount, block.timestamp, msg.sender, _to, _amount, _description);
    }
}
