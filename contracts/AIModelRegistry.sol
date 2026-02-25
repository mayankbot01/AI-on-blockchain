// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title AIModelRegistry
 * @dev Smart contract for registering and managing AI models on blockchain
 * @notice Implements immutable model versioning, cryptographic verification, and access control
 * @author Mayank Kumar - AI on Blockchain Project
 */
contract AIModelRegistry is AccessControl, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    // Role definitions
    bytes32 public constant MODEL_OWNER_ROLE = keccak256("MODEL_OWNER_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    
    // Model counter
    Counters.Counter private _modelIdCounter;
    
    // Structs
    struct ModelMetadata {
        uint256 modelId;
        string modelName;
        string modelType; // LLM, CNN, Transformer, etc.
        string framework; // PyTorch, TensorFlow, JAX
        address owner;
        string ipfsHash; // IPFS hash of model weights
        bytes32 modelHash; // Cryptographic hash of model
        uint256 version;
        uint256 createdAt;
        uint256 updatedAt;
        bool isActive;
        string description;
    }
    
    struct ModelMetrics {
        uint256 accuracy; // stored as percentage * 100 (e.g., 9550 = 95.50%)
        uint256 f1Score;
        uint256 precision;
        uint256 recall;
        uint256 trainingEpochs;
        string datasetHash; // IPFS hash of training dataset metadata
        uint256 trainingTime; // in seconds
    }
    
    struct ModelVersion {
        uint256 versionNumber;
        bytes32 modelHash;
        string ipfsHash;
        uint256 timestamp;
        string changeLog;
    }
    
    // Storage mappings
    mapping(uint256 => ModelMetadata) public models;
    mapping(uint256 => ModelMetrics) public modelMetrics;
    mapping(uint256 => ModelVersion[]) public modelVersionHistory;
    mapping(bytes32 => bool) public modelHashExists;
    mapping(address => uint256[]) public ownerModels;
    
    // Events
    event ModelRegistered(
        uint256 indexed modelId,
        address indexed owner,
        string modelName,
        bytes32 modelHash,
        uint256 timestamp
    );
    
    event ModelUpdated(
        uint256 indexed modelId,
        uint256 newVersion,
        bytes32 newModelHash,
        uint256 timestamp
    );
    
    event ModelDeactivated(
        uint256 indexed modelId,
        uint256 timestamp
    );
    
    event ModelMetricsUpdated(
        uint256 indexed modelId,
        uint256 accuracy,
        uint256 f1Score,
        uint256 timestamp
    );
    
    event ModelVerified(
        uint256 indexed modelId,
        address indexed validator,
        bool isValid,
        uint256 timestamp
    );
    
    // Constructor
    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(MODEL_OWNER_ROLE, msg.sender);
        _setupRole(VALIDATOR_ROLE, msg.sender);
    }
    
    /**
     * @dev Register a new AI model on the blockchain
     * @param _modelName Name of the model
     * @param _modelType Type of the model (e.g., "LLM", "CNN")
     * @param _framework Framework used (e.g., "PyTorch")
     * @param _ipfsHash IPFS hash of model weights
     * @param _modelHash Cryptographic hash of the model
     * @param _description Description of the model
     */
    function registerModel(
        string memory _modelName,
        string memory _modelType,
        string memory _framework,
        string memory _ipfsHash,
        bytes32 _modelHash,
        string memory _description
    ) external nonReentrant returns (uint256) {
        require(bytes(_modelName).length > 0, "Model name cannot be empty");
        require(!modelHashExists[_modelHash], "Model hash already registered");
        
        _modelIdCounter.increment();
        uint256 newModelId = _modelIdCounter.current();
        
        ModelMetadata storage newModel = models[newModelId];
        newModel.modelId = newModelId;
        newModel.modelName = _modelName;
        newModel.modelType = _modelType;
        newModel.framework = _framework;
        newModel.owner = msg.sender;
        newModel.ipfsHash = _ipfsHash;
        newModel.modelHash = _modelHash;
        newModel.version = 1;
        newModel.createdAt = block.timestamp;
        newModel.updatedAt = block.timestamp;
        newModel.isActive = true;
        newModel.description = _description;
        
        modelHashExists[_modelHash] = true;
        ownerModels[msg.sender].push(newModelId);
        
        // Add initial version to history
        modelVersionHistory[newModelId].push(ModelVersion({
            versionNumber: 1,
            modelHash: _modelHash,
            ipfsHash: _ipfsHash,
            timestamp: block.timestamp,
            changeLog: "Initial model registration"
        }));
        
        // Grant model owner role
        grantRole(MODEL_OWNER_ROLE, msg.sender);
        
        emit ModelRegistered(newModelId, msg.sender, _modelName, _modelHash, block.timestamp);
        
        return newModelId;
    }
    
    /**
     * @dev Update an existing model with a new version
     * @param _modelId ID of the model to update
     * @param _ipfsHash New IPFS hash
     * @param _newModelHash New cryptographic hash
     * @param _changeLog Description of changes
     */
    function updateModel(
        uint256 _modelId,
        string memory _ipfsHash,
        bytes32 _newModelHash,
        string memory _changeLog
    ) external nonReentrant {
        require(models[_modelId].owner == msg.sender, "Not the model owner");
        require(models[_modelId].isActive, "Model is not active");
        require(!modelHashExists[_newModelHash], "New model hash already exists");
        
        ModelMetadata storage model = models[_modelId];
        model.version += 1;
        model.ipfsHash = _ipfsHash;
        model.modelHash = _newModelHash;
        model.updatedAt = block.timestamp;
        
        modelHashExists[_newModelHash] = true;
        
        // Add version to history
        modelVersionHistory[_modelId].push(ModelVersion({
            versionNumber: model.version,
            modelHash: _newModelHash,
            ipfsHash: _ipfsHash,
            timestamp: block.timestamp,
            changeLog: _changeLog
        }));
        
        emit ModelUpdated(_modelId, model.version, _newModelHash, block.timestamp);
    }
    
    /**
     * @dev Update model performance metrics
     * @param _modelId ID of the model
     * @param _accuracy Accuracy percentage (e.g., 9550 for 95.50%)
     * @param _f1Score F1 score
     * @param _precision Precision metric
     * @param _recall Recall metric
     * @param _trainingEpochs Number of training epochs
     * @param _datasetHash IPFS hash of dataset metadata
     * @param _trainingTime Training time in seconds
     */
    function updateModelMetrics(
        uint256 _modelId,
        uint256 _accuracy,
        uint256 _f1Score,
        uint256 _precision,
        uint256 _recall,
        uint256 _trainingEpochs,
        string memory _datasetHash,
        uint256 _trainingTime
    ) external {
        require(models[_modelId].owner == msg.sender, "Not the model owner");
        
        ModelMetrics storage metrics = modelMetrics[_modelId];
        metrics.accuracy = _accuracy;
        metrics.f1Score = _f1Score;
        metrics.precision = _precision;
        metrics.recall = _recall;
        metrics.trainingEpochs = _trainingEpochs;
        metrics.datasetHash = _datasetHash;
        metrics.trainingTime = _trainingTime;
        
        emit ModelMetricsUpdated(_modelId, _accuracy, _f1Score, block.timestamp);
    }
    
    /**
     * @dev Verify model integrity by checking hash
     * @param _modelId ID of the model
     * @param _providedHash Hash to verify against
     * @return isValid Whether the hash matches
     */
    function verifyModelIntegrity(
        uint256 _modelId,
        bytes32 _providedHash
    ) external view returns (bool isValid) {
        require(models[_modelId].isActive, "Model is not active");
        return models[_modelId].modelHash == _providedHash;
    }
    
    /**
     * @dev Deactivate a model
     * @param _modelId ID of the model to deactivate
     */
    function deactivateModel(uint256 _modelId) external {
        require(models[_modelId].owner == msg.sender, "Not the model owner");
        require(models[_modelId].isActive, "Model already deactivated");
        
        models[_modelId].isActive = false;
        models[_modelId].updatedAt = block.timestamp;
        
        emit ModelDeactivated(_modelId, block.timestamp);
    }
    
    /**
     * @dev Get model details
     * @param _modelId ID of the model
     * @return Model metadata
     */
    function getModel(uint256 _modelId) external view returns (ModelMetadata memory) {
        require(_modelId <= _modelIdCounter.current(), "Model does not exist");
        return models[_modelId];
    }
    
    /**
     * @dev Get model metrics
     * @param _modelId ID of the model
     * @return Model metrics
     */
    function getModelMetrics(uint256 _modelId) external view returns (ModelMetrics memory) {
        require(_modelId <= _modelIdCounter.current(), "Model does not exist");
        return modelMetrics[_modelId];
    }
    
    /**
     * @dev Get version history of a model
     * @param _modelId ID of the model
     * @return Array of model versions
     */
    function getModelVersionHistory(uint256 _modelId) external view returns (ModelVersion[] memory) {
        return modelVersionHistory[_modelId];
    }
    
    /**
     * @dev Get all models owned by an address
     * @param _owner Address of the owner
     * @return Array of model IDs
     */
    function getOwnerModels(address _owner) external view returns (uint256[] memory) {
        return ownerModels[_owner];
    }
    
    /**
     * @dev Get total number of registered models
     * @return Total count
     */
    function getTotalModels() external view returns (uint256) {
        return _modelIdCounter.current();
    }
    
    /**
     * @dev Validator function to mark model as verified
     * @param _modelId ID of the model
     * @param _isValid Verification result
     */
    function validateModel(uint256 _modelId, bool _isValid) external onlyRole(VALIDATOR_ROLE) {
        require(models[_modelId].isActive, "Model is not active");
        emit ModelVerified(_modelId, msg.sender, _isValid, block.timestamp);
    }
}
