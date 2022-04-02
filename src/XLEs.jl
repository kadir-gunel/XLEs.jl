module XLEs

    export normalizeEmbedding, unit, unitByVar, center, vecLength, p_norm, parallelCosine,
    sequentialCosine, parallelIdx
    export readEmbeddings, readValidation, readBinaryEmbeddings,
    writeBinaryEmbeddings, convertBin2Txt, writeEmbeds, EmbeddingData, readData
    export train, trainWithNN, validate, advancedMapping, word2idx,
    buildSeedDictionary, buildNNSeedDictionary,cutVocabulary, validateCSLS,
    replaceSingulars, printResults, validateModel, getIDX_NN, getIDX_CSLS,
    Postprocessing, SplitInfo, trainWithLowestCondition, findLowestConditions,
    buildSubSpace, mapOrthogonal, softSeedDictionary, buildSeedDictionary2, buildMahalanobisDictionary


    include("./Embeddings.jl")
    include("./FileIO.jl")
    include("./Utils.jl")

end
