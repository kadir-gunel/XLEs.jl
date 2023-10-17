module XLEs

    export normalizeEmbedding, unit, unitByVar, center, vecLength,
    p_norm, MMD, gram_rbf, gram_rbf2, center_gram, corrAndCov,
    doubleCenter!
    export readEmbeddings, readValidation, readBinaryEmbeddings,
    writeBinaryEmbeddings, convertBin2Txt, writeEmbeds, EmbeddingData,
    readData, convertTxt2Bin
    export train, main, validate, word2idx, buildSeedDictionary,
    cutVocabulary, validateCSLS, validateModel, getIDX_NN,
    getIDX_CSLS, mapOrthogonal, sqrt_eigen


    include("./Embeddings.jl")
    include("./FileIO.jl")
    include("./Utils.jl")

end
