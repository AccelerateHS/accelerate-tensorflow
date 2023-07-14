-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Prelude
  where

import Data.Array.Accelerate.Test.NoFib.Prelude.Backpermute
import Data.Array.Accelerate.Test.NoFib.Prelude.Fold
import Data.Array.Accelerate.Test.NoFib.Prelude.Generate
import Data.Array.Accelerate.Test.NoFib.Prelude.Map
import Data.Array.Accelerate.Test.NoFib.Prelude.ZipWith
import Data.Array.Accelerate.Test.NoFib.Prelude.Foreign

import Data.Array.Accelerate.TensorFlow.Lite (ConverterPy)

import Test.Tasty


test_prelude :: ConverterPy -> TestTree
test_prelude converter =
  testGroup "prelude"
    [ test_map converter
    , test_zipWith converter
    , test_fold converter
    , test_backpermute converter
    , test_generate converter
    , test_foreign converter
    ]

