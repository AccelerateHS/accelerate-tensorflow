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

import Test.Tasty


test_prelude :: TestTree
test_prelude =
  testGroup "prelude"
    [ test_map
    , test_zipWith
    , test_fold
    , test_backpermute
    , test_generate
    ]

