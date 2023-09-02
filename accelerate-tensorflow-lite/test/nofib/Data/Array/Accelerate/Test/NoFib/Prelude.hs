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

import Data.Array.Accelerate.Test.NoFib.Base
import Data.Array.Accelerate.Test.NoFib.Prelude.Backpermute
import Data.Array.Accelerate.Test.NoFib.Prelude.Fold
import Data.Array.Accelerate.Test.NoFib.Prelude.Generate
import Data.Array.Accelerate.Test.NoFib.Prelude.Map
import Data.Array.Accelerate.Test.NoFib.Prelude.ZipWith
import Data.Array.Accelerate.Test.NoFib.Prelude.Foreign

import Test.Tasty


test_prelude :: TestContext -> TestTree
test_prelude tc =
  testGroup "prelude"
    [ test_map tc
    , test_zipWith tc
    , test_fold tc
    , test_backpermute tc
    , test_generate tc
    , test_foreign tc
    ]

