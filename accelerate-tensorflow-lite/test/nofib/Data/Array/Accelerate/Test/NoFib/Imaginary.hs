-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Imaginary
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Imaginary
  where

import Data.Array.Accelerate.Test.NoFib.Imaginary.DotP
import Data.Array.Accelerate.Test.NoFib.Imaginary.SASUM
import Data.Array.Accelerate.Test.NoFib.Imaginary.SAXPY

import Test.Tasty

test_imaginary :: TestTree
test_imaginary =
  testGroup "imaginary"
    [ test_saxpy
    , test_sasum
    , test_dotp
    ]

