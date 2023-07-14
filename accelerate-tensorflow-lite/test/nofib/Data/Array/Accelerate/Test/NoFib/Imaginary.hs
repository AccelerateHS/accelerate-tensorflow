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

import Data.Array.Accelerate.TensorFlow.Lite                        ( ConverterPy )

import Test.Tasty

test_imaginary :: ConverterPy -> TestTree
test_imaginary converter =
  testGroup "imaginary"
    [ test_saxpy converter
    , test_sasum converter
    , test_dotp converter
    ]

