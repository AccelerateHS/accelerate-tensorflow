-- |
-- Module      : nofib-tensorflow-lite
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Main (main) where

import Test.Tasty
import Test.Tasty.Hedgehog

import Data.Array.Accelerate.Test.NoFib.Prelude
import Data.Array.Accelerate.Test.NoFib.Imaginary
import Data.Array.Accelerate.Test.NoFib.Misc
import Data.Array.Accelerate.Test.NoFib.Unit

import Data.Array.Accelerate.TensorFlow.Lite


converterSettings :: ConverterSettings
converterSettings = defaultConverterSettings { csVerbose = False }

main :: IO ()
main
  = withDeviceContext
  $ withConverterPy' converterSettings $ \converter ->
    defaultMain
  $ localOption (HedgehogTestLimit (Just 30))
  $ localOption (HedgehogShrinkLimit (Just 0))
  $ testGroup "nofib-tensorflow-lite"
      [ test_prelude converter
      , test_imaginary converter
      , test_misc converter
      , test_unit converter
      ]

