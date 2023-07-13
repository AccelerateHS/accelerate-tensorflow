-- |
-- Module      : nofib-tensorflow-lite
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Main where

import Test.Tasty
import Test.Tasty.Hedgehog

import Data.Array.Accelerate.Test.NoFib.Prelude
import Data.Array.Accelerate.Test.NoFib.Imaginary
import Data.Array.Accelerate.Test.NoFib.Misc


main :: IO ()
main
  = defaultMain
  $ localOption (HedgehogTestLimit (Just 5))
  $ localOption (HedgehogShrinkLimit (Just 0))
  $ testGroup "nofib-tensorflow-lite"
      [ test_prelude
      , test_imaginary
      , test_misc
      ]

