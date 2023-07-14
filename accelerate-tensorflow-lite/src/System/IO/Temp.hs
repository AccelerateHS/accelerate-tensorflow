-- |
-- Module      : System.IO.Temp
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module System.IO.Temp where

import Control.Exception
import System.Directory
import System.Posix.Temp


withTemporaryDirectory :: String -> (FilePath -> IO a) -> IO a
withTemporaryDirectory prefix = bracket (mkdtemp prefix) removeDirectoryRecursive
