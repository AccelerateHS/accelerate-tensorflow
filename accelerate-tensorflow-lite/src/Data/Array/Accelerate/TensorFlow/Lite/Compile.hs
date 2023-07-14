{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Compile
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Compile (

  compileTfun,
  compileTfunIn,

) where

import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import qualified Data.Array.Accelerate.Debug.Internal               as Debug

import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
import Data.Array.Accelerate.TensorFlow.TypeDicts

import Data.Array.Accelerate.TensorFlow.Lite.ConverterPy
import Data.Array.Accelerate.TensorFlow.Lite.Representation.Args

import qualified Proto.Tensorflow.Core.Framework.Graph              as TF
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields       as TF
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields     as TF
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF

import Control.DeepSeq
import Control.Exception
import Data.ByteString                                              ( ByteString )
import Data.Functor.Identity
import qualified Data.Set                                           as Set
import Data.ProtoLens
import Formatting
import Lens.Family2
import System.Exit
import System.FilePath
import System.IO
import System.IO.Temp
import System.Process
import System.Process.Extra
import Text.Printf
import qualified Data.ByteString                                    as B
import qualified Data.Text                                          as T


-- | Compile a tensorflow graph together with the given representative data
-- into a quantized tensorflow-lite model. Returns the produced .tflite file as
-- a binary blob.
--
-- This version starts a new converter.py process just for this compilation
-- job.
compileTfun :: Tfun f -> ArgsNames f -> [Args f] -> IO ByteString
compileTfun f argsnames xs = withConverterPy $ \converter ->
  compileTfunIn converter f argsnames xs

-- | Compile a tensorflow graph together with the given representative data
-- into a quantized tensorflow-lite model.
compileTfunIn :: ConverterPy -> Tfun f -> ArgsNames f -> [Args f] -> IO ByteString
compileTfunIn converter f argsnames xs = do
  let graph = graph_of_model f
  let actualInputs =
        Set.fromList $ filter (T.isPrefixOf "input") $ map (view TF.name) (graph ^. TF.node)
  --
  tflite <- runConverterJob converter graph (serialiseReprData argsnames actualInputs xs)
  model  <- edgetpu_compile tflite
  return model


edgetpu_compile :: ByteString -> IO ByteString
edgetpu_compile tfliteBlob = withTemporaryDirectory "acctflite-compile" $ \tmpdir -> do
  B.writeFile (tmpdir </> "model.tflite") tfliteBlob

  let cp    = (proc "edgetpu_compiler" flags) { std_in = NoStream, std_out = CreatePipe, std_err = CreatePipe }
      flags = [ "--show_operations"
              , "--out_dir=" ++ tmpdir
              , tmpdir </> "model.tflite"
              ]
      edgetpu_file = tmpdir </> "model_edgetpu.tflite"

  -- Invoke 'edgetpu_compile' to convert the tflite file into something
  -- suitable for the edge tpu
  withCreateProcess cp $ \Nothing (Just outh) (Just errh) ph -> do

    -- fork off threads to start consuming stdout and stderr
    out <- hGetContents outh
    err <- hGetContents errh
    withForkWait (evaluate (rnf out)) $ \waitOut -> do
      withForkWait (evaluate (rnf err)) $ \waitErr -> do
        waitOut
        hClose outh

        waitErr
        hClose errh

    -- wait on the process
    ex <- waitForProcess ph
    case ex of
      ExitFailure r -> error $ printf "edgetpu_compiler %s (exit %d)\n%s" (unwords flags) r err
      ExitSuccess   -> return ()

    Debug.traceM Debug.dump_cc ("cc: edgetpu_compiler\n" % reindented 2 string) out

  B.readFile edgetpu_file


graph_of_model :: OpenTfun aenf t -> TF.GraphDef
graph_of_model (Tlam _ f)         = graph_of_model f
graph_of_model (Tbody arrR model) =
  let
      go :: TF.MonadBuild m => ArraysR a -> Tensors a -> m ()
      go TupRunit              ()                         = return ()
      go (TupRpair aR bR)      (a, b)                     = go aR a >> go bR b
      go (TupRsingle ArrayR{}) (Tensor (ArrayR _ aR) _ a) = array aR a

      array :: TF.MonadBuild m => TypeR t -> TensorArrayData t -> m ()
      array TupRunit         ()     = return ()
      array (TupRpair aR bR) (a, b) = array aR a >> array bR b
      array (TupRsingle aR)  a      = buildTypeDictsScalar aR render a

      render :: TF.MonadBuild m => TF.Tensor TF.Build a -> m ()
      render t = TF.render t >> return ()

      nodes = runIdentity $ TF.evalBuildT (go arrR model >> TF.flushNodeBuffer)
      graph = defMessage & TF.node .~ nodes :: TF.GraphDef
  in
  graph

