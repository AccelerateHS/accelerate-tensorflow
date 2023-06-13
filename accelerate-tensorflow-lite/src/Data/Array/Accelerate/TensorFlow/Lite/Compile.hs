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

  compileTfunWith,

) where

import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import qualified Data.Array.Accelerate.Debug.Internal               as Debug

import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
import Data.Array.Accelerate.TensorFlow.TypeDicts

import Data.Array.Accelerate.TensorFlow.Lite.Representation.Args

import qualified Proto.Tensorflow.Core.Framework.Graph              as TF
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields       as TF
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields     as TF
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF

import Control.DeepSeq
import Control.Exception
import Data.ByteString                                              ( ByteString )
import Data.ByteString.Builder                                      ( Builder )
import Data.Functor.Identity
import Data.List
import Data.ProtoLens
import Formatting
import Lens.Family2
import System.Directory
import System.Exit
import System.FilePath
import System.IO
import System.Process
import System.Process.Extra
import Text.Printf
import qualified Data.ByteString                                    as B
import qualified Data.ByteString.Builder                            as B
import qualified Data.Text                                          as T

import Paths_accelerate_tensorflow_lite


-- | Compile a tensorflow graph together with the given representative data
-- into a quantized tensorflow-lite model.
--
compileTfunWith :: Tfun f -> [Args f] -> IO ByteString
compileTfunWith f xs = do
  let f'  = graph_of_model f
      xs' = B.word8 (genericLength xs) <> foldMap buildArgs xs
  --
  tflite <- tflite_model f' xs'
  edge   <- edgetpu_compile tflite
  model  <- B.readFile edge
  return model


-- TODO: The intermediate files are created in a temporary directory, but
-- we should still clean them up afterwards...
--
edgetpu_compile :: FilePath -> IO FilePath
edgetpu_compile path = do
  let
      cp    = (proc "edgetpu_compiler" flags) { std_in = NoStream, std_out = CreatePipe, std_err = CreatePipe }
      flags = [ "--show_operations"
              , "--out_dir=" ++ dropFileName path
              , path
              ]
      edgetpu_file = dropExtension path ++ "_edgetpu" <.> "tflite"

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

  return edgetpu_file


tflite_model :: TF.GraphDef -> Builder -> IO FilePath
tflite_model graph xs = do
  convert         <- getDataFileName "converter.py"
  python_exe      <- getDataFileName "tf-python-venv/bin/python3"
  tmp_dir         <- getTemporaryDirectory
  (pb_file, pb_h) <- openBinaryTempFile tmp_dir "model.pb"      -- TODO: be exception safe
  (tf_file, tf_h) <- openBinaryTempFile tmp_dir "model.tflite"  -- TODO: be exception safe
  (rd_file, rd_h) <- openBinaryTempFile tmp_dir "data.bin"      -- TODO: be exception safe
  --
  B.hPut pb_h (encodeMessage graph)
  B.hPutBuilder rd_h xs
  hClose pb_h
  hClose tf_h
  hClose rd_h
  --
  let names   = map (view TF.name) (graph ^. TF.node)
      inputs  = filter (T.isPrefixOf "input") names
      outputs = filter (T.isPrefixOf "output") names
      --
      cp      = (proc python_exe flags) { std_in = NoStream, std_out = NoStream, std_err = CreatePipe }
      flags   = [ convert
                , "--graph_def_file=" ++ pb_file
                , "--output_file=" ++ tf_file
                , "--data_file=" ++ rd_file
                , if null inputs  then "" else "--input_arrays="  ++ T.unpack (T.intercalate "," inputs)
                , if null outputs then "" else "--output_arrays=" ++ T.unpack (T.intercalate "," outputs)
                ]

  -- Invoke 'tflite_convert' to convert the protobuf file to the tflite representation
  withCreateProcess cp $ \Nothing Nothing (Just errh) ph -> do

    -- fork off threads to start consuming stdout and stderr
    err <- hGetContents errh
    withForkWait (evaluate (rnf err)) $ \waitErr -> do
      waitErr
      hClose errh

    -- wait on the process
    ex <- waitForProcess ph
    case ex of
      ExitFailure r -> error $ printf "python3 %s (exit %d)\n%s" (unwords flags) r err
      ExitSuccess   -> return ()

  removeFile pb_file
  return tf_file


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

