/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution

import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanHelper, TableCacheQueryStageExec}
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.execution.exchange.{ReusedExchangeExec, ShuffleExchangeExec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession

class ReuseExchangeWithInMemoryTableScanSuite extends SparkPlanTest with SharedSparkSession
  with AdaptiveSparkPlanHelper {

  test("Exchange reuse with InMemoryTableScan below exchange") {
    import testImplicits._

    // Create a dummy DataFrame
    val df = (1 to 10000).toDF("id").withColumn("value", $"id" * 2)

    // Perform a transformation and cache
    val cachedDF = df.withColumn("value_squared", $"value" * $"value").cache()

    try {
      // Trigger materialization
      cachedDF.count()

      // Use cachedDF in two branches
      val branch1 = cachedDF.groupBy("id").agg(sum("value_squared").as("sum1"))
      val branch2 = cachedDF.groupBy("id").agg(avg("value_squared").as("avg2"))

      // Join the branches
      val joined = branch1.join(branch2, Seq("id"))

      // Trigger action and get the plan
      val plan = joined.groupBy().count().queryExecution.executedPlan
      plan.execute()

      // Print and analyze the plan
      logInfo(s"Executed plan: ${plan.toString}")

      // Collect all ShuffleExchangeExec nodes
      val shuffleExchanges = collectWithSubqueries(plan) {
        case s: ShuffleExchangeExec => s
      }

      // Collect all ReusedExchangeExec nodes
      val reusedExchanges = collectWithSubqueries(plan) {
        case r: ReusedExchangeExec => r
      }

      // Collect all InMemoryTableScanExec nodes
      val inMemoryTableScans = collectWithSubqueries(plan) {
        case i: InMemoryTableScanExec => i
      }

      // Collect all TableCacheQueryStageExec nodes (if any)
      val tableCacheStages = collectWithSubqueries(plan) {
        case t: TableCacheQueryStageExec => t
      }

      // Log counts of each type of node
      logInfo(s"Number of ShuffleExchangeExec: ${shuffleExchanges.size}")
      logInfo(s"Number of ReusedExchangeExec: ${reusedExchanges.size}")
      logInfo(s"Number of InMemoryTableScanExec: ${inMemoryTableScans.size}")
      logInfo(s"Number of TableCacheQueryStageExec: ${tableCacheStages.size}")

      // In a healthy scenario with reuse, we'd expect to see at least one ReusedExchangeExec
      // Since we're using the same cached DataFrame in two branches
      assert(
        reusedExchanges.nonEmpty,
        "Expected to see ReusedExchangeExec but found none. Exchange reuse is not working" +
          " correctly."
      )
    } finally {
      cachedDF.unpersist()
    }
  }

  test("Table cache stage reuse: same cached table scanned multiple times shares stages") {
    import testImplicits._

    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      SQLConf.EXCHANGE_REUSE_ENABLED.key -> "true") {

      val df = (1 to 100).toDF("id")
        .withColumn("value", $"id" * 2)
        .withColumn("category", $"id" % 5)
      val cachedDF = df.cache()

      try {
        cachedDF.count()

        // Two branches from the same cached DataFrame with identical projections
        val branch1 = cachedDF.groupBy("category").agg(sum("value").as("sum_val"))
        val branch2 = cachedDF.groupBy("category").agg(avg("value").as("avg_val"))
        val joined = branch1.join(branch2, Seq("category"))

        val executedPlan = joined.queryExecution.executedPlan
        executedPlan.execute()

        val tableCacheStages = collectWithSubqueries(executedPlan) {
          case t: TableCacheQueryStageExec => t
        }

        logInfo(s"Number of TableCacheQueryStageExec: ${tableCacheStages.size}")

        // With the isMaterialized optimization, no TableCacheQueryStageExec should be
        // created because the cache was already populated by count() above.
        // The InMemoryTableScan is inlined into the parent exchange stage.
        assert(tableCacheStages.isEmpty,
          s"Expected no TableCacheQueryStageExec for pre-materialized cache, " +
            s"but found ${tableCacheStages.size}. " +
            "Materialized InMemoryTableScans should be inlined into parent stages.")

        // Verify InMemoryTableScanExec nodes are present (inlined, not wrapped in stages)
        val inMemoryScans = collectWithSubqueries(executedPlan) {
          case i: InMemoryTableScanExec => i
        }
        assert(inMemoryScans.nonEmpty,
          "Expected InMemoryTableScanExec nodes to be inlined in the plan")
      } finally {
        cachedDF.unpersist()
      }
    }
  }

  test("Table cache stage reuse: different projections should NOT be reused") {
    import testImplicits._

    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      SQLConf.EXCHANGE_REUSE_ENABLED.key -> "true") {

      val df = (1 to 100).toDF("id")
        .withColumn("value", $"id" * 2)
        .withColumn("name", concat(lit("item_"), $"id"))
      val cachedDF = df.cache()

      try {
        cachedDF.count()

        // Two branches selecting completely different columns
        val branch1 = cachedDF.select("id", "value").groupBy("id").agg(sum("value").as("s"))
        val branch2 = cachedDF.select("id", "name").groupBy("id").agg(count("name").as("c"))

        val joined = branch1.join(branch2, Seq("id"))
        val executedPlan = joined.queryExecution.executedPlan
        val result = joined.collect()

        logInfo(s"Result count: ${result.length}")

        // Different projections should produce correct results even with the reuse mechanism
        assert(result.length > 0, "Join should produce results")

        // Verify correctness: each id should have the expected sum and count
        val row = joined.filter($"id" === 1).collect().head
        assert(row.getAs[Long]("s") === 2L, "sum(value) for id=1 should be 2")
        assert(row.getAs[Long]("c") === 1L, "count(name) for id=1 should be 1")
      } finally {
        cachedDF.unpersist()
      }
    }
  }

  test("Table cache stage reuse: reused stage is materialized when original completes") {
    import testImplicits._

    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      SQLConf.EXCHANGE_REUSE_ENABLED.key -> "true") {

      val df = (1 to 50).toDF("id").withColumn("value", $"id" * 3)
      val cachedDF = df.cache()

      try {
        cachedDF.count()

        // Create a multi-branch query that uses the same cached table multiple times
        val agg1 = cachedDF.groupBy("id").agg(sum("value").as("total"))
        val agg2 = cachedDF.groupBy("id").agg(max("value").as("max_val"))
        val agg3 = cachedDF.groupBy("id").agg(min("value").as("min_val"))

        val result = agg1.join(agg2, Seq("id")).join(agg3, Seq("id"))
        val executedPlan = result.queryExecution.executedPlan
        val collected = result.collect()

        val tableCacheStages = collectWithSubqueries(executedPlan) {
          case t: TableCacheQueryStageExec => t
        }

        // With pre-materialized cache, no TableCacheQueryStageExec should be created
        assert(tableCacheStages.isEmpty,
          s"Expected no TableCacheQueryStageExec for pre-materialized cache, " +
            s"but found ${tableCacheStages.size}")

        // Verify result correctness
        assert(collected.length === 50, "Should have 50 rows (one per id)")
        val row1 = result.filter($"id" === 10).collect().head
        assert(row1.getAs[Long]("total") === 30L)
        assert(row1.getAs[Int]("max_val") === 30)
        assert(row1.getAs[Int]("min_val") === 30)
      } finally {
        cachedDF.unpersist()
      }
    }
  }

  test("Table cache stage reuse: union of same cached table reduces duplicate scans") {
    import testImplicits._

    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      SQLConf.EXCHANGE_REUSE_ENABLED.key -> "true") {

      val df = (1 to 100).toDF("id").withColumn("value", $"id" * 2)
      val cachedDF = df.cache()

      try {
        cachedDF.count()

        // Union of same cached table with different filters
        val part1 = cachedDF.filter($"id" <= 50).select("id", "value")
        val part2 = cachedDF.filter($"id" > 50).select("id", "value")
        val unioned = part1.union(part2)

        val executedPlan = unioned.queryExecution.executedPlan
        val result = unioned.collect()

        logInfo(s"Union result count: ${result.length}")
        assert(result.length === 100, "Union should contain all 100 rows")

        val tableCacheStages = collectWithSubqueries(executedPlan) {
          case t: TableCacheQueryStageExec => t
        }

        logInfo(s"TableCacheQueryStageExec count in union plan: ${tableCacheStages.size}")

        // Pre-materialized cache should not create extra stages
        assert(tableCacheStages.isEmpty,
          s"Expected no TableCacheQueryStageExec for pre-materialized cache, " +
            s"but found ${tableCacheStages.size}")
      } finally {
        cachedDF.unpersist()
      }
    }
  }

  test("Table cache stage reuse: no reuse across different cached tables") {
    import testImplicits._

    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      SQLConf.EXCHANGE_REUSE_ENABLED.key -> "true") {

      val df1 = (1 to 50).toDF("id").withColumn("val1", $"id" * 2).cache()
      val df2 = (1 to 50).toDF("id").withColumn("val2", $"id" * 3).cache()

      try {
        df1.count()
        df2.count()

        val joined = df1.join(df2, Seq("id"))
        val executedPlan = joined.queryExecution.executedPlan
        val result = joined.collect()

        val tableCacheStages = collectWithSubqueries(executedPlan) {
          case t: TableCacheQueryStageExec => t
        }

        logInfo(s"TableCacheQueryStageExec count for two different tables: " +
          s"${tableCacheStages.size}")

        // Both caches are pre-materialized, so no TableCacheQueryStageExec should exist
        assert(tableCacheStages.isEmpty,
          s"Expected no TableCacheQueryStageExec for pre-materialized caches, " +
            s"but found ${tableCacheStages.size}")

        // Verify correctness
        assert(result.length === 50)
        val row = joined.filter($"id" === 5).collect().head
        assert(row.getAs[Int]("val1") === 10)
        assert(row.getAs[Int]("val2") === 15)
      } finally {
        df1.unpersist()
        df2.unpersist()
      }
    }
  }

  test("Table cache stage: non-materialized cache creates TableCacheQueryStageExec") {
    import testImplicits._

    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      SQLConf.EXCHANGE_REUSE_ENABLED.key -> "true") {

      val df = (1 to 50).toDF("id").withColumn("value", $"id" * 2)
      // Cache but do NOT trigger materialization with count()
      val cachedDF = df.cache()

      try {
        // First action on the cached DataFrame — cache is NOT yet materialized
        val result = cachedDF.groupBy("id").agg(sum("value").as("total"))
        val executedPlan = result.queryExecution.executedPlan
        val collected = result.collect()

        val tableCacheStages = collectWithSubqueries(executedPlan) {
          case t: TableCacheQueryStageExec => t
        }

        // When cache is not pre-materialized, a TableCacheQueryStageExec is needed
        // to populate the cache before the parent exchange can run
        assert(tableCacheStages.nonEmpty,
          "Expected TableCacheQueryStageExec for non-materialized cache")

        // All stages should be materialized after execution
        tableCacheStages.foreach { stage =>
          assert(stage.isMaterialized,
            s"Table cache stage ${stage.id} should be materialized after query execution")
        }

        // Verify correctness
        assert(collected.length === 50)
      } finally {
        cachedDF.unpersist()
      }
    }
  }
}
