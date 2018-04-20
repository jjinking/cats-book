package catsbook.unit

import org.scalatest._
import catsbook.Ch4._

class Ch4Spec extends FlatSpec with Matchers {

  it should "Section4_7_3 Writer Monad" in {
    import Section4_7_3._

    import scala.concurrent._
    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration._

    val results = Await.result(
      Future.sequence(
        Vector(
          Future(factorialWithWriter(3)),
          Future(factorialWithWriter(2))
        )), 5.seconds)

    val r1 = results(0)
    val r2 = results(1)
    r1.written shouldEqual Vector("fact 0 1", "fact 1 1", "fact 2 2", "fact 3 6")
    //r1.value shouldEqual 6
    r2.written shouldEqual Vector("fact 0 1", "fact 1 1", "fact 2 2")
    //r2.value shouldEqual 2
  }

  it should "Section4_8_3 DbReader" in {
    import Section4_8_3._

    val users = Map(
      1 -> "dade",
      2 -> "kate",
      3 -> "margo"
    )

    val passwords = Map(
      "dade"  -> "zerocool",
      "kate"  -> "acidburn",
      "margo" -> "secret"
    )

    val db = Db(users, passwords)

    checkLogin(1, "zerocool").run(db) shouldEqual true
    checkLogin(4, "davinci").run(db) shouldEqual false
  }

  it should "Section4_9_3 Post order calculator" in {
    import Section4_9_3._

    val program = for {
      _   <- evalOne("1")
      _   <- evalOne("2")
      ans <- evalOne("+")
    } yield ans
    program.runA(Nil).value shouldEqual 3

    val program2 = evalAll(List("1", "2", "+", "3", "*"))
    program2.runA(Nil).value shouldEqual 9

    evalInput("1 2 + 4   *") shouldEqual 12
    evalInput("1 2 + 3 4 + *") shouldEqual 21
  }

}
