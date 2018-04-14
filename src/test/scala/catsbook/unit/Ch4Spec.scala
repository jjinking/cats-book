package catsbook.unit

import org.scalatest._
import catsbook.Ch4._

class Ch4Spec extends FlatSpec with Matchers {

  it should "Section4 Writer Monad" in {
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

}
