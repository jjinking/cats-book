package catsbook.unit

import org.scalatest._
import catsbook.Ch1._

class Ch1Spec extends FlatSpec with Matchers {

  it should "Section3" in {
    Section3.run
  }

  it should "Section4" in {
    import Section4._
    import cats.syntax.show._

    thor.show shouldEqual "Thor is a 0 year-old orange cat."
    loki.show shouldEqual "Loki is a 0 year-old black cat."
  }

  it should "Section5" in {
    // val convertToEqualizer = ()  // shadow ScalaTest
    Section5.run
  }
}
