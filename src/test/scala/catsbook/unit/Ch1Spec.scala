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
    import Section5._
    import cats.syntax.option._

    val cat1 = Cat("Garfield",   38, "orange and black")
    val cat2 = Cat("Heathcliff", 33, "orange and black")
    val optionCat1 = Option(cat1)
    val optionCat2 = Option.empty[Cat]

    (thor =!= loki) shouldEqual true
    (cat1 === cat1) shouldEqual true
    (cat1 =!= cat2) shouldEqual true
    (optionCat1 =!= optionCat2) shouldEqual true
    (optionCat1 === Some(cat1)) shouldEqual true
    (optionCat2 === optionCat2) shouldEqual true
  }
}
