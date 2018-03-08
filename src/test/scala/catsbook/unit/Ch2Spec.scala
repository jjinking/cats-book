package catsbook.unit

import cats.syntax.semigroup._
import org.scalatest._
import catsbook.Ch2._

class Ch2Spec extends FlatSpec with Matchers {

  val intSet1 = Set[Int](1, 2, 3)
  val intSet2 = Set.empty[Int]

  it should "Section3 UnionSet" in {
    import Section4.UnionSet._

    intSet1 |+| intSet2 shouldEqual intSet1
    intSet2 |+| intSet2 shouldEqual intSet2
  }

  it should "Section3 IntersectSet" in {
    import Section4.IntersectSet._

    intSet1 |+| intSet2 shouldEqual intSet2
    intSet1 |+| intSet1 shouldEqual intSet1
  }

}
