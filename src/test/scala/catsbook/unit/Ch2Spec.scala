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

  it should "Section5" in {
    import Section5._

    add(List(1,2,3)) shouldEqual 6
  }

  it should "Section5Option" in {
    import Section5Option._
    import cats.instances.int._
    import cats.instances.option._

    add(List(Some(3), None, Some(4))) shouldEqual Option(7)

    val o1 = Order(1, 2)
    val o2 = Order(3, 4)
    val o3 = Order(4, 6)
    add(List(o1, o2)) shouldEqual o3
  }

}
