package catsbook.unit

import org.scalatest._
import catsbook.Ch3._

class Ch3Spec extends FlatSpec with Matchers {

  it should "Section5 Tree Functor" in {
    import Section5._

    val l1 = Leaf(1)
    val l2 = Leaf(2)
    val l3 = Leaf(3)
    val br23 = Branch(l2, l3)
    val root = Branch(l1, br23)

    val root2 = Branch(
      Leaf(4),
      Branch(
        Leaf(5),
        Leaf(6)
      )
    )
    treeFunctor.map(root)(_ + 3) shouldEqual root2

    // With syntax extension
    import cats.syntax.functor._

    // Just label the type for the root node as Tree[A]
    val root3 = Tree.branch(
      Leaf(1),
      Branch(
        Leaf(2),
        Leaf(3)
      )
    )
    root3.map(_ + 3) shouldEqual root2

    // Same idea
    val root4: Tree[Int] = root
    root4.map(_ + 3) shouldEqual root2
  }

  it should "Section6 Contravariant Printable" in {
    import Section6._

    format(Box("hello world")) shouldEqual "\"hello world\""

    format(Box(true)) shouldEqual "yes"

  }

}
