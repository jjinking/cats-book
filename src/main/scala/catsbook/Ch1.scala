package catsbook

object Ch1 {

  final case class Cat(name: String, age: Int, color: String)
  val thor = Cat("Thor", 0, "orange")
  val loki = Cat("Loki", 0, "black")

  trait Printable[A] {
    def format(a: A): String
  }

  object PrintableInstances {
    implicit val printableString = new Printable[String] {
      override def format(a: String): String = a
    }

    implicit val printableInt = new Printable[Int] {
      override def format(a: Int): String = a.toString
    }

    implicit val printableCat = new Printable[Cat] {
      override def format(a: Cat): String = {
        val nameFmt  = Printable.format(a.name)
        val ageFmt   = Printable.format(a.age)
        val colorFmt = Printable.format(a.color)
        s"$nameFmt is a $ageFmt year-old $colorFmt cat."
      }
    }
  }

  object Printable {
    def format[A](a: A)(implicit printableA: Printable[A]): String =
      printableA.format(a)

    def print[A](a: A)(implicit printableA: Printable[A]): Unit =
      println(format(a))
  }

  object PrintableSyntax {
    implicit class PrintableOps[A](a: A) {
      def format(implicit printableA: Printable[A]): String =
        printableA.format(a)

      def print(implicit printableA: Printable[A]): Unit =
        Printable.print(a)
    }
  }

  object Section3 {
    def run() = {
      import PrintableInstances._
      import PrintableSyntax._

      Printable.print(thor)
      println(loki.format)
      loki.print
    }
  }

  object Section4 {
    import cats._
    import cats.implicits._

    implicit val catShow = new Show[Cat] {
      def show(cat: Cat): String = {
        val nameFmt  = cat.name.show
        val ageFmt   = cat.age.show
        val colorFmt = cat.color.show
        s"$nameFmt is a $ageFmt year-old $colorFmt cat."
      }
    }

    // implicit val catShowUsingHelper = Show.show[Cat] { cat =>
    //   val nameFmt  = cat.name.show
    //   val ageFmt   = cat.age.show
    //   val colorFmt = cat.color.show
    //   s"$nameFmt is a $ageFmt year-old $colorFmt cat."
    // }
  }

  object Section5 {
    import cats.Eq
    import cats.syntax.eq._
    import cats.instances.int._
    import cats.instances.string._

    implicit val catEq: Eq[Cat] = Eq.instance[Cat] { (cat1, cat2) =>
      (cat1.name === cat2.name) &&
        (cat1.age === cat2.age) &&
        (cat1.color === cat2.color)
    }
  }

}
