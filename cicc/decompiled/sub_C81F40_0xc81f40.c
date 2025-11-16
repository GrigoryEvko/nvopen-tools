// Function: sub_C81F40
// Address: 0xc81f40
//
char *__fastcall sub_C81F40(char *a1, unsigned __int64 a2, unsigned int a3)
{
  while ( a2 > 2 && *a1 == 46 && sub_C80220(a1[1], a3) )
  {
    a2 -= 2LL;
    a1 += 2;
    while ( sub_C80220(*a1, a3) )
    {
      ++a1;
      if ( !--a2 )
        return a1;
    }
  }
  return a1;
}
