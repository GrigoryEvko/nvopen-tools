// Function: sub_E0CB60
// Address: 0xe0cb60
//
char *__fastcall sub_E0CB60(int a1)
{
  if ( !a1 )
    return "EXTERNAL";
  if ( a1 != 1 )
    BUG();
  return "STATIC";
}
