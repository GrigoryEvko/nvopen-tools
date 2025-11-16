// Function: sub_2241520
// Address: 0x2241520
//
unsigned __int64 *__fastcall sub_2241520(unsigned __int64 *a1, char *a2)
{
  unsigned __int64 v2; // rdx

  v2 = strlen(a2);
  if ( 0x3FFFFFFFFFFFFFFFLL - a1[1] < v2 )
    sub_4262D8((__int64)"basic_string::append");
  return sub_2241490(a1, a2, v2);
}
