// Function: sub_16949F0
// Address: 0x16949f0
//
__int64 *__fastcall sub_16949F0(__int64 *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  __int64 v6; // rcx
  __int64 i; // rax

  *a1 = (__int64)(a1 + 2);
  sub_1693C00(a1, "__profn_", (__int64)"");
  if ( a3 > 0x3FFFFFFFFFFFFFFFLL - a1[1] )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(a1, a2, a3, v6);
  if ( (unsigned int)(a4 - 7) <= 1 )
  {
    for ( i = sub_22418E0(a1, "-:<>/\"'", 0, 7); i != -1; i = sub_22418E0(a1, "-:<>/\"'", i + 1, 7) )
      *(_BYTE *)(*a1 + i) = 95;
  }
  return a1;
}
