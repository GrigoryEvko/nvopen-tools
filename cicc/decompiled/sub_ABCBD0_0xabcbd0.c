// Function: sub_ABCBD0
// Address: 0xabcbd0
//
__int64 __fastcall sub_ABCBD0(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, char a5)
{
  if ( a3 == 17 )
  {
    sub_ABAE70(a1, a2, a4, a5, 0);
    return a1;
  }
  if ( a3 > 0x11 )
  {
    if ( a3 == 25 )
    {
      sub_AB9570(a1, a2, a4, a5, 0);
      return a1;
    }
    goto LABEL_9;
  }
  if ( a3 != 13 )
  {
    if ( a3 == 15 )
    {
      sub_ABA6B0(a1, a2, (__int64)a4, a5, 0);
      return a1;
    }
LABEL_9:
    sub_ABCAA0(a1, a2, a3, a4);
    return a1;
  }
  sub_ABA0E0(a1, a2, (__int64)a4, a5, 0);
  return a1;
}
