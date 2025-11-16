// Function: sub_11F1ED0
// Address: 0x11f1ed0
//
__int64 __fastcall sub_11F1ED0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rdx
  __int64 v7; // rax

  if ( a3 > 0xD7 )
  {
    v6 = a3 - 436;
    if ( (unsigned int)v6 > 0x3B )
      return 0;
    v7 = 0x8C0000000000023LL;
    if ( !_bittest64(&v7, v6) )
      return 0;
    return sub_11F1B40(a2, 0, a4, a4, a5);
  }
  if ( a3 > 0xD4 )
    return sub_11F1B40(a2, 0, a4, a4, a5);
  if ( a3 > 0xCF )
  {
    if ( a3 != 211 )
      return 0;
  }
  else if ( a3 <= 0xCD )
  {
    return 0;
  }
  return sub_11F1B40(a2, 1, a4, a4, a5);
}
