// Function: sub_216EC30
// Address: 0x216ec30
//
__int64 __fastcall sub_216EC30(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rbx
  int v6; // eax

  v5 = sub_1F43D80(a1[2], *a1, a3, a4);
  v6 = sub_1F43D70(a1[2], a2);
  if ( v6 != 54 )
  {
    if ( v6 > 54 )
    {
      if ( (unsigned int)(v6 - 118) > 2 || BYTE4(v5) != 6 )
        return sub_216EA10(a1, a2, a3, 0, 0);
      return (unsigned int)(2 * v5);
    }
    if ( v6 != 52 )
      return sub_216EA10(a1, a2, a3, 0, 0);
  }
  if ( BYTE4(v5) != 6 )
    return sub_216EA10(a1, a2, a3, 0, 0);
  return (unsigned int)(2 * v5);
}
