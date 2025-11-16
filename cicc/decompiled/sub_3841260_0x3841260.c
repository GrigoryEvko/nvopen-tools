// Function: sub_3841260
// Address: 0x3841260
//
unsigned __int8 *__fastcall sub_3841260(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __m128i a5)
{
  int v5; // eax

  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 <= 390 )
  {
    if ( v5 <= 388 )
    {
      if ( v5 <= 386 )
      {
        if ( v5 > 381 )
          return (unsigned __int8 *)sub_37AE0F0(a1, a3, a4);
        goto LABEL_12;
      }
      return sub_383B380(a1, a3, a4);
    }
  }
  else
  {
    if ( v5 <= 477 )
    {
      if ( v5 <= 475 )
      {
        if ( (unsigned int)(v5 - 471) <= 4 )
          return (unsigned __int8 *)sub_37AE0F0(a1, a3, a4);
LABEL_12:
        BUG();
      }
      return sub_383B380(a1, a3, a4);
    }
    if ( (unsigned int)(v5 - 478) > 1 )
      goto LABEL_12;
  }
  return sub_37AF270(a1, a3, a4, a5);
}
