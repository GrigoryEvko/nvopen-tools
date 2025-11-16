// Function: sub_10BF400
// Address: 0x10bf400
//
__int64 __fastcall sub_10BF400(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  unsigned int v6; // r8d
  unsigned int v7; // r8d

  v5 = *(_QWORD *)(a2 - 32);
  if ( *(_QWORD *)(v5 + 8) == *(_QWORD *)(a2 + 8) )
  {
    return 0;
  }
  else
  {
    v6 = 0;
    if ( *(_BYTE *)v5 > 0x15u )
    {
      v6 = 1;
      if ( (unsigned __int8)(*(_BYTE *)v5 - 67) <= 0xCu )
      {
        LOBYTE(v7) = (unsigned int)sub_10FFC90(a1, v5, a2, a4, 1) == 0;
        return v7;
      }
    }
  }
  return v6;
}
