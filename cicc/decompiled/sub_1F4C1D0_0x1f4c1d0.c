// Function: sub_1F4C1D0
// Address: 0x1f4c1d0
//
char __fastcall sub_1F4C1D0(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rsi
  __int16 v4; // ax
  _WORD *v5; // rsi
  __int16 v6; // ax
  bool v7; // cf

  v2 = *(unsigned __int16 *)(*(_QWORD *)(a1[23] + 8LL) + ((unsigned __int64)a2 << 6) + 6);
  if ( sub_1F4B690((__int64)a1) )
  {
    v3 = 0;
    if ( sub_1F4B690((__int64)a1) )
      v3 = a1 + 9;
    LOBYTE(v4) = sub_38D7610((unsigned __int16)v2, v3);
  }
  else
  {
    LOBYTE(v4) = sub_1F4B670((__int64)a1);
    if ( (_BYTE)v4 )
    {
      v5 = (_WORD *)(a1[5] + 14 * v2);
      v6 = *v5 & 0x3FFF;
      v7 = v6 == 16382;
      v4 = v6 - 16382;
      if ( !v7 && v4 != 1 )
        LOBYTE(v4) = sub_38D7430(a1[22], v5);
    }
  }
  return v4;
}
