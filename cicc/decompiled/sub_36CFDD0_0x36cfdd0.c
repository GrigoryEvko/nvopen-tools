// Function: sub_36CFDD0
// Address: 0x36cfdd0
//
__int64 __fastcall sub_36CFDD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  unsigned __int8 *v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int16 v11; // bx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  bool v15; // zf

  if ( !(_BYTE)qword_50409C8 )
    return 3;
  v4 = 0;
  if ( *(_BYTE *)a2 == 85 )
  {
    v4 = *(_QWORD *)(a2 - 32);
    if ( v4 )
    {
      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a2 + 80) )
      {
        v15 = (*(_BYTE *)(v4 + 33) & 0x20) == 0;
        v4 = 0;
        if ( !v15 )
          v4 = a2;
      }
      else
      {
        v4 = 0;
      }
    }
  }
  if ( *(_BYTE *)a3 != 85 )
    return 3;
  v6 = *(unsigned __int8 **)(a3 - 32);
  if ( !v6 )
    return 3;
  v7 = *v6;
  if ( (_BYTE)v7
    || *((_QWORD *)v6 + 3) != *(_QWORD *)(a3 + 80)
    || (v6[33] & 0x20) == 0
    || !v4
    || !sub_36CDE90(v4) && !sub_36CDE90(a3) )
  {
    return 3;
  }
  v11 = sub_36CDF30(v10, a2, v8, v9, v10);
  if ( (v11 & (unsigned __int16)sub_36CDF30(a3, a2, v12, v13, v14)) != 0 )
    return 3;
  return v7;
}
