// Function: sub_1185380
// Address: 0x1185380
//
__int64 __fastcall sub_1185380(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rdi
  __int64 v5; // r14
  _BYTE *v7; // rdi
  unsigned __int8 *v8; // rbx
  unsigned int v9; // eax
  char v10; // al
  __int64 v11; // rsi
  __int64 *v12; // rbx
  unsigned int v13; // eax
  char v14; // al
  __int64 v15; // rsi

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v4 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  if ( !sub_BCAC40(v4, 1) )
    return 0;
  if ( *(_BYTE *)a2 == 57 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v12 = *(__int64 **)(a2 - 8);
    else
      v12 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v5 = *v12;
    v8 = (unsigned __int8 *)v12[4];
    LOBYTE(v2) = sub_9987C0(a1, 30, (unsigned __int8 *)v5) && v5 != 0;
    if ( !(_BYTE)v2 )
    {
      LOBYTE(v13) = sub_9987C0(a1, 30, v8);
      v2 = v13;
      if ( (_BYTE)v13 )
      {
        if ( v8 )
        {
          **(_QWORD **)(a1 + 16) = v8;
          if ( v5 )
            goto LABEL_25;
        }
      }
      return 0;
    }
    **(_QWORD **)(a1 + 16) = v5;
    if ( !v8 )
    {
      sub_9987C0(a1, 30, 0);
      return 0;
    }
LABEL_28:
    **(_QWORD **)(a1 + 24) = v8;
    return v2;
  }
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v5 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v5 + 8) )
    return 0;
  v7 = *(_BYTE **)(a2 - 32);
  if ( *v7 > 0x15u )
    return 0;
  v8 = *(unsigned __int8 **)(a2 - 64);
  LOBYTE(v9) = sub_AC30F0((__int64)v7);
  v2 = v9;
  if ( !(_BYTE)v9 )
    return 0;
  if ( *(_BYTE *)v5 == 59 )
  {
    if ( (v14 = sub_995B10((_QWORD **)a1, *(_QWORD *)(v5 - 64)), v15 = *(_QWORD *)(v5 - 32), v14)
      && v15 == *(_QWORD *)(a1 + 8)
      || (unsigned __int8)sub_995B10((_QWORD **)a1, v15) && *(_QWORD *)(v5 - 64) == *(_QWORD *)(a1 + 8) )
    {
      **(_QWORD **)(a1 + 16) = v5;
      if ( v8 )
        goto LABEL_28;
    }
  }
  if ( *v8 == 59 )
  {
    if ( (v10 = sub_995B10((_QWORD **)a1, *((_QWORD *)v8 - 8)), v11 = *((_QWORD *)v8 - 4), v10)
      && v11 == *(_QWORD *)(a1 + 8)
      || (unsigned __int8)sub_995B10((_QWORD **)a1, v11) && *((_QWORD *)v8 - 8) == *(_QWORD *)(a1 + 8) )
    {
      **(_QWORD **)(a1 + 16) = v8;
LABEL_25:
      **(_QWORD **)(a1 + 24) = v5;
      return v2;
    }
  }
  return 0;
}
