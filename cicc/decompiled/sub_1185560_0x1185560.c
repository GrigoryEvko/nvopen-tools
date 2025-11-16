// Function: sub_1185560
// Address: 0x1185560
//
__int64 __fastcall sub_1185560(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r14
  _BYTE *v10; // rdi
  unsigned __int8 *v11; // rbx
  unsigned int v12; // eax
  char v13; // al
  __int64 v14; // rsi
  __int64 *v15; // rbx
  unsigned int v16; // eax
  char v17; // al
  __int64 v18; // rsi

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v4 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  if ( !sub_BCAC40(v4, 1) )
    return 0;
  if ( *(_BYTE *)a2 == 58 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v15 = *(__int64 **)(a2 - 8);
    else
      v15 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v8 = *v15;
    v11 = (unsigned __int8 *)v15[4];
    LOBYTE(v2) = sub_9987C0(a1, 30, (unsigned __int8 *)v8) && v8 != 0;
    if ( !(_BYTE)v2 )
    {
      LOBYTE(v16) = sub_9987C0(a1, 30, v11);
      v2 = v16;
      if ( (_BYTE)v16 )
      {
        if ( v11 )
        {
          **(_QWORD **)(a1 + 16) = v11;
          if ( v8 )
            goto LABEL_25;
        }
      }
      return 0;
    }
    **(_QWORD **)(a1 + 16) = v8;
    if ( !v11 )
    {
      sub_9987C0(a1, 30, 0);
      return 0;
    }
LABEL_28:
    **(_QWORD **)(a1 + 24) = v11;
    return v2;
  }
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v8 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v8 + 8) )
    return 0;
  v10 = *(_BYTE **)(a2 - 64);
  if ( *v10 > 0x15u )
    return 0;
  v11 = *(unsigned __int8 **)(a2 - 32);
  LOBYTE(v12) = sub_AD7A80(v10, 1, v5, v6, v7);
  v2 = v12;
  if ( !(_BYTE)v12 )
    return 0;
  if ( *(_BYTE *)v8 == 59 )
  {
    if ( (v17 = sub_995B10((_QWORD **)a1, *(_QWORD *)(v8 - 64)), v18 = *(_QWORD *)(v8 - 32), v17)
      && v18 == *(_QWORD *)(a1 + 8)
      || (unsigned __int8)sub_995B10((_QWORD **)a1, v18) && *(_QWORD *)(v8 - 64) == *(_QWORD *)(a1 + 8) )
    {
      **(_QWORD **)(a1 + 16) = v8;
      if ( v11 )
        goto LABEL_28;
    }
  }
  if ( *v11 == 59 )
  {
    if ( (v13 = sub_995B10((_QWORD **)a1, *((_QWORD *)v11 - 8)), v14 = *((_QWORD *)v11 - 4), v13)
      && v14 == *(_QWORD *)(a1 + 8)
      || (unsigned __int8)sub_995B10((_QWORD **)a1, v14) && *((_QWORD *)v11 - 8) == *(_QWORD *)(a1 + 8) )
    {
      **(_QWORD **)(a1 + 16) = v11;
LABEL_25:
      **(_QWORD **)(a1 + 24) = v8;
      return v2;
    }
  }
  return 0;
}
