// Function: sub_10D0060
// Address: 0x10d0060
//
bool __fastcall sub_10D0060(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  unsigned __int8 *v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // r8
  int v11; // eax
  __int64 v12; // rcx
  unsigned __int8 *v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // r8
  int v16; // eax
  char v17; // al
  __int64 v18; // rcx
  unsigned __int8 *v19; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v20; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 || *(_QWORD *)(v5 + 8) || *(_BYTE *)v4 != 56 )
    goto LABEL_4;
  v13 = *(unsigned __int8 **)(v4 - 64);
  v14 = *v13;
  if ( (unsigned __int8)v14 <= 0x1Cu )
  {
    if ( (_BYTE)v14 != 5 )
      goto LABEL_4;
    v16 = *((unsigned __int16 *)v13 + 1);
    if ( (*((_WORD *)v13 + 1) & 0xFFFD) != 0xD && (v16 & 0xFFF7) != 0x11 )
      goto LABEL_4;
  }
  else
  {
    if ( (unsigned __int8)v14 > 0x36u )
      goto LABEL_4;
    v15 = 0x40540000000000LL;
    if ( !_bittest64(&v15, v14) )
      goto LABEL_4;
    v16 = (unsigned __int8)v14 - 29;
  }
  if ( v16 != 15 || (v13[1] & 4) == 0 || (v20 = a3, v17 = sub_10B8310((_QWORD **)a1, (__int64)v13), a3 = v20, !v17) )
  {
LABEL_4:
    v6 = *((_QWORD *)a3 - 4);
    goto LABEL_5;
  }
  result = sub_F17ED0((_QWORD *)(a1 + 16), *(_QWORD *)(v18 - 32));
  a3 = v20;
  v6 = *((_QWORD *)v20 - 4);
  if ( !result || **(_QWORD **)(a1 + 24) != v6 )
  {
LABEL_5:
    v7 = *(_QWORD *)(v6 + 16);
    if ( !v7 || *(_QWORD *)(v7 + 8) || *(_BYTE *)v6 != 56 )
      return 0;
    v8 = *(unsigned __int8 **)(v6 - 64);
    v9 = *v8;
    if ( (unsigned __int8)v9 <= 0x1Cu )
    {
      if ( (_BYTE)v9 != 5 )
        return 0;
      v11 = *((unsigned __int16 *)v8 + 1);
      if ( (*((_WORD *)v8 + 1) & 0xFFF7) != 0x11 && (v11 & 0xFFFD) != 0xD )
        return 0;
    }
    else
    {
      if ( (unsigned __int8)v9 > 0x36u )
        return 0;
      v10 = 0x40540000000000LL;
      if ( !_bittest64(&v10, v9) )
        return 0;
      v11 = (unsigned __int8)v9 - 29;
    }
    v19 = a3;
    if ( v11 == 15
      && (v8[1] & 4) != 0
      && (unsigned __int8)sub_10B8310((_QWORD **)a1, (__int64)v8)
      && sub_F17ED0((_QWORD *)(a1 + 16), *(_QWORD *)(v12 - 32)) )
    {
      return **(_QWORD **)(a1 + 24) == *((_QWORD *)v19 - 8);
    }
    return 0;
  }
  return result;
}
