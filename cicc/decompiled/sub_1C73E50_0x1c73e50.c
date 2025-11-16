// Function: sub_1C73E50
// Address: 0x1c73e50
//
__int64 __fastcall sub_1C73E50(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  _QWORD *v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // rax
  unsigned int v10; // r12d
  __int64 v12; // r12
  __int64 v13; // r15
  bool v14; // r13
  _QWORD *v15; // r14
  __int64 v16; // rax
  __int64 v17; // r15
  _QWORD *v18; // rax
  char v19; // dl
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rax
  char v28; // dl

  v2 = *(_QWORD *)(a1 + 152);
  v3 = *(_QWORD *)(v2 + 48);
  if ( v3 == v2 + 40 )
  {
LABEL_5:
    if ( *(_QWORD *)(a1 + 160) != sub_157F1C0(v2) )
      return 0;
    if ( sub_157F0B0(*(_QWORD *)(a1 + 152)) )
      goto LABEL_9;
    v4 = *(_QWORD **)(a1 + 160);
  }
  else
  {
    while ( 1 )
    {
      if ( !v3 )
        JUMPOUT(0x41FEE0);
      if ( *(_BYTE *)(v3 - 8) != 26 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v3 == v2 + 40 )
        goto LABEL_5;
    }
    v9 = sub_157F1C0(v2);
    v4 = *(_QWORD **)(a1 + 160);
    if ( v4 != (_QWORD *)v9 )
      return 0;
  }
  v5 = sub_1AA91E0(*(_QWORD **)(a1 + 152), v4, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 16));
  *(_BYTE *)(a1 + 265) = 1;
  *(_QWORD *)(a1 + 152) = v5;
LABEL_9:
  v6 = *(_QWORD *)(a1 + 176);
  v7 = *(_QWORD *)(v6 + 48);
  if ( v7 != v6 + 40 )
  {
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      v8 = *(_BYTE *)(v7 - 8);
      if ( v8 != 77 && v8 != 26 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 + 40 == v7 )
        goto LABEL_20;
    }
    if ( *(_QWORD *)(a1 + 168) != sub_157F0B0(v6) )
      return 0;
    goto LABEL_16;
  }
LABEL_20:
  if ( *(_QWORD *)(a1 + 168) != sub_157F0B0(v6) )
    return 0;
  if ( !sub_157F1C0(*(_QWORD *)(a1 + 176)) )
  {
LABEL_16:
    sub_1AA91E0(*(_QWORD **)(a1 + 168), *(_QWORD **)(a1 + 176), *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 16));
    *(_BYTE *)(a1 + 265) = 1;
  }
  v12 = *(_QWORD *)(a1 + 152);
  v13 = *(_QWORD *)(a1 + 176);
  v14 = v13 == 0 || v12 == 0;
  if ( v14 )
    return 0;
  v15 = (_QWORD *)sub_157F0B0(*(_QWORD *)(a1 + 152));
  if ( !v15 )
    return 0;
  v16 = sub_157F1C0(v13);
  v17 = v16;
  if ( !v16 )
    return 0;
  v10 = sub_1C73B00((__int64)v15, v12, v16);
  if ( !(_BYTE)v10
    || (unsigned int)sub_1C73DD0((__int64)v15) != 1
    || (unsigned int)sub_1C73DD0(v17) != 2
    || !sub_157F1C0(v17) )
  {
    return 0;
  }
  v18 = (_QWORD *)v15[6];
  if ( v15 + 5 != v18 )
  {
    while ( 1 )
    {
      if ( !v18 )
        BUG();
      v19 = *((_BYTE *)v18 - 8);
      if ( v19 != 75 && v19 != 26 )
        break;
      v18 = (_QWORD *)v18[1];
      if ( v15 + 5 == v18 )
        goto LABEL_36;
    }
    v14 = v10;
  }
LABEL_36:
  v20 = sub_157EBA0((__int64)v15);
  if ( *(_BYTE *)(v20 + 16) != 26 )
    return 0;
  v21 = *(_QWORD *)(v20 - 72);
  if ( *(_BYTE *)(v21 + 16) != 75 )
    return 0;
  v22 = *(_QWORD *)(v21 + 32);
  if ( !v22 || v20 != v22 - 24 )
    return 0;
  if ( v14 )
  {
    sub_1AA8CA0(v15, v21, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 16));
    *(_BYTE *)(a1 + 265) = 1;
  }
  v23 = sub_157F1C0(*(_QWORD *)(a1 + 176));
  v24 = (_QWORD *)v23;
  if ( !v23 )
    return 0;
  v25 = *(_QWORD *)(v23 + 48);
  v26 = v23 + 40;
  if ( v23 + 40 != v25 )
  {
    v27 = *(_QWORD *)(v23 + 48);
    while ( 1 )
    {
      if ( !v27 )
        BUG();
      v28 = *(_BYTE *)(v27 - 8);
      if ( v28 != 77 && v28 != 26 )
        break;
      v27 = *(_QWORD *)(v27 + 8);
      if ( v26 == v27 )
        return v10;
    }
    if ( v25 )
      v25 -= 24;
    sub_1AA8CA0(v24, v25, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 16));
    *(_BYTE *)(a1 + 265) = 1;
  }
  return v10;
}
