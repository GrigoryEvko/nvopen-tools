// Function: sub_1A7A270
// Address: 0x1a7a270
//
unsigned __int64 __fastcall sub_1A7A270(_QWORD *a1, char a2)
{
  __int64 v4; // rdi
  _QWORD *v5; // r8
  __int64 v6; // rbx
  _QWORD *v7; // r13
  _QWORD *v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // r12
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 *v17; // r11
  char v18; // r9
  unsigned __int64 v19; // r13
  _QWORD *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  int v24; // r14d
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // r13
  bool v31; // zf
  __int64 v32; // rcx

  v4 = a1[5];
  v5 = *(_QWORD **)(v4 + 48);
  v6 = *(_QWORD *)(v4 + 56);
  if ( v5 )
  {
    v7 = v5 - 3;
    if ( a1 == v5 - 3 )
      return 0;
  }
  else
  {
    v7 = 0;
  }
  v8 = a1 + 3;
  while ( 1 )
  {
    if ( *((_BYTE *)v8 - 8) == 78 )
    {
      v9 = (_BYTE *)*(v8 - 6);
      v10 = 0;
      if ( !v9[16] )
        v10 = *(v8 - 6);
      if ( v10 == v6 )
        break;
    }
    if ( v5 == v8 )
      return 0;
    v8 = (_QWORD *)(*v8 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !v8 )
      BUG();
  }
  if ( (*((_WORD *)v8 - 3) & 3u) - 1 <= 1 && a2 )
    return 0;
  v11 = (unsigned __int64)(v8 - 3);
  v13 = *(_QWORD *)(v6 + 80);
  if ( !v13 )
    return v11;
  if ( v4 != v13 - 24 )
    return v11;
  if ( v11 != sub_1A7A220((__int64)(v7 + 3)) )
    return v11;
  v15 = sub_1A7A220(*(_QWORD *)(v14 + 8));
  if ( v18 || v15 != v16 || sub_14A29D0(v17, v9) )
    return v11;
  v19 = v11 & 0xFFFFFFFFFFFFFFF8LL;
  v20 = (_QWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  if ( *(char *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 23) < 0 )
  {
    v21 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
    v23 = v21 + v22;
    if ( *(char *)(v19 + 23) >= 0 )
    {
      if ( (unsigned int)(v23 >> 4) )
        goto LABEL_42;
    }
    else if ( (unsigned int)((v23 - sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL)) >> 4) )
    {
      if ( *(char *)(v19 + 23) < 0 )
      {
        v24 = *(_DWORD *)(sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL) + 8);
        if ( *(char *)(v19 + 23) >= 0 )
          BUG();
        v25 = sub_1648A40(v11 & 0xFFFFFFFFFFFFFFF8LL);
        v27 = (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v24);
        goto LABEL_26;
      }
LABEL_42:
      BUG();
    }
  }
  v27 = 0;
LABEL_26:
  v28 = v19 - 24 * v27 - 24;
  if ( (*(_BYTE *)(v6 + 18) & 1) != 0 )
  {
    sub_15E08E0(v6, (__int64)v9);
    v29 = *(_QWORD *)(v6 + 88);
    v30 = v29;
    if ( (*(_BYTE *)(v6 + 18) & 1) != 0 )
    {
      sub_15E08E0(v6, (__int64)v9);
      v29 = *(_QWORD *)(v6 + 88);
    }
  }
  else
  {
    v29 = *(_QWORD *)(v6 + 88);
    v30 = v29;
  }
  v31 = v28 == (_QWORD)v20;
  v32 = v29 + 40LL * *(_QWORD *)(v6 + 96);
  while ( !v31 )
  {
    if ( v30 == v32 || *v20 != v30 )
      return v11;
    v20 += 3;
    v30 += 40;
    v31 = v20 == (_QWORD *)v28;
  }
  if ( v30 == v32 )
    return 0;
  return v11;
}
