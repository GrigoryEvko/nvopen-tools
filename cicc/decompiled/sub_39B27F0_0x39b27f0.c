// Function: sub_39B27F0
// Address: 0x39b27f0
//
char __fastcall sub_39B27F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  _DWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  _DWORD *v13; // r8
  _DWORD *v14; // rdi
  int v15; // esi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rbx
  __int64 v22; // r13
  char v23; // al
  __int64 v24; // rdx
  _BYTE **v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  int v29; // [rsp+4h] [rbp-3Ch]

  v6 = *(_QWORD *)(a1 + 16);
  v7 = sub_16D5D50();
  v8 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_26;
  v9 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v10 = v8[2];
      v11 = v8[3];
      if ( v7 <= v8[4] )
        break;
      v8 = (_QWORD *)v8[3];
      if ( !v11 )
        goto LABEL_6;
    }
    v9 = v8;
    v8 = (_QWORD *)v8[2];
  }
  while ( v10 );
LABEL_6:
  if ( v9 == dword_4FA0208 )
    goto LABEL_26;
  if ( v7 < *((_QWORD *)v9 + 4) )
    goto LABEL_26;
  v12 = *((_QWORD *)v9 + 7);
  v13 = v9 + 12;
  if ( !v12 )
    goto LABEL_26;
  v14 = v9 + 12;
  v15 = qword_5057460[1];
  do
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v12 + 16);
      v17 = *(_QWORD *)(v12 + 24);
      if ( *(_DWORD *)(v12 + 32) >= v15 )
        break;
      v12 = *(_QWORD *)(v12 + 24);
      if ( !v17 )
        goto LABEL_13;
    }
    v14 = (_DWORD *)v12;
    v12 = *(_QWORD *)(v12 + 16);
  }
  while ( v16 );
LABEL_13:
  if ( v13 == v14 || v15 < v14[8] || (int)v14[9] <= 0 )
  {
LABEL_26:
    LODWORD(v25) = *(_DWORD *)(*(_QWORD *)(v6 + 160) + 8LL);
    v29 = (int)v25;
    if ( !(_DWORD)v25 )
      return (char)v25;
  }
  else
  {
    v29 = qword_5057460[20];
  }
  v18 = *(_QWORD *)(a2 + 32);
  v19 = *(_QWORD *)(a2 + 40);
  v20 = a1 + 8;
  if ( v18 != v19 )
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)(*(_QWORD *)v18 + 48LL);
      v22 = *(_QWORD *)v18 + 40LL;
      if ( v21 != v22 )
        break;
LABEL_23:
      v18 += 8;
      if ( v19 == v18 )
        goto LABEL_24;
    }
    while ( 1 )
    {
      if ( !v21 )
        BUG();
      v23 = *(_BYTE *)(v21 - 8);
      v24 = v21 - 24;
      if ( v23 == 78 )
      {
        v26 = v24 | 4;
      }
      else
      {
        if ( v23 != 29 )
          goto LABEL_22;
        v26 = v24 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v26 & 4) != 0 )
        v25 = (_BYTE **)(v27 - 24);
      else
        v25 = (_BYTE **)(v27 - 72);
      if ( (*v25)[16] )
        return (char)v25;
      LOBYTE(v25) = sub_14A2090(v20, *v25);
      if ( (_BYTE)v25 )
        return (char)v25;
LABEL_22:
      v21 = *(_QWORD *)(v21 + 8);
      if ( v22 == v21 )
        goto LABEL_23;
    }
  }
LABEL_24:
  LOBYTE(v25) = a4;
  *(_BYTE *)(a4 + 49) = 1;
  *(_DWORD *)(a4 + 12) = v29;
  *(_DWORD *)(a4 + 8) = 0;
  *(_DWORD *)(a4 + 16) = 0;
  *(_DWORD *)(a4 + 40) = 2;
  *(_WORD *)(a4 + 44) = 257;
  return (char)v25;
}
