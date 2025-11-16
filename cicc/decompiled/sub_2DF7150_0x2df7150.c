// Function: sub_2DF7150
// Address: 0x2df7150
//
__int64 __fastcall sub_2DF7150(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v6; // esi
  __int64 v7; // r10
  _DWORD *v8; // r8
  int v9; // r14d
  unsigned int v10; // r9d
  int *v11; // rax
  int v12; // ecx
  __int64 v13; // rcx
  _QWORD *v14; // r8
  __int64 result; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 i; // rcx
  int v20; // eax
  int v21; // ecx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // eax
  int v26; // edi
  int v27; // r11d
  _DWORD *v28; // r10
  int v29; // eax
  int v30; // eax
  __int64 v31; // r13
  __int64 v32; // r9
  int v33; // edi
  int v34; // esi
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v5 = a1 + 1112;
  v6 = *(_DWORD *)(a1 + 1136);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 1112);
    goto LABEL_29;
  }
  v7 = *(_QWORD *)(a1 + 1120);
  v8 = 0;
  v9 = 1;
  v10 = (v6 - 1) & (37 * a2);
  v11 = (int *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( *v11 == a2 )
  {
LABEL_3:
    v13 = *((_QWORD *)v11 + 1);
    v14 = v11 + 2;
    goto LABEL_4;
  }
  while ( v12 != -1 )
  {
    if ( !v8 && v12 == -2 )
      v8 = v11;
    v10 = (v6 - 1) & (v9 + v10);
    v11 = (int *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v8 )
    v8 = v11;
  v20 = *(_DWORD *)(a1 + 1128);
  ++*(_QWORD *)(a1 + 1112);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v6 )
  {
LABEL_29:
    v35 = a3;
    sub_2DF61B0(v5, 2 * v6);
    v22 = *(_DWORD *)(a1 + 1136);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 1120);
      a3 = v35;
      v25 = (v22 - 1) & (37 * a2);
      v21 = *(_DWORD *)(a1 + 1128) + 1;
      v8 = (_DWORD *)(v24 + 16LL * v25);
      v26 = *v8;
      if ( *v8 == a2 )
        goto LABEL_24;
      v27 = 1;
      v28 = 0;
      while ( v26 != -1 )
      {
        if ( !v28 && v26 == -2 )
          v28 = v8;
        v25 = v23 & (v27 + v25);
        v8 = (_DWORD *)(v24 + 16LL * v25);
        v26 = *v8;
        if ( *v8 == a2 )
          goto LABEL_24;
        ++v27;
      }
LABEL_33:
      if ( v28 )
        v8 = v28;
      goto LABEL_24;
    }
LABEL_49:
    ++*(_DWORD *)(a1 + 1128);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 1132) - v21 <= v6 >> 3 )
  {
    v36 = a3;
    sub_2DF61B0(v5, v6);
    v29 = *(_DWORD *)(a1 + 1136);
    if ( v29 )
    {
      v30 = v29 - 1;
      v28 = 0;
      a3 = v36;
      LODWORD(v31) = v30 & (37 * a2);
      v32 = *(_QWORD *)(a1 + 1120);
      v21 = *(_DWORD *)(a1 + 1128) + 1;
      v33 = 1;
      v8 = (_DWORD *)(v32 + 16LL * (unsigned int)v31);
      v34 = *v8;
      if ( *v8 == a2 )
        goto LABEL_24;
      while ( v34 != -1 )
      {
        if ( v34 == -2 && !v28 )
          v28 = v8;
        v31 = v30 & (unsigned int)(v31 + v33);
        v8 = (_DWORD *)(v32 + 16 * v31);
        v34 = *v8;
        if ( *v8 == a2 )
          goto LABEL_24;
        ++v33;
      }
      goto LABEL_33;
    }
    goto LABEL_49;
  }
LABEL_24:
  *(_DWORD *)(a1 + 1128) = v21;
  if ( *v8 != -1 )
    --*(_DWORD *)(a1 + 1132);
  *v8 = a2;
  v13 = 0;
  v14 = v8 + 2;
  *v14 = 0;
LABEL_4:
  result = *(_QWORD *)(a3 + 40);
  do
  {
    v16 = result;
    result = *(_QWORD *)(result + 40);
  }
  while ( v16 != result );
  *(_QWORD *)(a3 + 40) = result;
  if ( v13 )
  {
    v17 = *(_QWORD *)(v13 + 40);
    do
    {
      v18 = v17;
      v17 = *(_QWORD *)(v17 + 40);
    }
    while ( v18 != v17 );
    *(_QWORD *)(v13 + 40) = v17;
    if ( v16 != v17 )
    {
      for ( i = *(_QWORD *)(result + 48); i; i = *(_QWORD *)(i + 48) )
      {
        *(_QWORD *)(result + 40) = v18;
        result = i;
      }
      *(_QWORD *)(result + 40) = v18;
      *(_QWORD *)(result + 48) = *(_QWORD *)(v18 + 48);
      *(_QWORD *)(v18 + 48) = v16;
    }
  }
  else
  {
    v17 = result;
  }
  *v14 = v17;
  return result;
}
