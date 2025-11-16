// Function: sub_9D82B0
// Address: 0x9d82b0
//
void __fastcall sub_9D82B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v6; // esi
  __int64 v7; // rcx
  int v8; // r15d
  _QWORD *v9; // r10
  unsigned __int64 v10; // r13
  unsigned int v11; // r8d
  _QWORD *v12; // rax
  __int64 v13; // r14
  int v14; // eax
  int v15; // ecx
  int v16; // ecx
  int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // r8
  int v21; // r11d
  _QWORD *v22; // r9
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // eax
  int v27; // r11d
  __int64 v28; // rdi
  __int64 v29; // [rsp-40h] [rbp-40h]
  __int64 v30; // [rsp-40h] [rbp-40h]

  if ( a3 == a2 || !a3 )
    return;
  v5 = a1 + 304;
  v6 = *(_DWORD *)(a1 + 328);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 304);
    goto LABEL_23;
  }
  v7 = *(_QWORD *)(a1 + 312);
  v8 = 1;
  v9 = 0;
  v10 = (0xBF58476D1CE4E5B9LL * a3) ^ ((0xBF58476D1CE4E5B9LL * a3) >> 31);
  v11 = v10 & (v6 - 1);
  v12 = (_QWORD *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a3 )
  {
LABEL_5:
    if ( v12[1] != a2 )
      v12[1] = 0;
    return;
  }
  while ( v13 != -1 )
  {
    if ( !v9 && v13 == -2 )
      v9 = v12;
    v11 = (v6 - 1) & (v8 + v11);
    v12 = (_QWORD *)(v7 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a3 )
      goto LABEL_5;
    ++v8;
  }
  if ( !v9 )
    v9 = v12;
  v14 = *(_DWORD *)(a1 + 320);
  ++*(_QWORD *)(a1 + 304);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v6 )
  {
LABEL_23:
    v29 = a3;
    sub_9D80B0(v5, 2 * v6);
    v16 = *(_DWORD *)(a1 + 328);
    if ( v16 )
    {
      a3 = v29;
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 312);
      v15 = *(_DWORD *)(a1 + 320) + 1;
      v19 = v17 & (((0xBF58476D1CE4E5B9LL * a3) >> 31) ^ (484763065 * a3));
      v9 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v9;
      if ( *v9 == v29 )
        goto LABEL_19;
      v21 = 1;
      v22 = 0;
      while ( v20 != -1 )
      {
        if ( v20 == -2 && !v22 )
          v22 = v9;
        v19 = v17 & (v21 + v19);
        v9 = (_QWORD *)(v18 + 16LL * v19);
        v20 = *v9;
        if ( *v9 == v29 )
          goto LABEL_19;
        ++v21;
      }
LABEL_27:
      if ( v22 )
        v9 = v22;
      goto LABEL_19;
    }
LABEL_43:
    ++*(_DWORD *)(a1 + 320);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 324) - v15 <= v6 >> 3 )
  {
    v30 = a3;
    sub_9D80B0(v5, v6);
    v23 = *(_DWORD *)(a1 + 328);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 312);
      v22 = 0;
      v26 = v24 & v10;
      a3 = v30;
      v27 = 1;
      v15 = *(_DWORD *)(a1 + 320) + 1;
      v9 = (_QWORD *)(v25 + 16LL * (v24 & (unsigned int)v10));
      v28 = *v9;
      if ( *v9 == v30 )
        goto LABEL_19;
      while ( v28 != -1 )
      {
        if ( v28 == -2 && !v22 )
          v22 = v9;
        v26 = v24 & (v27 + v26);
        v9 = (_QWORD *)(v25 + 16LL * v26);
        v28 = *v9;
        if ( *v9 == v30 )
          goto LABEL_19;
        ++v27;
      }
      goto LABEL_27;
    }
    goto LABEL_43;
  }
LABEL_19:
  *(_DWORD *)(a1 + 320) = v15;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 324);
  *v9 = a3;
  v9[1] = a2;
}
