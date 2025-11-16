// Function: sub_A594C0
// Address: 0xa594c0
//
_DWORD *__fastcall sub_A594C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // r14d
  unsigned int v6; // esi
  __int64 v7; // rcx
  int v8; // r15d
  _QWORD *v9; // r10
  unsigned __int64 v10; // r13
  unsigned int v11; // r8d
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _DWORD *result; // rax
  int v15; // eax
  int v16; // edx
  int v17; // edx
  int v18; // esi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 v21; // rax
  int v22; // r9d
  _QWORD *v23; // r8
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // eax
  int v28; // r9d
  __int64 v29; // rsi

  v4 = a1 + 296;
  v5 = *(_DWORD *)(v4 + 32);
  *(_DWORD *)(v4 + 32) = v5 + 1;
  v6 = *(_DWORD *)(a1 + 320);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_19;
  }
  v7 = *(_QWORD *)(a1 + 304);
  v8 = 1;
  v9 = 0;
  v10 = (0xBF58476D1CE4E5B9LL * a2) ^ ((0xBF58476D1CE4E5B9LL * a2) >> 31);
  v11 = v10 & (v6 - 1);
  v12 = (_QWORD *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a2 )
  {
LABEL_3:
    result = v12 + 1;
    goto LABEL_4;
  }
  while ( v13 != -1 )
  {
    if ( !v9 && v13 == -2 )
      v9 = v12;
    v11 = (v6 - 1) & (v8 + v11);
    v12 = (_QWORD *)(v7 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a2 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v9 )
    v9 = v12;
  v15 = *(_DWORD *)(a1 + 312);
  ++*(_QWORD *)(a1 + 296);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v6 )
  {
LABEL_19:
    sub_9E25D0(v4, 2 * v6);
    v17 = *(_DWORD *)(a1 + 320);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 304);
      v16 = *(_DWORD *)(a1 + 312) + 1;
      v20 = v18 & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
      v9 = (_QWORD *)(v19 + 16LL * v20);
      v21 = *v9;
      if ( *v9 == a2 )
        goto LABEL_15;
      v22 = 1;
      v23 = 0;
      while ( v21 != -1 )
      {
        if ( !v23 && v21 == -2 )
          v23 = v9;
        v20 = v18 & (v22 + v20);
        v9 = (_QWORD *)(v19 + 16LL * v20);
        v21 = *v9;
        if ( *v9 == a2 )
          goto LABEL_15;
        ++v22;
      }
LABEL_23:
      if ( v23 )
        v9 = v23;
      goto LABEL_15;
    }
LABEL_39:
    ++*(_DWORD *)(a1 + 312);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 316) - v16 <= v6 >> 3 )
  {
    sub_9E25D0(v4, v6);
    v24 = *(_DWORD *)(a1 + 320);
    if ( v24 )
    {
      v25 = v24 - 1;
      v23 = 0;
      v26 = *(_QWORD *)(a1 + 304);
      v27 = v25 & v10;
      v28 = 1;
      v16 = *(_DWORD *)(a1 + 312) + 1;
      v9 = (_QWORD *)(v26 + 16LL * (v25 & (unsigned int)v10));
      v29 = *v9;
      if ( *v9 == a2 )
        goto LABEL_15;
      while ( v29 != -1 )
      {
        if ( !v23 && v29 == -2 )
          v23 = v9;
        v27 = v25 & (v28 + v27);
        v9 = (_QWORD *)(v26 + 16LL * v27);
        v29 = *v9;
        if ( *v9 == a2 )
          goto LABEL_15;
        ++v28;
      }
      goto LABEL_23;
    }
    goto LABEL_39;
  }
LABEL_15:
  *(_DWORD *)(a1 + 312) = v16;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 316);
  *v9 = a2;
  result = v9 + 1;
  *((_DWORD *)v9 + 2) = 0;
LABEL_4:
  *result = v5;
  return result;
}
