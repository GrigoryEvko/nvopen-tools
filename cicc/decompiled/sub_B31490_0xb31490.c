// Function: sub_B31490
// Address: 0xb31490
//
_QWORD *__fastcall sub_B31490(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r14d
  __int64 v9; // r9
  _QWORD *v10; // rcx
  unsigned int v11; // r8d
  _QWORD *v12; // rax
  __int64 v13; // rdx
  int v15; // eax
  int v16; // edx
  int v17; // eax
  int v18; // edi
  __int64 v19; // r8
  unsigned int v20; // esi
  __int64 v21; // rax
  int v22; // r10d
  _QWORD *v23; // r9
  int v24; // eax
  int v25; // esi
  int v26; // r9d
  _QWORD *v27; // r8
  __int64 v28; // rdi
  unsigned int v29; // r13d
  __int64 v30; // rax

  v4 = sub_BD5C60(a1, a2, a3);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(*(_QWORD *)v4 + 3376LL);
  v7 = *(_QWORD *)v4 + 3352LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 3352);
    goto LABEL_18;
  }
  v8 = 1;
  v9 = *(_QWORD *)(v5 + 3360);
  v10 = 0;
  v11 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v12 = (_QWORD *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( a1 == *v12 )
    return v12 + 1;
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = (v6 - 1) & (v8 + v11);
    v12 = (_QWORD *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( a1 == *v12 )
      return v12 + 1;
    ++v8;
  }
  if ( !v10 )
    v10 = v12;
  v15 = *(_DWORD *)(v5 + 3368);
  ++*(_QWORD *)(v5 + 3352);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v6 )
  {
LABEL_18:
    sub_B31010(v7, 2 * v6);
    v17 = *(_DWORD *)(v5 + 3376);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v5 + 3360);
      v20 = (v17 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v16 = *(_DWORD *)(v5 + 3368) + 1;
      v10 = (_QWORD *)(v19 + 16LL * v20);
      v21 = *v10;
      if ( a1 != *v10 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( !v23 && v21 == -8192 )
            v23 = v10;
          v20 = v18 & (v22 + v20);
          v10 = (_QWORD *)(v19 + 16LL * v20);
          v21 = *v10;
          if ( a1 == *v10 )
            goto LABEL_14;
          ++v22;
        }
        if ( v23 )
          v10 = v23;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v6 - *(_DWORD *)(v5 + 3372) - v16 <= v6 >> 3 )
  {
    sub_B31010(v7, v6);
    v24 = *(_DWORD *)(v5 + 3376);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = 1;
      v27 = 0;
      v28 = *(_QWORD *)(v5 + 3360);
      v29 = (v24 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v16 = *(_DWORD *)(v5 + 3368) + 1;
      v10 = (_QWORD *)(v28 + 16LL * v29);
      v30 = *v10;
      if ( a1 != *v10 )
      {
        while ( v30 != -4096 )
        {
          if ( !v27 && v30 == -8192 )
            v27 = v10;
          v29 = v25 & (v26 + v29);
          v10 = (_QWORD *)(v28 + 16LL * v29);
          v30 = *v10;
          if ( a1 == *v10 )
            goto LABEL_14;
          ++v26;
        }
        if ( v27 )
          v10 = v27;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v5 + 3368);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v5 + 3368) = v16;
  if ( *v10 != -4096 )
    --*(_DWORD *)(v5 + 3372);
  *((_BYTE *)v10 + 8) &= 0xF0u;
  *v10 = a1;
  return v10 + 1;
}
