// Function: sub_B311F0
// Address: 0xb311f0
//
_BYTE *__fastcall sub_B311F0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // rdi
  int v10; // r15d
  _QWORD *v11; // r10
  unsigned int v12; // ecx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _BYTE *result; // rax
  int v16; // eax
  int v17; // edx
  int v18; // eax
  int v19; // esi
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // rax
  int v23; // r9d
  _QWORD *v24; // r8
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rsi
  int v28; // r8d
  _QWORD *v29; // rdi
  unsigned int v30; // r14d
  __int64 v31; // rax

  v3 = a2;
  v5 = sub_BD5C60(a1, a2, a3);
  v6 = *(_QWORD *)v5;
  v7 = *(_DWORD *)(*(_QWORD *)v5 + 3376LL);
  v8 = *(_QWORD *)v5 + 3352LL;
  if ( !v7 )
  {
    ++*(_QWORD *)(v6 + 3352);
    goto LABEL_19;
  }
  v9 = *(_QWORD *)(v6 + 3360);
  v10 = 1;
  v11 = 0;
  v12 = (v7 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v13 = (_QWORD *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( a1 == *v13 )
  {
LABEL_3:
    result = v13 + 1;
    goto LABEL_4;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v7 - 1) & (v10 + v12);
    v13 = (_QWORD *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( a1 == *v13 )
      goto LABEL_3;
    ++v10;
  }
  if ( !v11 )
    v11 = v13;
  v16 = *(_DWORD *)(v6 + 3368);
  ++*(_QWORD *)(v6 + 3352);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v7 )
  {
LABEL_19:
    sub_B31010(v8, 2 * v7);
    v18 = *(_DWORD *)(v6 + 3376);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v6 + 3360);
      v21 = (v18 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v17 = *(_DWORD *)(v6 + 3368) + 1;
      v11 = (_QWORD *)(v20 + 16LL * v21);
      v22 = *v11;
      if ( a1 != *v11 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( !v24 && v22 == -8192 )
            v24 = v11;
          v21 = v19 & (v23 + v21);
          v11 = (_QWORD *)(v20 + 16LL * v21);
          v22 = *v11;
          if ( a1 == *v11 )
            goto LABEL_15;
          ++v23;
        }
        if ( v24 )
          v11 = v24;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  if ( v7 - *(_DWORD *)(v6 + 3372) - v17 <= v7 >> 3 )
  {
    sub_B31010(v8, v7);
    v25 = *(_DWORD *)(v6 + 3376);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v6 + 3360);
      v28 = 1;
      v29 = 0;
      v30 = (v25 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v17 = *(_DWORD *)(v6 + 3368) + 1;
      v11 = (_QWORD *)(v27 + 16LL * v30);
      v31 = *v11;
      if ( a1 != *v11 )
      {
        while ( v31 != -4096 )
        {
          if ( !v29 && v31 == -8192 )
            v29 = v11;
          v30 = v26 & (v28 + v30);
          v11 = (_QWORD *)(v27 + 16LL * v30);
          v31 = *v11;
          if ( a1 == *v11 )
            goto LABEL_15;
          ++v28;
        }
        if ( v29 )
          v11 = v29;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(v6 + 3368);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(v6 + 3368) = v17;
  if ( *v11 != -4096 )
    --*(_DWORD *)(v6 + 3372);
  *((_BYTE *)v11 + 8) &= 0xF0u;
  result = v11 + 1;
  *v11 = a1;
LABEL_4:
  *result = v3;
  *(_BYTE *)(a1 + 34) |= 1u;
  return result;
}
