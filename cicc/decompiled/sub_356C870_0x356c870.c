// Function: sub_356C870
// Address: 0x356c870
//
_QWORD *__fastcall sub_356C870(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  unsigned int v7; // esi
  __int64 v8; // rdi
  int v9; // r15d
  _QWORD *v10; // r10
  unsigned int v11; // ecx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *result; // rax
  int v15; // eax
  int v16; // edx
  int v17; // eax
  int v18; // esi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 v21; // rax
  int v22; // r9d
  _QWORD *v23; // r8
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rsi
  int v27; // r8d
  unsigned int v28; // r14d
  _QWORD *v29; // rdi
  __int64 v30; // rax

  v3 = a1 + 40;
  v7 = *(_DWORD *)(a1 + 64);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_19;
  }
  v8 = *(_QWORD *)(a1 + 48);
  v9 = 1;
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (_QWORD *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a2 )
  {
LABEL_3:
    result = v12 + 1;
    goto LABEL_4;
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = (v7 - 1) & (v9 + v11);
    v12 = (_QWORD *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a2 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v15 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v7 )
  {
LABEL_19:
    sub_356C3A0(v3, 2 * v7);
    v17 = *(_DWORD *)(a1 + 64);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 48);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(a1 + 56) + 1;
      v10 = (_QWORD *)(v19 + 16LL * v20);
      v21 = *v10;
      if ( *v10 != a2 )
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
          if ( *v10 == a2 )
            goto LABEL_15;
          ++v22;
        }
        if ( v23 )
          v10 = v23;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  if ( v7 - *(_DWORD *)(a1 + 60) - v16 <= v7 >> 3 )
  {
    sub_356C3A0(v3, v7);
    v24 = *(_DWORD *)(a1 + 64);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 48);
      v27 = 1;
      v28 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(a1 + 56) + 1;
      v29 = 0;
      v10 = (_QWORD *)(v26 + 16LL * v28);
      v30 = *v10;
      if ( *v10 != a2 )
      {
        while ( v30 != -4096 )
        {
          if ( !v29 && v30 == -8192 )
            v29 = v10;
          v28 = v25 & (v27 + v28);
          v10 = (_QWORD *)(v26 + 16LL * v28);
          v30 = *v10;
          if ( *v10 == a2 )
            goto LABEL_15;
          ++v27;
        }
        if ( v29 )
          v10 = v29;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 56) = v16;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 60);
  *v10 = a2;
  result = v10 + 1;
  v10[1] = 0;
LABEL_4:
  *result = a3;
  return result;
}
