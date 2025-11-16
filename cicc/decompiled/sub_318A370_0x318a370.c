// Function: sub_318A370
// Address: 0x318a370
//
_QWORD *__fastcall sub_318A370(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r11d
  __int64 v7; // rcx
  _QWORD *v8; // r14
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r9
  _QWORD *result; // rax
  int v13; // eax
  int v14; // edx
  unsigned __int64 v15; // rdi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rsi
  int v21; // r9d
  _QWORD *v22; // r8
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  _QWORD *v26; // rdi
  unsigned int v27; // r13d
  int v28; // r8d
  __int64 v29; // rcx

  v4 = a1 + 120;
  v5 = *(_DWORD *)(a1 + 144);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_22;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 128);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return (_QWORD *)v10[1];
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return (_QWORD *)v10[1];
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 136);
  ++*(_QWORD *)(a1 + 120);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_22:
    sub_318A170(v4, 2 * v5);
    v16 = *(_DWORD *)(a1 + 144);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 128);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 136) + 1;
      v8 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v8;
      if ( a2 != *v8 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -4096 )
        {
          if ( !v22 && v20 == -8192 )
            v22 = v8;
          v19 = v17 & (v21 + v19);
          v8 = (_QWORD *)(v18 + 16LL * v19);
          v20 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v21;
        }
        if ( v22 )
          v8 = v22;
      }
      goto LABEL_15;
    }
    goto LABEL_45;
  }
  if ( v5 - *(_DWORD *)(a1 + 140) - v14 <= v5 >> 3 )
  {
    sub_318A170(v4, v5);
    v23 = *(_DWORD *)(a1 + 144);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 128);
      v26 = 0;
      v27 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = 1;
      v14 = *(_DWORD *)(a1 + 136) + 1;
      v8 = (_QWORD *)(v25 + 16LL * v27);
      v29 = *v8;
      if ( a2 != *v8 )
      {
        while ( v29 != -4096 )
        {
          if ( !v26 && v29 == -8192 )
            v26 = v8;
          v27 = v24 & (v28 + v27);
          v8 = (_QWORD *)(v25 + 16LL * v27);
          v29 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v28;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_15;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 136);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 136) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 140);
  *v8 = a2;
  v8[1] = 0;
  result = (_QWORD *)sub_22077B0(0x10u);
  if ( result )
  {
    *result = a2;
    result[1] = a1;
  }
  v15 = v8[1];
  v8[1] = result;
  if ( v15 )
  {
    j_j___libc_free_0(v15);
    return (_QWORD *)v8[1];
  }
  return result;
}
