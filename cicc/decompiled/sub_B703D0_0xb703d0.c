// Function: sub_B703D0
// Address: 0xb703d0
//
_QWORD *__fastcall sub_B703D0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned int v4; // esi
  __int64 v5; // r9
  int v6; // r11d
  __int64 v7; // r8
  _QWORD *v8; // rdx
  unsigned int v9; // edi
  _QWORD *v10; // rax
  __int64 v11; // rcx
  int v13; // eax
  int v14; // ecx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // rdi
  int v20; // r10d
  _QWORD *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  _QWORD *v25; // r8
  unsigned int v26; // r13d
  int v27; // r9d
  __int64 v28; // rsi

  v3 = *a1;
  v4 = *(_DWORD *)(*a1 + 3488);
  v5 = *a1 + 3464;
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 3464);
    goto LABEL_18;
  }
  v6 = 1;
  v7 = *(_QWORD *)(v3 + 3472);
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 40LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return v10 + 1;
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v4 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 40LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return v10 + 1;
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(v3 + 3480);
  ++*(_QWORD *)(v3 + 3464);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_B6FDE0(v5, 2 * v4);
    v15 = *(_DWORD *)(v3 + 3488);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(v3 + 3472);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(v3 + 3480) + 1;
      v8 = (_QWORD *)(v17 + 40LL * v18);
      v19 = *v8;
      if ( a2 != *v8 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -4096 )
        {
          if ( !v21 && v19 == -8192 )
            v21 = v8;
          v18 = v16 & (v20 + v18);
          v8 = (_QWORD *)(v17 + 40LL * v18);
          v19 = *v8;
          if ( a2 == *v8 )
            goto LABEL_14;
          ++v20;
        }
        if ( v21 )
          v8 = v21;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v4 - *(_DWORD *)(v3 + 3484) - v14 <= v4 >> 3 )
  {
    sub_B6FDE0(v5, v4);
    v22 = *(_DWORD *)(v3 + 3488);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v3 + 3472);
      v25 = 0;
      v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = 1;
      v14 = *(_DWORD *)(v3 + 3480) + 1;
      v8 = (_QWORD *)(v24 + 40LL * v26);
      v28 = *v8;
      if ( a2 != *v8 )
      {
        while ( v28 != -4096 )
        {
          if ( !v25 && v28 == -8192 )
            v25 = v8;
          v26 = v23 & (v27 + v26);
          v8 = (_QWORD *)(v24 + 40LL * v26);
          v28 = *v8;
          if ( a2 == *v8 )
            goto LABEL_14;
          ++v27;
        }
        if ( v25 )
          v8 = v25;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v3 + 3480);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v3 + 3480) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(v3 + 3484);
  *v8 = a2;
  v8[1] = v8 + 3;
  v8[2] = 0;
  *((_BYTE *)v8 + 24) = 0;
  return v8 + 1;
}
