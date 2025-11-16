// Function: sub_B30A70
// Address: 0xb30a70
//
const char *__fastcall sub_B30A70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // esi
  __int64 v8; // r9
  int v9; // r11d
  __int64 v10; // r8
  _QWORD *v11; // rdx
  unsigned int v12; // r13d
  unsigned int v13; // edi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // eax
  int v20; // ecx
  __int64 v21; // rdi
  int v22; // eax
  int v23; // eax
  int v24; // eax
  int v25; // r9d
  _QWORD *v26; // r8
  __int64 v27; // rdi
  unsigned int v28; // r13d
  __int64 v29; // rsi
  int v30; // r10d
  _QWORD *v31; // r9

  if ( *(char *)(a1 + 33) >= 0 )
    return byte_3F871B3;
  v5 = sub_BD5C60(a1, a2, a3);
  v6 = *(_QWORD *)v5;
  v7 = *(_DWORD *)(*(_QWORD *)v5 + 3344LL);
  v8 = *(_QWORD *)v5 + 3320LL;
  if ( !v7 )
  {
    ++*(_QWORD *)(v6 + 3320);
    goto LABEL_7;
  }
  v9 = 1;
  v10 = *(_QWORD *)(v6 + 3328);
  v11 = 0;
  v12 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v13 = (v7 - 1) & v12;
  v14 = (_QWORD *)(v10 + 24LL * ((v7 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4))));
  v15 = *v14;
  if ( a1 == *v14 )
    return (const char *)v14[1];
  while ( v15 != -4096 )
  {
    if ( !v11 && v15 == -8192 )
      v11 = v14;
    v13 = (v7 - 1) & (v9 + v13);
    v14 = (_QWORD *)(v10 + 24LL * v13);
    v15 = *v14;
    if ( a1 == *v14 )
      return (const char *)v14[1];
    ++v9;
  }
  if ( !v11 )
    v11 = v14;
  v22 = *(_DWORD *)(v6 + 3336);
  ++*(_QWORD *)(v6 + 3320);
  v20 = v22 + 1;
  if ( 4 * (v22 + 1) >= 3 * v7 )
  {
LABEL_7:
    sub_B30870(v8, 2 * v7);
    v16 = *(_DWORD *)(v6 + 3344);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v6 + 3328);
      v19 = (v16 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v20 = *(_DWORD *)(v6 + 3336) + 1;
      v11 = (_QWORD *)(v18 + 24LL * v19);
      v21 = *v11;
      if ( a1 != *v11 )
      {
        v30 = 1;
        v31 = 0;
        while ( v21 != -4096 )
        {
          if ( !v31 && v21 == -8192 )
            v31 = v11;
          v19 = v17 & (v30 + v19);
          v11 = (_QWORD *)(v18 + 24LL * v19);
          v21 = *v11;
          if ( a1 == *v11 )
            goto LABEL_9;
          ++v30;
        }
        if ( v31 )
          v11 = v31;
      }
      goto LABEL_9;
    }
    goto LABEL_43;
  }
  if ( v7 - *(_DWORD *)(v6 + 3340) - v20 <= v7 >> 3 )
  {
    sub_B30870(v8, v7);
    v23 = *(_DWORD *)(v6 + 3344);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = 1;
      v26 = 0;
      v27 = *(_QWORD *)(v6 + 3328);
      v28 = v24 & v12;
      v20 = *(_DWORD *)(v6 + 3336) + 1;
      v11 = (_QWORD *)(v27 + 24LL * v28);
      v29 = *v11;
      if ( a1 != *v11 )
      {
        while ( v29 != -4096 )
        {
          if ( !v26 && v29 == -8192 )
            v26 = v11;
          v28 = v24 & (v25 + v28);
          v11 = (_QWORD *)(v27 + 24LL * v28);
          v29 = *v11;
          if ( a1 == *v11 )
            goto LABEL_9;
          ++v25;
        }
        if ( v26 )
          v11 = v26;
      }
      goto LABEL_9;
    }
LABEL_43:
    ++*(_DWORD *)(v6 + 3336);
    BUG();
  }
LABEL_9:
  *(_DWORD *)(v6 + 3336) = v20;
  if ( *v11 != -4096 )
    --*(_DWORD *)(v6 + 3340);
  *v11 = a1;
  v11[1] = 0;
  v11[2] = 0;
  return 0;
}
