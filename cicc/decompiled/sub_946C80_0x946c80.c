// Function: sub_946C80
// Address: 0x946c80
//
__int64 __fastcall sub_946C80(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v5; // esi
  int v6; // r14d
  __int64 v7; // r8
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // r10
  __int64 *v12; // r13
  __int64 result; // rax
  int v14; // eax
  int v15; // ecx
  const char *v16; // rsi
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rdi
  int v22; // r10d
  _QWORD *v23; // r9
  int v24; // eax
  int v25; // eax
  int v26; // r9d
  _QWORD *v27; // r8
  __int64 v28; // rdi
  unsigned int v29; // r13d
  __int64 v30; // rsi

  v3 = a1 + 464;
  v5 = *(_DWORD *)(a1 + 488);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 464);
    goto LABEL_22;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 472);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
  {
LABEL_3:
    v12 = v10 + 1;
    result = v10[1];
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 480);
  ++*(_QWORD *)(a1 + 464);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_22:
    sub_946AA0(v3, 2 * v5);
    v17 = *(_DWORD *)(a1 + 488);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 472);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 480) + 1;
      v8 = (_QWORD *)(v19 + 16LL * v20);
      v21 = *v8;
      if ( *v8 != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( !v23 && v21 == -8192 )
            v23 = v8;
          v20 = v18 & (v22 + v20);
          v8 = (_QWORD *)(v19 + 16LL * v20);
          v21 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v22;
        }
        if ( v23 )
          v8 = v23;
      }
      goto LABEL_15;
    }
    goto LABEL_45;
  }
  if ( v5 - *(_DWORD *)(a1 + 484) - v15 <= v5 >> 3 )
  {
    sub_946AA0(v3, v5);
    v24 = *(_DWORD *)(a1 + 488);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = 1;
      v27 = 0;
      v28 = *(_QWORD *)(a1 + 472);
      v29 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 480) + 1;
      v8 = (_QWORD *)(v28 + 16LL * v29);
      v30 = *v8;
      if ( *v8 != a2 )
      {
        while ( v30 != -4096 )
        {
          if ( !v27 && v30 == -8192 )
            v27 = v8;
          v29 = v25 & (v26 + v29);
          v8 = (_QWORD *)(v28 + 16LL * v29);
          v30 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v26;
        }
        if ( v27 )
          v8 = v27;
      }
      goto LABEL_15;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 480);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 480) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 484);
  *v8 = a2;
  v12 = v8 + 1;
  v8[1] = 0;
LABEL_18:
  v16 = *(const char **)(*(_QWORD *)(a2 + 72) + 8LL);
  if ( !v16 )
    v16 = "compiler_generated_label";
  result = sub_945CA0(a1, (__int64)v16, 0, 0);
  *v12 = result;
  return result;
}
