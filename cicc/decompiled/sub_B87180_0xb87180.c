// Function: sub_B87180
// Address: 0xb87180
//
_QWORD *__fastcall sub_B87180(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v5; // r13
  unsigned int v6; // esi
  int v7; // r15d
  __int64 v8; // rdi
  _QWORD *v9; // r10
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *result; // rax
  int v14; // eax
  int v15; // edx
  int v16; // eax
  int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 v20; // rax
  int v21; // r9d
  _QWORD *v22; // r8
  int v23; // eax
  int v24; // ecx
  __int64 v25; // rsi
  _QWORD *v26; // rdi
  unsigned int v27; // r14d
  int v28; // r8d
  __int64 v29; // rax

  v2 = a1 + 208;
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(_DWORD *)(a1 + 232);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 208);
    goto LABEL_19;
  }
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 216);
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v11 = (_QWORD *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( v5 == *v11 )
  {
LABEL_3:
    result = v11 + 1;
    goto LABEL_4;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = (v6 - 1) & (v7 + v10);
    v11 = (_QWORD *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v5 == *v11 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v9 )
    v9 = v11;
  v14 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v6 )
  {
LABEL_19:
    sub_B86FA0(v2, 2 * v6);
    v16 = *(_DWORD *)(a1 + 232);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 216);
      v19 = (v16 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v15 = *(_DWORD *)(a1 + 224) + 1;
      v9 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v9;
      if ( v5 != *v9 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -4096 )
        {
          if ( !v22 && v20 == -8192 )
            v22 = v9;
          v19 = v17 & (v21 + v19);
          v9 = (_QWORD *)(v18 + 16LL * v19);
          v20 = *v9;
          if ( v5 == *v9 )
            goto LABEL_15;
          ++v21;
        }
        if ( v22 )
          v9 = v22;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  if ( v6 - *(_DWORD *)(a1 + 228) - v15 <= v6 >> 3 )
  {
    sub_B86FA0(v2, v6);
    v23 = *(_DWORD *)(a1 + 232);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 216);
      v26 = 0;
      v27 = (v23 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v28 = 1;
      v15 = *(_DWORD *)(a1 + 224) + 1;
      v9 = (_QWORD *)(v25 + 16LL * v27);
      v29 = *v9;
      if ( v5 != *v9 )
      {
        while ( v29 != -4096 )
        {
          if ( !v26 && v29 == -8192 )
            v26 = v9;
          v27 = v24 & (v28 + v27);
          v9 = (_QWORD *)(v25 + 16LL * v27);
          v29 = *v9;
          if ( v5 == *v9 )
            goto LABEL_15;
          ++v28;
        }
        if ( v26 )
          v9 = v26;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 224);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 224) = v15;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 228);
  *v9 = v5;
  result = v9 + 1;
  v9[1] = 0;
LABEL_4:
  *result = a2;
  return result;
}
