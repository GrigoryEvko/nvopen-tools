// Function: sub_9EB470
// Address: 0x9eb470
//
_QWORD *__fastcall sub_9EB470(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r12
  __int64 v7; // r8
  int v8; // r10d
  __int64 *v9; // rdi
  unsigned __int64 v10; // r13
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rdi
  __int64 v16; // rax
  int v17; // edx
  int v18; // ecx
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 v21; // rsi
  int v22; // edx
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // eax
  _QWORD *v26; // rax
  int v27; // eax
  int v28; // ecx
  __int64 v29; // r8
  int v30; // r10d
  unsigned int v31; // eax
  __int64 v32; // rsi
  _QWORD v33[6]; // [rsp+0h] [rbp-30h] BYREF

  v33[0] = a2;
  v33[1] = a3;
  if ( !a3 || *a2 != 1 )
  {
    v4 = sub_B2F650(a2, a3);
    v5 = *(_DWORD *)(a1 + 24);
    v6 = v4;
    if ( v5 )
      goto LABEL_4;
LABEL_8:
    ++*(_QWORD *)a1;
    goto LABEL_9;
  }
  v16 = sub_B2F650(a2 + 1, a3 - 1);
  v5 = *(_DWORD *)(a1 + 24);
  v6 = v16;
  if ( !v5 )
    goto LABEL_8;
LABEL_4:
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (0xBF58476D1CE4E5B9LL * v6) ^ ((0xBF58476D1CE4E5B9LL * v6) >> 31);
  v11 = v10 & (v5 - 1);
  v12 = (__int64 *)(v7 + 56LL * v11);
  v13 = *v12;
  if ( v6 == *v12 )
  {
LABEL_5:
    v14 = v12 + 1;
    return sub_9D35B0(v14, (__int64)v33);
  }
  while ( v13 != -1 )
  {
    if ( v13 == -2 && !v9 )
      v9 = v12;
    v11 = (v5 - 1) & (v8 + v11);
    v12 = (__int64 *)(v7 + 56LL * v11);
    v13 = *v12;
    if ( v6 == *v12 )
      goto LABEL_5;
    ++v8;
  }
  if ( !v9 )
    v9 = v12;
  v25 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v22 = v25 + 1;
  if ( 4 * (v25 + 1) < 3 * v5 )
  {
    if ( v5 - *(_DWORD *)(a1 + 20) - v22 > v5 >> 3 )
      goto LABEL_26;
    sub_9EB160(a1, v5);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 8);
      v30 = 1;
      v31 = v28 & v10;
      v24 = 0;
      v9 = (__int64 *)(v29 + 56LL * (v28 & (unsigned int)v10));
      v32 = *v9;
      v22 = *(_DWORD *)(a1 + 16) + 1;
      if ( v6 == *v9 )
        goto LABEL_26;
      while ( v32 != -1 )
      {
        if ( v32 == -2 && !v24 )
          v24 = v9;
        v31 = v28 & (v30 + v31);
        v9 = (__int64 *)(v29 + 56LL * v31);
        v32 = *v9;
        if ( v6 == *v9 )
          goto LABEL_26;
        ++v30;
      }
LABEL_13:
      if ( v24 )
        v9 = v24;
      goto LABEL_26;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_9:
  sub_9EB160(a1, 2 * v5);
  v17 = *(_DWORD *)(a1 + 24);
  if ( !v17 )
    goto LABEL_42;
  v18 = v17 - 1;
  v19 = *(_QWORD *)(a1 + 8);
  v20 = (v17 - 1) & (((0xBF58476D1CE4E5B9LL * v6) >> 31) ^ (484763065 * v6));
  v9 = (__int64 *)(v19 + 56LL * v20);
  v21 = *v9;
  v22 = *(_DWORD *)(a1 + 16) + 1;
  if ( v6 != *v9 )
  {
    v23 = 1;
    v24 = 0;
    while ( v21 != -1 )
    {
      if ( v21 == -2 && !v24 )
        v24 = v9;
      v20 = v18 & (v23 + v20);
      v9 = (__int64 *)(v19 + 56LL * v20);
      v21 = *v9;
      if ( v6 == *v9 )
        goto LABEL_26;
      ++v23;
    }
    goto LABEL_13;
  }
LABEL_26:
  *(_DWORD *)(a1 + 16) = v22;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 20);
  v26 = v9 + 2;
  *v9 = v6;
  v14 = v9 + 1;
  *((_DWORD *)v14 + 2) = 0;
  v14[2] = 0;
  v14[3] = v26;
  v14[4] = v26;
  v14[5] = 0;
  return sub_9D35B0(v14, (__int64)v33);
}
