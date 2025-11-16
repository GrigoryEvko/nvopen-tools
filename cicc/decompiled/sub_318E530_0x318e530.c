// Function: sub_318E530
// Address: 0x318e530
//
_QWORD *__fastcall sub_318E530(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rdi
  unsigned int v4; // esi
  int v5; // r11d
  __int64 v6; // rcx
  __int64 *v7; // r14
  unsigned int v8; // r13d
  unsigned int v9; // edx
  __int64 *v10; // rax
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
  __int64 *v22; // r8
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  __int64 *v26; // rdi
  unsigned int v27; // r13d
  int v28; // r8d
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 savedregs; // [rsp+10h] [rbp+0h] BYREF

  v30 = sub_BCB2D0(*(_QWORD **)a1);
  if ( !v30 )
    return 0;
  savedregs = (__int64)&savedregs;
  v1 = v30;
  v3 = a1 + 152;
  v4 = *(_DWORD *)(a1 + 176);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_24;
  }
  v5 = 1;
  v6 = *(_QWORD *)(a1 + 160);
  v7 = 0;
  v8 = ((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4);
  v9 = (v4 - 1) & v8;
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v1 == *v10 )
    return (_QWORD *)v10[1];
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v7 )
      v7 = v10;
    v9 = (v4 - 1) & (v5 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v1 == *v10 )
      return (_QWORD *)v10[1];
    ++v5;
  }
  if ( !v7 )
    v7 = v10;
  v13 = *(_DWORD *)(a1 + 168);
  ++*(_QWORD *)(a1 + 152);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_24:
    sub_318CD80(v3, 2 * v4);
    v16 = *(_DWORD *)(a1 + 176);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 160);
      v19 = (v16 - 1) & (((unsigned int)v1 >> 9) ^ ((unsigned int)v1 >> 4));
      v14 = *(_DWORD *)(a1 + 168) + 1;
      v7 = (__int64 *)(v18 + 16LL * v19);
      v20 = *v7;
      if ( v1 != *v7 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -4096 )
        {
          if ( !v22 && v20 == -8192 )
            v22 = v7;
          v19 = v17 & (v21 + v19);
          v7 = (__int64 *)(v18 + 16LL * v19);
          v20 = *v7;
          if ( v1 == *v7 )
            goto LABEL_16;
          ++v21;
        }
        if ( v22 )
          v7 = v22;
      }
      goto LABEL_16;
    }
    goto LABEL_47;
  }
  if ( v4 - *(_DWORD *)(a1 + 172) - v14 <= v4 >> 3 )
  {
    sub_318CD80(v3, v4);
    v23 = *(_DWORD *)(a1 + 176);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 160);
      v26 = 0;
      v27 = v24 & v8;
      v28 = 1;
      v14 = *(_DWORD *)(a1 + 168) + 1;
      v7 = (__int64 *)(v25 + 16LL * v27);
      v29 = *v7;
      if ( v1 != *v7 )
      {
        while ( v29 != -4096 )
        {
          if ( !v26 && v29 == -8192 )
            v26 = v7;
          v27 = v24 & (v28 + v27);
          v7 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v7;
          if ( v1 == *v7 )
            goto LABEL_16;
          ++v28;
        }
        if ( v26 )
          v7 = v26;
      }
      goto LABEL_16;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 168);
    BUG();
  }
LABEL_16:
  *(_DWORD *)(a1 + 168) = v14;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 172);
  *v7 = v1;
  v7[1] = 0;
  result = (_QWORD *)sub_22077B0(0x10u);
  if ( result )
  {
    *result = v1;
    result[1] = a1;
  }
  v15 = v7[1];
  v7[1] = (__int64)result;
  if ( v15 )
  {
    j_j___libc_free_0(v15);
    return (_QWORD *)v7[1];
  }
  return result;
}
