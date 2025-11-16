// Function: sub_B31D10
// Address: 0xb31d10
//
__int64 __fastcall sub_B31D10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // r9
  int v8; // r11d
  __int64 v9; // r8
  _QWORD *v10; // rdx
  unsigned int v11; // r13d
  unsigned int v12; // edi
  _QWORD *v13; // rax
  __int64 v14; // rcx
  int v16; // eax
  int v17; // ecx
  int v18; // eax
  int v19; // eax
  __int64 v20; // r8
  unsigned int v21; // esi
  __int64 v22; // rdi
  int v23; // r10d
  _QWORD *v24; // r9
  int v25; // eax
  int v26; // eax
  int v27; // r9d
  _QWORD *v28; // r8
  __int64 v29; // rdi
  unsigned int v30; // r13d
  __int64 v31; // rsi

  v4 = sub_BD5C60(a1, a2, a3);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(*(_QWORD *)v4 + 3312LL);
  v7 = *(_QWORD *)v4 + 3288LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 3288);
    goto LABEL_18;
  }
  v8 = 1;
  v9 = *(_QWORD *)(v5 + 3296);
  v10 = 0;
  v11 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v12 = (v6 - 1) & v11;
  v13 = (_QWORD *)(v9 + 24LL * ((v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4))));
  v14 = *v13;
  if ( a1 == *v13 )
    return v13[1];
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = (v6 - 1) & (v8 + v12);
    v13 = (_QWORD *)(v9 + 24LL * v12);
    v14 = *v13;
    if ( a1 == *v13 )
      return v13[1];
    ++v8;
  }
  if ( !v10 )
    v10 = v13;
  v16 = *(_DWORD *)(v5 + 3304);
  ++*(_QWORD *)(v5 + 3288);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v6 )
  {
LABEL_18:
    sub_B31800(v7, 2 * v6);
    v18 = *(_DWORD *)(v5 + 3312);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v5 + 3296);
      v21 = v19 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v17 = *(_DWORD *)(v5 + 3304) + 1;
      v10 = (_QWORD *)(v20 + 24LL * v21);
      v22 = *v10;
      if ( a1 != *v10 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( !v24 && v22 == -8192 )
            v24 = v10;
          v21 = v19 & (v23 + v21);
          v10 = (_QWORD *)(v20 + 24LL * v21);
          v22 = *v10;
          if ( a1 == *v10 )
            goto LABEL_14;
          ++v23;
        }
        if ( v24 )
          v10 = v24;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v6 - *(_DWORD *)(v5 + 3308) - v17 <= v6 >> 3 )
  {
    sub_B31800(v7, v6);
    v25 = *(_DWORD *)(v5 + 3312);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = 1;
      v28 = 0;
      v29 = *(_QWORD *)(v5 + 3296);
      v30 = v26 & v11;
      v17 = *(_DWORD *)(v5 + 3304) + 1;
      v10 = (_QWORD *)(v29 + 24LL * v30);
      v31 = *v10;
      if ( a1 != *v10 )
      {
        while ( v31 != -4096 )
        {
          if ( !v28 && v31 == -8192 )
            v28 = v10;
          v30 = v26 & (v27 + v30);
          v10 = (_QWORD *)(v29 + 24LL * v30);
          v31 = *v10;
          if ( a1 == *v10 )
            goto LABEL_14;
          ++v27;
        }
        if ( v28 )
          v10 = v28;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(v5 + 3304);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v5 + 3304) = v17;
  if ( *v10 != -4096 )
    --*(_DWORD *)(v5 + 3308);
  *v10 = a1;
  v10[1] = 0;
  v10[2] = 0;
  return v10[1];
}
