// Function: sub_1426290
// Address: 0x1426290
//
__int64 __fastcall sub_1426290(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r11d
  _QWORD *v8; // r14
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r9
  __int64 result; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // rdi
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
  int v26; // r8d
  unsigned int v27; // r13d
  _QWORD *v28; // rdi
  __int64 v29; // rcx

  v4 = a1 + 88;
  v5 = *(_DWORD *)(a1 + 112);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 88);
    goto LABEL_22;
  }
  v6 = *(_QWORD *)(a1 + 96);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return v10[1];
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return v10[1];
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 104);
  ++*(_QWORD *)(a1 + 88);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_22:
    sub_14260A0(v4, 2 * v5);
    v16 = *(_DWORD *)(a1 + 112);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 96);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 104) + 1;
      v8 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v8;
      if ( a2 != *v8 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
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
  if ( v5 - *(_DWORD *)(a1 + 108) - v14 <= v5 >> 3 )
  {
    sub_14260A0(v4, v5);
    v23 = *(_DWORD *)(a1 + 112);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 96);
      v26 = 1;
      v27 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 104) + 1;
      v28 = 0;
      v8 = (_QWORD *)(v25 + 16LL * v27);
      v29 = *v8;
      if ( a2 != *v8 )
      {
        while ( v29 != -8 )
        {
          if ( !v28 && v29 == -16 )
            v28 = v8;
          v27 = v24 & (v26 + v27);
          v8 = (_QWORD *)(v25 + 16LL * v27);
          v29 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v26;
        }
        if ( v28 )
          v8 = v28;
      }
      goto LABEL_15;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 104);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 104) = v14;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 108);
  *v8 = a2;
  v8[1] = 0;
  result = sub_22077B0(16);
  if ( result )
  {
    *(_QWORD *)(result + 8) = result;
    *(_QWORD *)result = result | 4;
  }
  v15 = v8[1];
  v8[1] = result;
  if ( v15 )
  {
    j_j___libc_free_0(v15, 16);
    return v8[1];
  }
  return result;
}
