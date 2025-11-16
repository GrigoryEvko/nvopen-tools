// Function: sub_2B5C350
// Address: 0x2b5c350
//
unsigned __int64 __fastcall sub_2B5C350(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned int v5; // esi
  unsigned __int64 v6; // r12
  __int64 v7; // r8
  unsigned __int64 result; // rax
  unsigned int v9; // edi
  unsigned __int64 *v10; // rcx
  unsigned __int64 v11; // rdx
  int v12; // r13d
  unsigned __int64 *v13; // r11
  int v14; // ecx
  int v15; // ecx
  int v16; // edx
  int v17; // edx
  __int64 v18; // rdi
  unsigned __int64 v19; // rsi
  int v20; // r9d
  unsigned __int64 *v21; // r8
  int v22; // edx
  int v23; // edx
  int v24; // r9d
  __int64 v25; // rdi
  unsigned __int64 v26; // rsi

  v4 = sub_27B0000(a2, (__int64)&a2[a3]);
  v5 = *(_DWORD *)(a1 + 2192);
  v6 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 2168);
    goto LABEL_14;
  }
  v7 = *(_QWORD *)(a1 + 2176);
  result = ((0xBF58476D1CE4E5B9LL * v4) >> 31) ^ (0xBF58476D1CE4E5B9LL * v4);
  v9 = result & (v5 - 1);
  v10 = (unsigned __int64 *)(v7 + 8LL * v9);
  v11 = *v10;
  if ( v6 == *v10 )
    return result;
  v12 = 1;
  v13 = 0;
  while ( v11 != -1 )
  {
    if ( v13 || v11 != -2 )
      v10 = v13;
    v9 = (v5 - 1) & (v12 + v9);
    v11 = *(_QWORD *)(v7 + 8LL * v9);
    if ( v11 == v6 )
      return result;
    ++v12;
    v13 = v10;
    v10 = (unsigned __int64 *)(v7 + 8LL * v9);
  }
  if ( !v13 )
    v13 = v10;
  v14 = *(_DWORD *)(a1 + 2184);
  ++*(_QWORD *)(a1 + 2168);
  v15 = v14 + 1;
  if ( 4 * v15 >= 3 * v5 )
  {
LABEL_14:
    sub_A32210(a1 + 2168, 2 * v5);
    v16 = *(_DWORD *)(a1 + 2192);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 2176);
      result = v17 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v6) >> 31) ^ (484763065 * (_DWORD)v6));
      v13 = (unsigned __int64 *)(v18 + 8 * result);
      v19 = *v13;
      v15 = *(_DWORD *)(a1 + 2184) + 1;
      if ( *v13 == v6 )
        goto LABEL_10;
      v20 = 1;
      v21 = 0;
      while ( v19 != -1 )
      {
        if ( v19 == -2 && !v21 )
          v21 = v13;
        result = v17 & (unsigned int)(v20 + result);
        v13 = (unsigned __int64 *)(v18 + 8LL * (unsigned int)result);
        v19 = *v13;
        if ( *v13 == v6 )
          goto LABEL_10;
        ++v20;
      }
LABEL_18:
      if ( v21 )
        v13 = v21;
      goto LABEL_10;
    }
LABEL_39:
    ++*(_DWORD *)(a1 + 2184);
    BUG();
  }
  if ( v5 - *(_DWORD *)(a1 + 2188) - v15 <= v5 >> 3 )
  {
    sub_A32210(a1 + 2168, v5);
    v22 = *(_DWORD *)(a1 + 2192);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = 1;
      v21 = 0;
      v25 = *(_QWORD *)(a1 + 2176);
      result = v23 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v6) >> 31) ^ (484763065 * (_DWORD)v6));
      v13 = (unsigned __int64 *)(v25 + 8 * result);
      v26 = *v13;
      v15 = *(_DWORD *)(a1 + 2184) + 1;
      if ( *v13 == v6 )
        goto LABEL_10;
      while ( v26 != -1 )
      {
        if ( !v21 && v26 == -2 )
          v21 = v13;
        result = v23 & (unsigned int)(v24 + result);
        v13 = (unsigned __int64 *)(v25 + 8LL * (unsigned int)result);
        v26 = *v13;
        if ( *v13 == v6 )
          goto LABEL_10;
        ++v24;
      }
      goto LABEL_18;
    }
    goto LABEL_39;
  }
LABEL_10:
  *(_DWORD *)(a1 + 2184) = v15;
  if ( *v13 != -1 )
    --*(_DWORD *)(a1 + 2188);
  *v13 = v6;
  return result;
}
