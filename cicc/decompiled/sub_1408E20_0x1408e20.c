// Function: sub_1408E20
// Address: 0x1408e20
//
__int64 *__fastcall sub_1408E20(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 *v7; // r8
  __int64 v8; // rdx
  int v10; // r11d
  __int64 *v11; // r10
  int v12; // eax
  int v13; // edx
  __int64 *v14; // rax
  int v15; // eax
  int v16; // ecx
  __int64 v17; // r9
  unsigned int v18; // eax
  __int64 v19; // rdi
  int v20; // r11d
  __int64 *v21; // r10
  int v22; // eax
  int v23; // ecx
  __int64 v24; // r9
  int v25; // r11d
  unsigned int v26; // eax
  __int64 v27; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v7 = (__int64 *)(v5 + 168LL * v6);
  v8 = *v7;
  if ( *a2 == *v7 )
    return v7;
  v10 = 1;
  v11 = 0;
  while ( v8 != -8 )
  {
    if ( v8 == -16 && !v11 )
      v11 = v7;
    v6 = (v4 - 1) & (v10 + v6);
    v7 = (__int64 *)(v5 + 168LL * v6);
    v8 = *v7;
    if ( *a2 == *v7 )
      return v7;
    ++v10;
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( v11 )
    v7 = v11;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
  {
LABEL_18:
    sub_1408330(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v15 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (__int64 *)(v17 + 168LL * v18);
      v19 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v7 )
        goto LABEL_10;
      v20 = 1;
      v21 = 0;
      while ( v19 != -8 )
      {
        if ( !v21 && v19 == -16 )
          v21 = v7;
        v18 = v16 & (v20 + v18);
        v7 = (__int64 *)(v17 + 168LL * v18);
        v19 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_10;
        ++v20;
      }
LABEL_22:
      if ( v21 )
        v7 = v21;
      goto LABEL_10;
    }
LABEL_43:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_1408330(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v25 = 1;
      v26 = (v22 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (__int64 *)(v24 + 168LL * v26);
      v27 = *v7;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == *a2 )
        goto LABEL_10;
      while ( v27 != -8 )
      {
        if ( v27 == -16 && !v21 )
          v21 = v7;
        v26 = v23 & (v25 + v26);
        v7 = (__int64 *)(v24 + 168LL * v26);
        v27 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_10;
        ++v25;
      }
      goto LABEL_22;
    }
    goto LABEL_43;
  }
LABEL_10:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  memset(v7 + 1, 0, 0xA0u);
  *((_BYTE *)v7 + 16) = 1;
  v14 = v7 + 3;
  do
  {
    if ( v14 )
    {
      *v14 = -2;
      v14[1] = -8;
    }
    v14 += 2;
  }
  while ( v7 + 11 != v14 );
  v7[11] = (__int64)(v7 + 13);
  v7[12] = 0x400000000LL;
  return v7;
}
