// Function: sub_31A52A0
// Address: 0x31a52a0
//
__int64 __fastcall sub_31A52A0(__int64 a1)
{
  __int64 v2; // r15
  unsigned int *v3; // rbx
  unsigned int *v4; // r13
  unsigned int *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  char **v8; // rdx
  char *v9; // r15
  char v10; // al
  char *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  char **v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // r12d
  char *v17; // rax
  _BYTE *v18; // rsi
  char v19; // al
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rcx
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // rsi
  _QWORD *v30; // rdx
  __int64 *v32; // rsi
  __int64 v33; // r8
  const __m128i *v34; // rbx
  __m128i *v35; // rax
  __int64 v36; // rdi
  const void *v37; // rsi
  char *v38; // rbx
  __int64 v39; // [rsp+0h] [rbp-60h]
  char *v40; // [rsp+8h] [rbp-58h]
  _QWORD v41[10]; // [rsp+10h] [rbp-50h] BYREF

  if ( !(_BYTE)qword_5034F28 )
    return 0;
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL);
  if ( !*(_BYTE *)(v2 + 232) )
    return 0;
  v3 = *(unsigned int **)(v2 + 240);
  v4 = &v3[3 * *(unsigned int *)(v2 + 248)];
  if ( v3 == v4 )
    return 0;
  v5 = 0;
  do
  {
    if ( (unsigned int)sub_D354B0(v3[2]) == 2 )
    {
      if ( v3[2] != 2 || v5 )
        return 0;
      v5 = v3;
    }
    v3 += 3;
  }
  while ( v4 != v3 );
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(v2 + 56);
  if ( **(_BYTE **)(v6 + 8LL * *v5) != 61 )
    return 0;
  v7 = *(_QWORD *)(v6 + 8LL * v5[1]);
  if ( *(_BYTE *)v7 != 62 )
    return 0;
  v8 = (*(_BYTE *)(v7 + 7) & 0x40) != 0 ? *(char ***)(v7 - 8) : (char **)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
  v9 = *v8;
  v10 = **v8;
  if ( (unsigned __int8)(v10 - 42) > 0x11u )
    return 0;
  v11 = v8[4];
  if ( (unsigned __int8)*v11 <= 0x1Cu )
    return 0;
  v12 = *(_QWORD *)a1;
  v39 = **(_QWORD **)(a1 + 56);
  if ( v10 != 42 && v10 != 44 )
    return 0;
  v13 = *((_QWORD *)v9 - 8);
  if ( *(_BYTE *)v13 != 61 )
    return 0;
  v14 = (*(_BYTE *)(v13 + 7) & 0x40) != 0
      ? *(char ***)(v13 - 8)
      : (char **)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
  if ( v11 != *v14 )
    return 0;
  v15 = *((_QWORD *)v9 - 4);
  if ( !v15 )
    return 0;
  v40 = v11;
  v16 = sub_D48480(v12, v15, (__int64)v11, **(_QWORD **)(a1 + 56));
  if ( !(_BYTE)v16 )
    return 0;
  if ( *v40 != 63 )
    return 0;
  v17 = &v40[32 * (1LL - (*((_DWORD *)v40 + 1) & 0x7FFFFFF))];
  if ( v17 == v40 )
    return 0;
  v18 = *(_BYTE **)v17;
  if ( **(_BYTE **)v17 == 17 )
  {
    while ( 1 )
    {
      v17 += 32;
      if ( v40 == v17 )
        return 0;
      v18 = *(_BYTE **)v17;
      if ( **(_BYTE **)v17 != 17 )
      {
        if ( v40 == v17 + 32 )
          goto LABEL_29;
        return 0;
      }
    }
  }
  if ( v40 != v17 + 32 )
    return 0;
LABEL_29:
  v19 = *v18;
  if ( *v18 <= 0x1Cu )
    return 0;
  if ( v19 == 68 || v19 == 69 )
  {
    v20 = *((_QWORD *)v18 - 4);
    if ( *(_BYTE *)v20 != 61 )
      return 0;
    v21 = (*(_BYTE *)(v20 + 7) & 0x40) != 0
        ? *(__int64 **)(v20 - 8)
        : (__int64 *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF));
    v22 = *v21;
    if ( !*v21 )
      return 0;
  }
  else
  {
    if ( v19 != 61 )
      return 0;
    v32 = (v18[7] & 0x40) != 0
        ? (__int64 *)*((_QWORD *)v18 - 1)
        : (__int64 *)&v18[-32 * (*((_DWORD *)v18 + 1) & 0x7FFFFFF)];
    v22 = *v32;
    if ( !v22 )
      return 0;
  }
  v23 = sub_DD8400(*(_QWORD *)(v39 + 112), v22);
  if ( *((_WORD *)v23 + 12) != 8 )
    return 0;
  if ( v12 != v23[6] )
    return 0;
  v25 = *((_QWORD *)v9 - 8);
  v26 = *(_QWORD *)(v25 + 40);
  if ( v26 != *((_QWORD *)v9 + 5) || v26 != *(_QWORD *)(v7 + 40) )
    return 0;
  v27 = *(unsigned int *)(a1 + 544);
  v28 = *(_QWORD *)(a1 + 536);
  v29 = *(unsigned int *)(a1 + 548);
  v30 = (_QWORD *)(v28 + 24 * v27);
  if ( v27 >= v29 )
  {
    v33 = v27 + 1;
    v41[2] = v7;
    v34 = (const __m128i *)v41;
    v41[0] = v25;
    v41[1] = v9;
    if ( v29 < v27 + 1 )
    {
      v36 = a1 + 536;
      v37 = (const void *)(a1 + 552);
      if ( v28 > (unsigned __int64)v41 || v30 <= v41 )
      {
        sub_C8D5F0(v36, v37, v27 + 1, 0x18u, v33, v24);
        v27 = *(unsigned int *)(a1 + 544);
        v28 = *(_QWORD *)(a1 + 536);
      }
      else
      {
        v38 = (char *)v41 - v28;
        sub_C8D5F0(v36, v37, v27 + 1, 0x18u, v33, v24);
        v28 = *(_QWORD *)(a1 + 536);
        v27 = *(unsigned int *)(a1 + 544);
        v34 = (const __m128i *)&v38[v28];
      }
    }
    v35 = (__m128i *)(v28 + 24 * v27);
    *v35 = _mm_loadu_si128(v34);
    v35[1].m128i_i64[0] = v34[1].m128i_i64[0];
    ++*(_DWORD *)(a1 + 544);
  }
  else
  {
    if ( v30 )
    {
      *v30 = v25;
      v30[1] = v9;
      v30[2] = v7;
    }
    ++*(_DWORD *)(a1 + 544);
  }
  return v16;
}
