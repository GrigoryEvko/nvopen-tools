// Function: sub_FF8520
// Address: 0xff8520
//
__int64 __fastcall sub_FF8520(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int32 v4; // eax
  unsigned int v5; // esi
  __int32 v6; // r12d
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 *v9; // r11
  int v10; // r14d
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  int v17; // ecx
  __int64 result; // rax
  __m128i *v19; // rsi
  int v20; // eax
  int v21; // edx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // ecx
  __int64 v26; // rax
  int v27; // r10d
  __int64 *v28; // r9
  int v29; // eax
  int v30; // eax
  __int64 v31; // r8
  int v32; // r10d
  unsigned int v33; // ecx
  __int64 v34; // rsi
  __int64 v35; // [rsp+8h] [rbp-48h] BYREF
  __m128i v36; // [rsp+10h] [rbp-40h] BYREF
  __m128i v37; // [rsp+20h] [rbp-30h] BYREF

  v2 = a1 + 8;
  v4 = *(_DWORD *)a1;
  v35 = a2;
  v5 = *(_DWORD *)(a1 + 32);
  v6 = v4 + 1;
  *(_DWORD *)a1 = v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_32;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = v35;
  v9 = 0;
  v10 = 1;
  v11 = (v5 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v12 = (__int64 *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( v35 != *v12 )
  {
    while ( v13 != -4096 )
    {
      if ( !v9 && v13 == -8192 )
        v9 = v12;
      v11 = (v5 - 1) & (v10 + v11);
      v12 = (__int64 *)(v7 + 16LL * v11);
      v13 = *v12;
      if ( v35 == *v12 )
        goto LABEL_3;
      ++v10;
    }
    if ( !v9 )
      v9 = v12;
    v20 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 28) - v21 > v5 >> 3 )
        goto LABEL_25;
      sub_A4A350(v2, v5);
      v29 = *(_DWORD *)(a1 + 32);
      if ( v29 )
      {
        v8 = v35;
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 16);
        v28 = 0;
        v32 = 1;
        v33 = v30 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v21 = *(_DWORD *)(a1 + 24) + 1;
        v9 = (__int64 *)(v31 + 16LL * v33);
        v34 = *v9;
        if ( *v9 == v35 )
          goto LABEL_25;
        while ( v34 != -4096 )
        {
          if ( !v28 && v34 == -8192 )
            v28 = v9;
          v33 = v30 & (v32 + v33);
          v9 = (__int64 *)(v31 + 16LL * v33);
          v34 = *v9;
          if ( v35 == *v9 )
            goto LABEL_25;
          ++v32;
        }
LABEL_36:
        if ( v28 )
          v9 = v28;
LABEL_25:
        *(_DWORD *)(a1 + 24) = v21;
        if ( *v9 != -4096 )
          --*(_DWORD *)(a1 + 28);
        *v9 = v8;
        *((_DWORD *)v9 + 2) = 0;
        *((_DWORD *)v9 + 2) = v6;
        v14 = *(_BYTE **)(a1 + 48);
        if ( v14 != *(_BYTE **)(a1 + 56) )
          goto LABEL_4;
LABEL_28:
        sub_F46430(a1 + 40, v14, &v35);
        v15 = v35;
        goto LABEL_7;
      }
      goto LABEL_53;
    }
LABEL_32:
    sub_A4A350(v2, 2 * v5);
    v22 = *(_DWORD *)(a1 + 32);
    if ( v22 )
    {
      v8 = v35;
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 16);
      v21 = *(_DWORD *)(a1 + 24) + 1;
      v25 = (v22 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v9 = (__int64 *)(v24 + 16LL * v25);
      v26 = *v9;
      if ( *v9 == v35 )
        goto LABEL_25;
      v27 = 1;
      v28 = 0;
      while ( v26 != -4096 )
      {
        if ( !v28 && v26 == -8192 )
          v28 = v9;
        v25 = v23 & (v27 + v25);
        v9 = (__int64 *)(v24 + 16LL * v25);
        v26 = *v9;
        if ( v35 == *v9 )
          goto LABEL_25;
        ++v27;
      }
      goto LABEL_36;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_3:
  *((_DWORD *)v12 + 2) = v6;
  v14 = *(_BYTE **)(a1 + 48);
  if ( v14 == *(_BYTE **)(a1 + 56) )
    goto LABEL_28;
LABEL_4:
  v15 = v35;
  if ( v14 )
  {
    *(_QWORD *)v14 = v35;
    v14 = *(_BYTE **)(a1 + 48);
  }
  *(_QWORD *)(a1 + 48) = v14 + 8;
LABEL_7:
  v16 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v16 == v15 + 48 )
  {
    result = 0;
  }
  else
  {
    if ( !v16 )
      BUG();
    v17 = *(unsigned __int8 *)(v16 - 24);
    result = v16 - 24;
    if ( (unsigned int)(v17 - 30) >= 0xB )
      result = 0;
  }
  v37.m128i_i32[2] = *(_DWORD *)a1;
  v19 = *(__m128i **)(a1 + 96);
  v36.m128i_i64[0] = v15;
  v36.m128i_i64[1] = result;
  v37.m128i_i32[0] = 0;
  if ( v19 == *(__m128i **)(a1 + 104) )
    return sub_FF8390((const __m128i **)(a1 + 88), v19, &v36);
  if ( v19 )
  {
    *v19 = _mm_loadu_si128(&v36);
    v19[1] = _mm_loadu_si128(&v37);
    v19 = *(__m128i **)(a1 + 96);
  }
  *(_QWORD *)(a1 + 96) = v19 + 2;
  return result;
}
