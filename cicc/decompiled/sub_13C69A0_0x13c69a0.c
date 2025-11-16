// Function: sub_13C69A0
// Address: 0x13c69a0
//
__int64 (__fastcall *__fastcall sub_13C69A0(__int64 a1, __int64 a2))(__int64 a1)
{
  __int64 v2; // r8
  __int32 v4; // eax
  unsigned int v5; // esi
  __int32 v6; // ecx
  __int64 v7; // r10
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  _BYTE *v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rcx
  __int32 v15; // edx
  __int64 (__fastcall *result)(__int64); // rax
  __m128i *v17; // rsi
  int v18; // r13d
  __int64 *v19; // r12
  int v20; // ecx
  int v21; // ecx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // edx
  __int64 v26; // r8
  int v27; // r11d
  __int64 *v28; // r10
  int v29; // eax
  int v30; // esi
  __int64 v31; // r9
  __int64 *v32; // r10
  int v33; // r11d
  unsigned int v34; // edx
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
    goto LABEL_23;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = v35;
  v9 = (v5 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v35 == *v10 )
    goto LABEL_3;
  v18 = 1;
  v19 = 0;
  while ( v11 != -8 )
  {
    if ( !v19 && v11 == -16 )
      v19 = v10;
    v9 = (v5 - 1) & (v18 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v35 == *v10 )
      goto LABEL_3;
    ++v18;
  }
  v20 = *(_DWORD *)(a1 + 24);
  if ( v19 )
    v10 = v19;
  ++*(_QWORD *)(a1 + 8);
  v21 = v20 + 1;
  if ( 4 * v21 >= 3 * v5 )
  {
LABEL_23:
    sub_13C67E0(v2, 2 * v5);
    v22 = *(_DWORD *)(a1 + 32);
    if ( v22 )
    {
      v8 = v35;
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 16);
      v21 = *(_DWORD *)(a1 + 24) + 1;
      v25 = (v22 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v10 = (__int64 *)(v24 + 16LL * v25);
      v26 = *v10;
      if ( *v10 != v35 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -8 )
        {
          if ( !v28 && v26 == -16 )
            v28 = v10;
          v25 = v23 & (v27 + v25);
          v10 = (__int64 *)(v24 + 16LL * v25);
          v26 = *v10;
          if ( v35 == *v10 )
            goto LABEL_19;
          ++v27;
        }
        if ( v28 )
          v10 = v28;
      }
      goto LABEL_19;
    }
    goto LABEL_51;
  }
  if ( v5 - *(_DWORD *)(a1 + 28) - v21 <= v5 >> 3 )
  {
    sub_13C67E0(v2, v5);
    v29 = *(_DWORD *)(a1 + 32);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 16);
      v32 = 0;
      v33 = 1;
      v21 = *(_DWORD *)(a1 + 24) + 1;
      v34 = (v29 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v10 = (__int64 *)(v31 + 16LL * v34);
      v8 = *v10;
      if ( v35 != *v10 )
      {
        while ( v8 != -8 )
        {
          if ( !v32 && v8 == -16 )
            v32 = v10;
          v34 = v30 & (v33 + v34);
          v10 = (__int64 *)(v31 + 16LL * v34);
          v8 = *v10;
          if ( v35 == *v10 )
            goto LABEL_19;
          ++v33;
        }
        v8 = v35;
        if ( v32 )
          v10 = v32;
      }
      goto LABEL_19;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_19:
  *(_DWORD *)(a1 + 24) = v21;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 28);
  *v10 = v8;
  *((_DWORD *)v10 + 2) = 0;
  v6 = *(_DWORD *)a1;
LABEL_3:
  *((_DWORD *)v10 + 2) = v6;
  v12 = *(_BYTE **)(a1 + 48);
  if ( v12 == *(_BYTE **)(a1 + 56) )
  {
    sub_13C4280(a1 + 40, v12, &v35);
    v13 = v35;
  }
  else
  {
    v13 = v35;
    if ( v12 )
    {
      *(_QWORD *)v12 = v35;
      v12 = *(_BYTE **)(a1 + 48);
      v13 = v35;
    }
    *(_QWORD *)(a1 + 48) = v12 + 8;
  }
  v14 = *(_QWORD *)(v13 + 8);
  v15 = *(_DWORD *)a1;
  v36.m128i_i64[0] = v13;
  result = sub_13995B0;
  v17 = *(__m128i **)(a1 + 96);
  v36.m128i_i64[1] = v14;
  v37.m128i_i64[0] = (__int64)sub_13995B0;
  v37.m128i_i32[2] = v15;
  if ( v17 == *(__m128i **)(a1 + 104) )
    return (__int64 (__fastcall *)(__int64))sub_13C6650((const __m128i **)(a1 + 88), v17, &v36);
  if ( v17 )
  {
    *v17 = _mm_loadu_si128(&v36);
    v17[1] = _mm_loadu_si128(&v37);
    v17 = *(__m128i **)(a1 + 96);
  }
  *(_QWORD *)(a1 + 96) = v17 + 2;
  return result;
}
