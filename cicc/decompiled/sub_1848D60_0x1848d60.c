// Function: sub_1848D60
// Address: 0x1848d60
//
__int64 *__fastcall sub_1848D60(__int64 a1, __int64 a2)
{
  __m128i *v3; // r12
  __int64 v4; // rax
  __m128i *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // eax
  __m128i *v9; // rdx
  __m128i *i; // rbx
  __m128i v11; // xmm0
  __int64 v12; // rdx
  __int64 v13; // rax
  __m128i v14; // xmm0
  void (__fastcall *v15)(__m128i *, __m128i *, __int64); // rsi
  __m128i v16; // xmm0
  __int64 v17; // rsi
  __int64 v18; // rdx
  __m128i v19; // xmm0
  void (__fastcall *v20)(__m128i *, __m128i *, __int64); // rax
  __m128i v21; // xmm0
  __int64 v22; // rsi
  __int64 v23; // rdx
  __m128i v24; // xmm0
  void (__fastcall *v25)(__m128i *, __m128i *, __int64); // rax
  int v27; // eax
  __m128i v28; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v29)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v3 = *(__m128i **)a1;
  v4 = 104LL * *(unsigned int *)(a1 + 8);
  v5 = (__m128i *)(*(_QWORD *)a1 + v4);
  v6 = 0x4EC4EC4EC4EC4EC5LL * (v4 >> 3);
  v7 = v6 >> 2;
  if ( v6 >> 2 )
  {
    v8 = *(_DWORD *)(a2 + 96);
    v9 = &v3[26 * v7];
    while ( v3[6].m128i_i32[0] != v8 )
    {
      if ( v8 == v3[12].m128i_i32[2] )
      {
        v3 = (__m128i *)((char *)v3 + 104);
        goto LABEL_8;
      }
      if ( v8 == v3[19].m128i_i32[0] )
      {
        v3 += 13;
        goto LABEL_8;
      }
      if ( v8 == v3[25].m128i_i32[2] )
      {
        v3 = (__m128i *)((char *)v3 + 312);
        goto LABEL_8;
      }
      v3 += 26;
      if ( v9 == v3 )
      {
        v6 = 0x4EC4EC4EC4EC4EC5LL * (((char *)v5 - (char *)v3) >> 3);
        goto LABEL_21;
      }
    }
    goto LABEL_8;
  }
LABEL_21:
  if ( v6 == 2 )
  {
    v27 = *(_DWORD *)(a2 + 96);
LABEL_31:
    if ( v27 == v3[6].m128i_i32[0] )
      goto LABEL_8;
    v3 = (__m128i *)((char *)v3 + 104);
    goto LABEL_28;
  }
  if ( v6 == 3 )
  {
    v27 = *(_DWORD *)(a2 + 96);
    if ( v3[6].m128i_i32[0] == v27 )
      goto LABEL_8;
    v3 = (__m128i *)((char *)v3 + 104);
    goto LABEL_31;
  }
  if ( v6 != 1 )
  {
LABEL_24:
    v3 = v5;
    return sub_1848A70((__int64 *)a1, v3, v5);
  }
  v27 = *(_DWORD *)(a2 + 96);
LABEL_28:
  if ( v27 != v3[6].m128i_i32[0] )
    goto LABEL_24;
LABEL_8:
  if ( v5 != v3 )
  {
    for ( i = (__m128i *)((char *)v3 + 104); v5 != i; i = (__m128i *)((char *)i + 104) )
    {
      if ( i[6].m128i_i32[0] != *(_DWORD *)(a2 + 96) )
      {
        v11 = _mm_loadu_si128(i);
        *i = _mm_loadu_si128(&v28);
        v28 = v11;
        v12 = i[1].m128i_i64[0];
        v13 = i[1].m128i_i64[1];
        i[1].m128i_i64[0] = 0;
        i[1].m128i_i64[1] = v30;
        v14 = _mm_loadu_si128(&v28);
        v28 = _mm_loadu_si128(v3);
        v15 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v3[1].m128i_i64[0];
        *v3 = v14;
        v29 = v15;
        v3[1].m128i_i64[0] = v12;
        v30 = v3[1].m128i_i64[1];
        v3[1].m128i_i64[1] = v13;
        if ( v29 )
          v29(&v28, &v28, 3);
        v16 = _mm_loadu_si128(i + 2);
        i[2] = _mm_loadu_si128(&v28);
        v28 = v16;
        v17 = i[3].m128i_i64[0];
        v18 = i[3].m128i_i64[1];
        i[3].m128i_i64[0] = 0;
        i[3].m128i_i64[1] = v30;
        v19 = _mm_loadu_si128(&v28);
        v28 = _mm_loadu_si128(v3 + 2);
        v20 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v3[3].m128i_i64[0];
        v3[2] = v19;
        v29 = v20;
        v3[3].m128i_i64[0] = v17;
        v30 = v3[3].m128i_i64[1];
        v3[3].m128i_i64[1] = v18;
        if ( v20 )
          v20(&v28, &v28, 3);
        v21 = _mm_loadu_si128(i + 4);
        i[4] = _mm_loadu_si128(&v28);
        v28 = v21;
        v22 = i[5].m128i_i64[0];
        v23 = i[5].m128i_i64[1];
        i[5].m128i_i64[0] = 0;
        i[5].m128i_i64[1] = v30;
        v24 = _mm_loadu_si128(&v28);
        v28 = _mm_loadu_si128(v3 + 4);
        v25 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v3[5].m128i_i64[0];
        v3[4] = v24;
        v29 = v25;
        v3[5].m128i_i64[0] = v22;
        v30 = v3[5].m128i_i64[1];
        v3[5].m128i_i64[1] = v23;
        if ( v25 )
          v25(&v28, &v28, 3);
        v3 = (__m128i *)((char *)v3 + 104);
        v3[-1].m128i_i32[2] = i[6].m128i_i32[0];
        v3[-1].m128i_i8[12] = i[6].m128i_i8[4];
      }
    }
  }
  return sub_1848A70((__int64 *)a1, v3, v5);
}
