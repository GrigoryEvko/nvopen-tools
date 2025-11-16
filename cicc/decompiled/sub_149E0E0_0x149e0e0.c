// Function: sub_149E0E0
// Address: 0x149e0e0
//
__int64 __fastcall sub_149E0E0(
        __int64 a1,
        __m128i *a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__m128i *, __m128i *),
        __int64 a5,
        __int64 a6)
{
  __int64 result; // rax
  __m128i *v8; // rbx
  __m128i *v9; // r12
  __m128i *v10; // r15
  __m128i *v11; // rsi
  __m128i v12; // xmm1
  __m128i v13; // xmm0
  __int64 v14; // rax
  __int8 *v15; // r15
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int128 v19; // xmm4
  __int64 v20; // r12
  __int128 v21; // xmm5
  __int64 v22; // rdx
  __m128i *v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  __m128i *v25; // [rsp+18h] [rbp-68h]

  result = (__int64)a2->m128i_i64 - a1;
  v24 = a3;
  v23 = a2;
  if ( (__int64)a2->m128i_i64 - a1 <= 640 )
    return result;
  if ( !a3 )
  {
    v25 = a2;
    goto LABEL_14;
  }
  while ( 2 )
  {
    v8 = v23;
    v9 = (__m128i *)(a1 + 40);
    --v24;
    sub_149D9A0(
      (__m128i *)a1,
      (__m128i *)(a1 + 40),
      (__m128i *)(a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((__int64)v23->m128i_i64 - a1) >> 3)) / 2)),
      (__m128i *)((char *)v23 - 40),
      a4);
    while ( 1 )
    {
      v25 = v9;
      if ( a4(v9, (__m128i *)a1) )
        goto LABEL_9;
      v10 = (__m128i *)((char *)v8 - 40);
      do
      {
        v11 = v10;
        v8 = v10;
        v10 = (__m128i *)((char *)v10 - 40);
      }
      while ( a4((__m128i *)a1, v11) );
      if ( v9 >= v8 )
        break;
      v12 = _mm_loadu_si128(v9);
      v13 = _mm_loadu_si128(v9 + 1);
      v14 = v9[2].m128i_i64[0];
      *v9 = _mm_loadu_si128(v8);
      v9[1] = _mm_loadu_si128(v8 + 1);
      v9[2].m128i_i32[0] = v8[2].m128i_i32[0];
      v8[2].m128i_i32[0] = v14;
      *v8 = v12;
      v8[1] = v13;
LABEL_9:
      v9 = (__m128i *)((char *)v9 + 40);
    }
    sub_149E0E0(v9, v23, v24, a4);
    result = (__int64)v9->m128i_i64 - a1;
    if ( (__int64)v9->m128i_i64 - a1 > 640 )
    {
      if ( v24 )
      {
        v23 = v9;
        continue;
      }
LABEL_14:
      v15 = &v25[-3].m128i_i8[8];
      sub_149DF90((const __m128i *)a1, v25, (unsigned __int64)v25, (__int64 (__fastcall *)(__m128i *))a4, a5, a6);
      do
      {
        v18 = *((_QWORD *)v15 + 4);
        v19 = (__int128)_mm_loadu_si128((const __m128i *)v15);
        v20 = (__int64)&v15[-a1];
        v21 = (__int128)_mm_loadu_si128((const __m128i *)v15 + 1);
        *(__m128i *)v15 = _mm_loadu_si128((const __m128i *)a1);
        v22 = (__int64)&v15[-a1] >> 3;
        v15 -= 40;
        *(__m128i *)(v15 + 56) = _mm_loadu_si128((const __m128i *)(a1 + 16));
        *((_DWORD *)v15 + 18) = *(_DWORD *)(a1 + 32);
        result = sub_149DD50(
                   a1,
                   0,
                   0xCCCCCCCCCCCCCCCDLL * v22,
                   (__int64 (__fastcall *)(__m128i *))a4,
                   v16,
                   v17,
                   v19,
                   v21,
                   v18);
      }
      while ( v20 > 40 );
    }
    return result;
  }
}
