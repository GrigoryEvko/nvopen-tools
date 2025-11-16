// Function: sub_1B2C670
// Address: 0x1b2c670
//
void __fastcall sub_1B2C670(__m128i *src, const __m128i *a2, __int64 a3)
{
  const __m128i *i; // rbx
  __int32 v4; // r11d
  __int32 v5; // r10d
  __int32 v6; // r9d
  __int64 v7; // r8
  __int64 v8; // r14
  __int64 v9; // r13
  __int8 v10; // cl
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  __m128i v13; // xmm5
  __int64 m128i_i64; // rbx
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __m128i *v18; // r13
  __m128i v19; // xmm7
  __m128i v20; // xmm3
  __int32 v21; // [rsp+0h] [rbp-A0h]
  __int32 v22; // [rsp+4h] [rbp-9Ch]
  __int32 v23; // [rsp+8h] [rbp-98h]
  __int8 v24; // [rsp+Fh] [rbp-91h]
  __int64 v25; // [rsp+10h] [rbp-90h]
  const __m128i *v26; // [rsp+10h] [rbp-90h]
  __int64 v27; // [rsp+28h] [rbp-78h] BYREF
  __int64 v28; // [rsp+38h] [rbp-68h] BYREF
  __m128i v29; // [rsp+40h] [rbp-60h] BYREF
  __m128i v30; // [rsp+50h] [rbp-50h] BYREF
  __m128i v31[4]; // [rsp+60h] [rbp-40h] BYREF

  v27 = a3;
  if ( src != a2 )
  {
    for ( i = src + 3; a2 != i; src[2].m128i_i8[8] = v10 )
    {
      while ( !sub_1B2B020(&v27, (__int64)i, (__int64)src) )
      {
        v11 = _mm_loadu_si128(i);
        v12 = _mm_loadu_si128(i + 1);
        v13 = _mm_loadu_si128(i + 2);
        v26 = i;
        m128i_i64 = (__int64)i[-3].m128i_i64;
        v28 = v27;
        v29 = v11;
        v30 = v12;
        v31[0] = v13;
        while ( 1 )
        {
          v18 = (__m128i *)(m128i_i64 + 48);
          if ( !sub_1B2B020(&v28, (__int64)&v29, m128i_i64) )
            break;
          v15 = _mm_loadu_si128((const __m128i *)m128i_i64);
          v16 = _mm_loadu_si128((const __m128i *)(m128i_i64 + 16));
          m128i_i64 -= 48;
          v17 = _mm_loadu_si128((const __m128i *)(m128i_i64 + 80));
          *(__m128i *)(m128i_i64 + 96) = v15;
          *(__m128i *)(m128i_i64 + 112) = v16;
          *(__m128i *)(m128i_i64 + 128) = v17;
        }
        v19 = _mm_loadu_si128(&v30);
        v20 = _mm_loadu_si128(v31);
        *v18 = _mm_loadu_si128(&v29);
        i = v26 + 3;
        v18[1] = v19;
        v18[2] = v20;
        if ( a2 == &v26[3] )
          return;
      }
      v4 = i->m128i_i32[0];
      v5 = i->m128i_i32[1];
      v6 = i->m128i_i32[2];
      v7 = i[1].m128i_i64[0];
      v8 = i[1].m128i_i64[1];
      v9 = i[2].m128i_i64[0];
      v10 = i[2].m128i_i8[8];
      if ( src != i )
      {
        v21 = i->m128i_i32[2];
        v22 = i->m128i_i32[0];
        v23 = i->m128i_i32[1];
        v24 = i[2].m128i_i8[8];
        v25 = i[1].m128i_i64[0];
        memmove(&src[3], src, (char *)i - (char *)src);
        v6 = v21;
        v4 = v22;
        v5 = v23;
        v10 = v24;
        v7 = v25;
      }
      src->m128i_i32[0] = v4;
      i += 3;
      src->m128i_i32[1] = v5;
      src->m128i_i32[2] = v6;
      src[1].m128i_i64[0] = v7;
      src[1].m128i_i64[1] = v8;
      src[2].m128i_i64[0] = v9;
    }
  }
}
