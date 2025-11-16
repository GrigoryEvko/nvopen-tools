// Function: sub_2A45D80
// Address: 0x2a45d80
//
void __fastcall sub_2A45D80(__m128i *src, const __m128i *a2, __int64 a3)
{
  const __m128i *i; // rbx
  __int32 v4; // r11d
  __int32 v5; // r10d
  __int32 v6; // r9d
  __int64 v7; // r8
  __int64 v8; // r14
  __int64 v9; // r13
  __int8 v10; // cl
  char v11; // al
  __m128i v12; // xmm3
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  __int64 m128i_i64; // rbx
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i *v19; // r13
  char v20; // al
  __m128i v21; // xmm7
  __m128i v22; // xmm3
  __int32 v23; // [rsp+0h] [rbp-A0h]
  __int32 v24; // [rsp+4h] [rbp-9Ch]
  __int32 v25; // [rsp+8h] [rbp-98h]
  __int8 v26; // [rsp+Fh] [rbp-91h]
  __int64 v27; // [rsp+10h] [rbp-90h]
  const __m128i *v28; // [rsp+10h] [rbp-90h]
  __int64 v29; // [rsp+28h] [rbp-78h] BYREF
  __int64 v30; // [rsp+38h] [rbp-68h] BYREF
  __m128i v31; // [rsp+40h] [rbp-60h] BYREF
  __m128i v32; // [rsp+50h] [rbp-50h] BYREF
  __m128i v33[4]; // [rsp+60h] [rbp-40h] BYREF

  v29 = a3;
  if ( src != a2 )
  {
    for ( i = src + 3; a2 != i; src[2].m128i_i8[8] = v10 )
    {
      while ( 1 )
      {
        sub_2A44DC0((__int64)&v29, (__int64)i, (__int64)src);
        if ( v11 )
          break;
        v12 = _mm_loadu_si128(i);
        v13 = _mm_loadu_si128(i + 1);
        v14 = _mm_loadu_si128(i + 2);
        v28 = i;
        m128i_i64 = (__int64)i[-3].m128i_i64;
        v30 = v29;
        v31 = v12;
        v32 = v13;
        v33[0] = v14;
        while ( 1 )
        {
          v19 = (__m128i *)(m128i_i64 + 48);
          sub_2A44DC0((__int64)&v30, (__int64)&v31, m128i_i64);
          if ( !v20 )
            break;
          v16 = _mm_loadu_si128((const __m128i *)m128i_i64);
          v17 = _mm_loadu_si128((const __m128i *)(m128i_i64 + 16));
          m128i_i64 -= 48;
          v18 = _mm_loadu_si128((const __m128i *)(m128i_i64 + 80));
          *(__m128i *)(m128i_i64 + 96) = v16;
          *(__m128i *)(m128i_i64 + 112) = v17;
          *(__m128i *)(m128i_i64 + 128) = v18;
        }
        v21 = _mm_loadu_si128(&v32);
        v22 = _mm_loadu_si128(v33);
        *v19 = _mm_loadu_si128(&v31);
        i = v28 + 3;
        v19[1] = v21;
        v19[2] = v22;
        if ( a2 == &v28[3] )
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
        v23 = i->m128i_i32[2];
        v24 = i->m128i_i32[0];
        v25 = i->m128i_i32[1];
        v26 = i[2].m128i_i8[8];
        v27 = i[1].m128i_i64[0];
        memmove(&src[3], src, (char *)i - (char *)src);
        v6 = v23;
        v4 = v24;
        v5 = v25;
        v10 = v26;
        v7 = v27;
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
