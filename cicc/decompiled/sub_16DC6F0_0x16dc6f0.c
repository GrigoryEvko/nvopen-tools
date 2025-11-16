// Function: sub_16DC6F0
// Address: 0x16dc6f0
//
__int64 __fastcall sub_16DC6F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // r12
  const __m128i *i; // r15
  __int64 v5; // rsi
  __int64 v6; // rcx
  __m128i v7; // xmm1
  __m128i v8; // xmm0
  __int64 result; // rax
  __m128i *v10; // rdx
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __int64 v13; // [rsp-A0h] [rbp-A0h]
  __m128i v14; // [rsp-88h] [rbp-88h] BYREF
  __m128i v15; // [rsp-78h] [rbp-78h] BYREF
  __m128i v16; // [rsp-68h] [rbp-68h] BYREF
  _OWORD v17[5]; // [rsp-58h] [rbp-58h] BYREF

  v2 = a2 - a1;
  if ( v2 > 48 )
  {
    v13 = 0xAAAAAAAAAAAAAAABLL * (v2 >> 4);
    v3 = (v13 - 2) / 2;
    for ( i = (const __m128i *)(a1
                              + 16 * (v3 + ((v13 - 2 + ((unsigned __int64)(v13 - 2) >> 63)) & 0xFFFFFFFFFFFFFFFELL))
                              + 16); ; i -= 3 )
    {
      v10 = (__m128i *)i[-1].m128i_i64[0];
      if ( v10 == i )
      {
        v11 = _mm_loadu_si128(i);
        v6 = i[-1].m128i_i64[1];
        i->m128i_i8[0] = 0;
        v12 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[1] = 0;
        v16.m128i_i64[0] = (__int64)v17;
        v14 = v11;
        v15 = v12;
      }
      else
      {
        v5 = i->m128i_i64[0];
        v6 = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
        v7 = _mm_loadu_si128(i + 1);
        v14.m128i_i64[0] = v5;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v16.m128i_i64[0] = (__int64)v17;
        v15 = v7;
        if ( v10 != &v14 )
        {
          v16.m128i_i64[0] = (__int64)v10;
          *(_QWORD *)&v17[0] = v5;
          goto LABEL_5;
        }
      }
      v17[0] = _mm_load_si128(&v14);
LABEL_5:
      v16.m128i_i64[1] = v6;
      v8 = _mm_load_si128(&v15);
      v14.m128i_i8[0] = 0;
      v17[1] = v8;
      result = sub_16DA0A0(a1, v3, v13, &v16);
      if ( (_OWORD *)v16.m128i_i64[0] != v17 )
        result = j_j___libc_free_0(v16.m128i_i64[0], *(_QWORD *)&v17[0] + 1LL);
      if ( !v3 )
        return result;
      --v3;
    }
  }
  return result;
}
