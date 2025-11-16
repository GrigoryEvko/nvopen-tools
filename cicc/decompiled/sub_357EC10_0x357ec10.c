// Function: sub_357EC10
// Address: 0x357ec10
//
void __fastcall sub_357EC10(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v4; // r12
  const __m128i *i; // r15
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rcx
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // [rsp-A0h] [rbp-A0h]
  __m128i v12; // [rsp-88h] [rbp-88h] BYREF
  __int64 v13; // [rsp-78h] [rbp-78h]
  __m128i v14; // [rsp-68h] [rbp-68h] BYREF
  __m128i v15; // [rsp-58h] [rbp-58h] BYREF
  __int64 v16; // [rsp-48h] [rbp-48h]

  v2 = a2 - a1;
  if ( v2 > 40 )
  {
    v11 = 0xCCCCCCCCCCCCCCCDLL * (v2 >> 3);
    v4 = (v11 - 2) / 2;
    for ( i = (const __m128i *)(a1 + 40 * v4 + 16); ; i = (const __m128i *)((char *)i - 40) )
    {
      v9 = (__m128i *)i[-1].m128i_i64[0];
      if ( v9 == i )
      {
        v8 = i[1].m128i_i64[0];
        v10 = _mm_loadu_si128(i);
        v14.m128i_i64[0] = (__int64)&v15;
        v7 = i[-1].m128i_i64[1];
        i->m128i_i8[0] = 0;
        i[-1].m128i_i64[1] = 0;
        v13 = v8;
        v12 = v10;
      }
      else
      {
        v6 = i->m128i_i64[0];
        v7 = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
        v8 = i[1].m128i_i64[0];
        v12.m128i_i64[0] = v6;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        v13 = v8;
        v14.m128i_i64[0] = (__int64)&v15;
        if ( v9 != &v12 )
        {
          v14.m128i_i64[0] = (__int64)v9;
          v15.m128i_i64[0] = v6;
          goto LABEL_5;
        }
      }
      v15 = _mm_load_si128(&v12);
LABEL_5:
      v16 = v8;
      v14.m128i_i64[1] = v7;
      v12.m128i_i8[0] = 0;
      sub_357D140(a1, v4, v11, &v14);
      if ( (__m128i *)v14.m128i_i64[0] != &v15 )
        j_j___libc_free_0(v14.m128i_u64[0]);
      if ( !v4 )
        return;
      --v4;
    }
  }
}
