// Function: sub_35EF270
// Address: 0x35ef270
//
__m128i *__fastcall sub_35EF270(__m128i *a1, __int8 a2, const char *a3, __m128i *a4)
{
  __m128i *v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // r14
  size_t v9; // rax
  __int64 v10; // rax
  __m128i v12; // xmm0
  __m128i v13; // [rsp+28h] [rbp-78h] BYREF
  void *v14; // [rsp+40h] [rbp-60h]
  __m128i *v15; // [rsp+48h] [rbp-58h]
  __int64 v16; // [rsp+50h] [rbp-50h]
  __m128i v17[4]; // [rsp+58h] [rbp-48h] BYREF

  v6 = (__m128i *)a4->m128i_i64[0];
  v14 = &unk_49E64B0;
  if ( v6 == &a4[1] )
  {
    v12 = _mm_loadu_si128(a4 + 1);
    v8 = a4->m128i_i64[1];
    a4[1].m128i_i8[0] = 0;
    a4->m128i_i64[1] = 0;
    v17[0] = v12;
  }
  else
  {
    v7 = a4[1].m128i_i64[0];
    v8 = a4->m128i_i64[1];
    a4->m128i_i64[0] = (__int64)a4[1].m128i_i64;
    a4->m128i_i64[1] = 0;
    a4[1].m128i_i8[0] = 0;
    v15 = v6;
    v17[0].m128i_i64[0] = v7;
    v16 = v8;
    if ( v6 != v17 )
    {
      v13.m128i_i64[0] = v7;
      goto LABEL_4;
    }
  }
  v6 = &v13;
  v13 = _mm_loadu_si128(v17);
LABEL_4:
  v9 = 0;
  if ( a3 )
    v9 = strlen(a3);
  a1->m128i_i64[1] = v9;
  a1[1].m128i_i64[0] = (__int64)a1[5].m128i_i64;
  a1->m128i_i64[0] = (__int64)a3;
  a1[2].m128i_i8[0] = a2;
  a1[2].m128i_i64[1] = (__int64)&unk_49E64B0;
  a1[1].m128i_i64[1] = 1;
  a1[3].m128i_i64[0] = (__int64)a1[4].m128i_i64;
  if ( v6 == &v13 )
  {
    a1[4] = _mm_loadu_si128(&v13);
  }
  else
  {
    v10 = v13.m128i_i64[0];
    a1[3].m128i_i64[0] = (__int64)v6;
    a1[4].m128i_i64[0] = v10;
  }
  a1[3].m128i_i64[1] = v8;
  a1[5].m128i_i64[0] = (__int64)&a1[2].m128i_i64[1];
  return a1;
}
