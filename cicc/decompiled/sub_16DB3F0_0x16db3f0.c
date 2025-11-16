// Function: sub_16DB3F0
// Address: 0x16db3f0
//
__m128i *__fastcall sub_16DB3F0(_BYTE *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __m128i *result; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned int v9; // edx
  __int64 v10; // r14
  __m128i *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __m128i *v14; // rdi
  __m128i *v15; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+18h] [rbp-68h]
  __m128i v17; // [rsp+20h] [rbp-60h] BYREF
  __m128i *v18; // [rsp+30h] [rbp-50h] BYREF
  __int64 v19; // [rsp+38h] [rbp-48h]
  __m128i v20[4]; // [rsp+40h] [rbp-40h] BYREF

  result = (__m128i *)sub_16D40F0((__int64)&qword_4FA1650);
  v7 = qword_4FA1660;
  if ( result )
    v7 = result->m128i_i64[0];
  if ( v7 )
  {
    if ( a1 )
    {
      v15 = &v17;
      sub_16D9940((__int64 *)&v15, a1, (__int64)&a1[a2]);
      if ( a3 )
      {
LABEL_6:
        v18 = v20;
        sub_16D9940((__int64 *)&v18, a3, (__int64)&a3[a4]);
LABEL_7:
        v8 = sub_220F880();
        v9 = *(_DWORD *)(v7 + 8);
        v10 = v8;
        if ( v9 >= *(_DWORD *)(v7 + 12) )
        {
          sub_16D99F0((unsigned __int64 *)v7);
          v9 = *(_DWORD *)(v7 + 8);
        }
        result = (__m128i *)(*(_QWORD *)v7 + 80LL * v9);
        if ( result )
        {
          result->m128i_i64[0] = v10;
          result[1].m128i_i64[0] = (__int64)result[2].m128i_i64;
          v11 = v15;
          result->m128i_i64[1] = 0;
          if ( v11 == &v17 )
          {
            result[2] = _mm_load_si128(&v17);
          }
          else
          {
            result[1].m128i_i64[0] = (__int64)v11;
            result[2].m128i_i64[0] = v17.m128i_i64[0];
          }
          v12 = v16;
          v15 = &v17;
          v16 = 0;
          result[1].m128i_i64[1] = v12;
          result[3].m128i_i64[0] = (__int64)result[4].m128i_i64;
          v17.m128i_i8[0] = 0;
          if ( v18 == v20 )
          {
            result[4] = _mm_load_si128(v20);
          }
          else
          {
            result[3].m128i_i64[0] = (__int64)v18;
            result[4].m128i_i64[0] = v20[0].m128i_i64[0];
          }
          v13 = v19;
          v20[0].m128i_i8[0] = 0;
          v19 = 0;
          result[3].m128i_i64[1] = v13;
          ++*(_DWORD *)(v7 + 8);
        }
        else
        {
          v14 = v18;
          *(_DWORD *)(v7 + 8) = v9 + 1;
          if ( v14 != v20 )
            result = (__m128i *)j_j___libc_free_0(v14, v20[0].m128i_i64[0] + 1);
        }
        if ( v15 != &v17 )
          return (__m128i *)j_j___libc_free_0(v15, v17.m128i_i64[0] + 1);
        return result;
      }
    }
    else
    {
      v16 = 0;
      v15 = &v17;
      v17.m128i_i8[0] = 0;
      if ( a3 )
        goto LABEL_6;
    }
    v19 = 0;
    v18 = v20;
    v20[0].m128i_i8[0] = 0;
    goto LABEL_7;
  }
  return result;
}
