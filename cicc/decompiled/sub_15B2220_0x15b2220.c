// Function: sub_15B2220
// Address: 0x15b2220
//
__int8 *__fastcall sub_15B2220(__m128i *dest, _QWORD *a2, char *a3, unsigned __int64 a4, __int64 a5)
{
  __int8 *v5; // r12
  unsigned __int64 v7; // r14
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __int64 v11; // rax
  __m128i v12; // [rsp+10h] [rbp-80h] BYREF
  __m128i v13; // [rsp+20h] [rbp-70h] BYREF
  __m128i v14; // [rsp+30h] [rbp-60h] BYREF
  __int64 v15; // [rsp+40h] [rbp-50h]
  _QWORD src[7]; // [rsp+58h] [rbp-38h] BYREF

  v5 = a3 + 8;
  src[0] = a5;
  if ( a4 >= (unsigned __int64)(a3 + 8) )
  {
    *(_QWORD *)a3 = src[0];
  }
  else
  {
    v7 = a4 - (_QWORD)a3;
    memcpy(a3, src, a4 - (_QWORD)a3);
    if ( *a2 )
    {
      sub_1593A20((unsigned __int64 *)&dest[4], dest);
      *a2 += 64LL;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v12, dest->m128i_i64, dest[7].m128i_u64[1]);
      v9 = _mm_loadu_si128(&v13);
      v10 = _mm_loadu_si128(&v14);
      v11 = v15;
      dest[4] = _mm_loadu_si128(&v12);
      dest[7].m128i_i64[0] = v11;
      dest[5] = v9;
      dest[6] = v10;
      *a2 = 64;
    }
    v5 = &dest->m128i_i8[8 - v7];
    if ( a4 < (unsigned __int64)v5 )
      abort();
    memcpy(dest, (char *)src + v7, 8 - v7);
  }
  return v5;
}
