// Function: sub_355ABF0
// Address: 0x355abf0
//
__int64 *__fastcall sub_355ABF0(__int64 *a1, __int64 *a2, __int64 a3, __m128i *a4)
{
  char *v7; // rdx
  __int64 v8; // rax
  __m128i v9; // rdi
  __int64 v10; // rcx
  __m128i v11; // xmm3
  __int64 i; // r12
  __m128i v13; // xmm1
  char *v14; // rdx
  __int64 v15; // rsi
  __int64 v17; // rsi
  __int64 v19; // [rsp+10h] [rbp-70h] BYREF
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  __int64 v22; // [rsp+28h] [rbp-58h]
  __m128i v23; // [rsp+30h] [rbp-50h] BYREF
  __m128i v24; // [rsp+40h] [rbp-40h] BYREF

  v7 = *(char **)a3;
  v8 = a4[1].m128i_i64[1];
  v9 = *a4;
  v10 = a4[1].m128i_i64[0];
  if ( a2[3] == *(_QWORD *)(a3 + 24) )
  {
    v23 = v9;
    v17 = *a2;
    v24.m128i_i64[0] = v10;
    v24.m128i_i64[1] = v8;
    sub_355AA90(a1, v17, v7, v23.m128i_i64);
  }
  else
  {
    v20 = v9.m128i_i64[1];
    v9.m128i_i64[1] = *(_QWORD *)(a3 + 8);
    v19 = v9.m128i_i64[0];
    v21 = v10;
    v22 = v8;
    sub_355AA90(v23.m128i_i64, v9.m128i_i64[1], v7, &v19);
    v11 = _mm_loadu_si128(&v24);
    *a4 = _mm_loadu_si128(&v23);
    a4[1] = v11;
    for ( i = *(_QWORD *)(a3 + 24) - 8LL; a2[3] != i; a4[1] = v13 )
    {
      i -= 8;
      v19 = a4->m128i_i64[0];
      v20 = a4->m128i_i64[1];
      v21 = a4[1].m128i_i64[0];
      v22 = a4[1].m128i_i64[1];
      sub_355AA90(v23.m128i_i64, *(_QWORD *)(i + 8), (char *)(*(_QWORD *)(i + 8) + 512LL), &v19);
      v13 = _mm_loadu_si128(&v24);
      *a4 = _mm_loadu_si128(&v23);
    }
    v14 = (char *)a2[2];
    v15 = *a2;
    v23 = *a4;
    v24 = a4[1];
    sub_355AA90(a1, v15, v14, v23.m128i_i64);
  }
  return a1;
}
