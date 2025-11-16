// Function: sub_1405E20
// Address: 0x1405e20
//
_QWORD *__fastcall sub_1405E20(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  char *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  char *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __m128i v13; // xmm3
  __int64 i; // r12
  __m128i v15; // xmm1
  char *v16; // rsi
  __int64 v17; // rdx
  __int64 v19; // rdx
  char *v21; // [rsp+10h] [rbp-70h] BYREF
  __int64 v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  __int64 v24; // [rsp+28h] [rbp-58h]
  __m128i v25; // [rsp+30h] [rbp-50h] BYREF
  __m128i v26; // [rsp+40h] [rbp-40h] BYREF

  v7 = *(char **)a2;
  v8 = *(_QWORD *)(a4 + 16);
  v9 = *(_QWORD *)(a4 + 24);
  v10 = *(char **)a4;
  v11 = *(_QWORD *)(a4 + 8);
  if ( *(_QWORD *)(a2 + 24) == a3[3] )
  {
    v25.m128i_i64[0] = (__int64)v10;
    v26.m128i_i64[0] = v8;
    v19 = *a3;
    v25.m128i_i64[1] = v11;
    v26.m128i_i64[1] = v9;
    sub_1405CF0(a1, v7, v19, (char **)&v25);
  }
  else
  {
    v23 = v8;
    v12 = *(_QWORD *)(a2 + 16);
    v21 = v10;
    v22 = v11;
    v24 = v9;
    sub_1405CF0(&v25, v7, v12, &v21);
    v13 = _mm_loadu_si128(&v26);
    *(__m128i *)a4 = _mm_loadu_si128(&v25);
    *(__m128i *)(a4 + 16) = v13;
    for ( i = *(_QWORD *)(a2 + 24) + 8LL; a3[3] != i; *(__m128i *)(a4 + 16) = v15 )
    {
      i += 8;
      v21 = *(char **)a4;
      v22 = *(_QWORD *)(a4 + 8);
      v23 = *(_QWORD *)(a4 + 16);
      v24 = *(_QWORD *)(a4 + 24);
      sub_1405CF0(&v25, *(char **)(i - 8), *(_QWORD *)(i - 8) + 512LL, &v21);
      v15 = _mm_loadu_si128(&v26);
      *(__m128i *)a4 = _mm_loadu_si128(&v25);
    }
    v16 = (char *)a3[1];
    v17 = *a3;
    v25 = *(__m128i *)a4;
    v26 = *(__m128i *)(a4 + 16);
    sub_1405CF0(a1, v16, v17, (char **)&v25);
  }
  return a1;
}
