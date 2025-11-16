// Function: sub_24481E0
// Address: 0x24481e0
//
__int64 __fastcall sub_24481E0(__m128i *a1, __m128i *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rcx
  __int64 result; // rax
  __m128i v5; // xmm0
  __m128i *v6; // rdi
  unsigned __int64 v7; // rdx

  v2 = a2->m128i_i64[0];
  v3 = a2->m128i_u64[1];
  a1[3].m128i_i64[0] = 0;
  a1->m128i_i64[0] = v2;
  result = a2[1].m128i_i64[0];
  a1->m128i_i64[1] = v3;
  a1[1].m128i_i64[0] = result;
  v5 = _mm_loadu_si128(a2 + 2);
  a1[1].m128i_i64[1] = a2[1].m128i_i64[1];
  a1[2] = v5;
  if ( &a2[3] == (__m128i *)a2->m128i_i64[0] )
  {
    a1->m128i_i64[0] = (__int64)a1[3].m128i_i64;
    a1[3].m128i_i64[0] = a2[3].m128i_i64[0];
  }
  if ( result )
  {
    v6 = a1 + 1;
    v7 = *(int *)(result + 8) % v3;
    result = v6[-1].m128i_i64[0];
    *(_QWORD *)(result + 8 * v7) = v6;
  }
  a2[2].m128i_i64[1] = 0;
  a2->m128i_i64[1] = 1;
  a2[3].m128i_i64[0] = 0;
  a2->m128i_i64[0] = (__int64)a2[3].m128i_i64;
  a2[1].m128i_i64[0] = 0;
  a2[1].m128i_i64[1] = 0;
  return result;
}
