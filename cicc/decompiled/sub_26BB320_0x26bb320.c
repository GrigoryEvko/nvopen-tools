// Function: sub_26BB320
// Address: 0x26bb320
//
__int64 __fastcall sub_26BB320(__m128i *a1, __m128i *a2)
{
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rcx
  __int64 result; // rax
  __m128i *v7; // r13
  unsigned __int64 v8; // rdx

  v3 = (_QWORD *)a1[1].m128i_i64[0];
  while ( v3 )
  {
    v4 = (unsigned __int64)v3;
    v3 = (_QWORD *)*v3;
    j_j___libc_free_0(v4);
  }
  if ( (__m128i *)a1->m128i_i64[0] != &a1[3] )
    j_j___libc_free_0(a1->m128i_i64[0]);
  a1[2] = _mm_loadu_si128(a2 + 2);
  if ( (__m128i *)a2->m128i_i64[0] == &a2[3] )
  {
    a1->m128i_i64[0] = (__int64)a1[3].m128i_i64;
    a1[3].m128i_i64[0] = a2[3].m128i_i64[0];
  }
  else
  {
    a1->m128i_i64[0] = a2->m128i_i64[0];
  }
  v5 = a2->m128i_u64[1];
  a1->m128i_i64[1] = v5;
  result = a2[1].m128i_i64[0];
  a1[1].m128i_i64[0] = result;
  a1[1].m128i_i64[1] = a2[1].m128i_i64[1];
  if ( result )
  {
    v7 = a1 + 1;
    v8 = *(_QWORD *)(result + 8) % v5;
    result = v7[-1].m128i_i64[0];
    *(_QWORD *)(result + 8 * v8) = v7;
  }
  a2[2].m128i_i64[1] = 0;
  a2->m128i_i64[1] = 1;
  a2[3].m128i_i64[0] = 0;
  a2->m128i_i64[0] = (__int64)a2[3].m128i_i64;
  a2[1].m128i_i64[0] = 0;
  a2[1].m128i_i64[1] = 0;
  return result;
}
