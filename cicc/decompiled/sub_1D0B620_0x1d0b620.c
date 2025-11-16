// Function: sub_1D0B620
// Address: 0x1d0b620
//
__m128i *__fastcall sub_1D0B620(__m128i *a1, __int64 a2)
{
  __m128i *v3; // rax
  __int64 v4; // rcx
  __int64 *v5; // rdi
  _QWORD v7[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+10h] [rbp-20h] BYREF

  sub_1DD62B0(v7, *(_QWORD *)(a2 + 616));
  v3 = (__m128i *)sub_2241130(v7, 0, 0, "sunit-dag.", 10);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v3->m128i_i64[0] == &v3[1] )
  {
    a1[1] = _mm_loadu_si128(v3 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v3->m128i_i64[0];
    a1[1].m128i_i64[0] = v3[1].m128i_i64[0];
  }
  v4 = v3->m128i_i64[1];
  v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
  v5 = (__int64 *)v7[0];
  v3->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v4;
  v3[1].m128i_i8[0] = 0;
  if ( v5 != &v8 )
    j_j___libc_free_0(v5, v8 + 1);
  return a1;
}
