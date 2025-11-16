// Function: sub_335B9C0
// Address: 0x335b9c0
//
__m128i *__fastcall sub_335B9C0(__m128i *a1, __int64 a2)
{
  __m128i *v3; // rax
  __int64 v4; // rcx
  __int64 *v5; // rdi
  unsigned __int64 v7[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+10h] [rbp-20h] BYREF

  sub_2E31BE0((__int64)v7, *(_QWORD *)(a2 + 584));
  v3 = (__m128i *)sub_2241130(v7, 0, 0, "sunit-dag.", 0xAu);
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
    j_j___libc_free_0((unsigned __int64)v5);
  return a1;
}
