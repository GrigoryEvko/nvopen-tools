// Function: sub_CF93C0
// Address: 0xcf93c0
//
__m128i *__fastcall sub_CF93C0(__m128i *a1, _QWORD *a2, _DWORD *a3, __int64 a4)
{
  unsigned __int64 v6; // rax
  __int64 v8; // rax
  __m128i v9; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+10h] [rbp-40h]

  v6 = sub_CF8CE0(a2);
  if ( v6 && (sub_CF90E0((__int64)&v9, a2[3], v6), &a3[a4] != sub_CF8D60(a3, (__int64)&a3[a4], v9.m128i_i32)) )
  {
    v8 = v10;
    *a1 = _mm_loadu_si128(&v9);
    a1[1].m128i_i64[0] = v8;
  }
  else
  {
    a1->m128i_i32[0] = 0;
    a1->m128i_i64[1] = 0;
    a1[1].m128i_i64[0] = 0;
  }
  return a1;
}
