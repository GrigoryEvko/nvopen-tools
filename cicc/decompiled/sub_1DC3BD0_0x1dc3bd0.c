// Function: sub_1DC3BD0
// Address: 0x1dc3bd0
//
unsigned __int64 __fastcall sub_1DC3BD0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  unsigned __int64 result; // rax

  a1->m128i_i64[0] = a2;
  v6 = *(_QWORD *)(a2 + 40);
  a1[1].m128i_i64[0] = a3;
  a1->m128i_i64[1] = v6;
  a1[1].m128i_i64[1] = a4;
  a1[2].m128i_i64[0] = a5;
  result = sub_1DC3680(a1, a2, a3, a4, a5, a6);
  a1[9].m128i_i32[0] = 0;
  return result;
}
