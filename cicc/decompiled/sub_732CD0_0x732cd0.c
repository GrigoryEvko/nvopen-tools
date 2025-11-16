// Function: sub_732CD0
// Address: 0x732cd0
//
void __fastcall sub_732CD0(const __m128i *a1, __m128i **a2)
{
  __m128i *v2; // rax
  __m128i *v3; // r12

  v2 = (__m128i *)sub_726B30(a1[2].m128i_i8[8]);
  *a2 = v2;
  v3 = v2;
  sub_732B40(a1, v2);
  sub_7268E0((__int64)a1, 11);
  a1[4].m128i_i64[1] = (__int64)v3;
  v3[1].m128i_i64[1] = (__int64)a1;
}
