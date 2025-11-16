// Function: sub_8975E0
// Address: 0x8975e0
//
_QWORD *__fastcall sub_8975E0(const __m128i *a1, unsigned int a2, int a3)
{
  __int64 m128i_i64; // r12
  _QWORD *result; // rax

  m128i_i64 = (__int64)a1[18].m128i_i64;
  if ( a1[18].m128i_i64[1] )
    sub_7AEA70(a1 + 18);
  sub_7AE700((__int64)(qword_4F061C0 + 3), a1[17].m128i_i32[2], a2, a3, m128i_i64);
  sub_7AE210(m128i_i64);
  sub_7AE340(m128i_i64);
  result = (_QWORD *)a1[7].m128i_u32[3];
  if ( (_DWORD)result )
  {
    result = sub_7BDC00();
    a1[7].m128i_i32[3] = 0;
  }
  return result;
}
