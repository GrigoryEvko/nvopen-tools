// Function: sub_7E30B0
// Address: 0x7e30b0
//
_QWORD *__fastcall sub_7E30B0(__m128i *a1, unsigned __int64 a2, unsigned __int8 a3, __int64 a4, int a5)
{
  __int8 v8; // r13
  bool v9; // r13
  _QWORD *result; // rax

  v8 = a1[5].m128i_i8[8];
  sub_72BBE0((__int64)a1, a2, 8u);
  v9 = (v8 & 8) != 0;
  result = sub_7DF9B0(a1, a3, a4);
  if ( a5 )
  {
    if ( v9 )
      return (_QWORD *)sub_7604D0((__int64)a1, 2u);
  }
  return result;
}
