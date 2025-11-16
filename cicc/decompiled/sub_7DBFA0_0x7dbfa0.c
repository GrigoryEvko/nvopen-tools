// Function: sub_7DBFA0
// Address: 0x7dbfa0
//
__int64 sub_7DBFA0()
{
  __int64 result; // rax
  const __m128i *v1; // rax
  __m128i *v2; // rax
  _QWORD *v3; // rsi

  result = qword_4F18948;
  if ( !qword_4F18948 )
  {
    qword_4F18948 = sub_7E16B0(10);
    sub_7E1CA0(qword_4F18948);
    v1 = (const __m128i *)sub_7DB910(5u, 0);
    v2 = sub_73C570(v1, 1);
    sub_72D2E0(v2);
    sub_7E1B70((char *)"tinfo");
    v3 = sub_72BA30(7u);
    sub_7E1B70("offset_flags");
    sub_7E1C00(qword_4F18948, (unsigned __int64)v3);
    return qword_4F18948;
  }
  return result;
}
