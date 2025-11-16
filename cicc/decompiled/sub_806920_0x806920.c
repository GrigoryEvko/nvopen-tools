// Function: sub_806920
// Address: 0x806920
//
_QWORD *__fastcall sub_806920(__int64 a1, __int64 *a2)
{
  _QWORD *result; // rax
  __m128i v3[3]; // [rsp+0h] [rbp-30h] BYREF

  sub_7E1740(a1, (__int64)v3);
  if ( !a2 )
    return sub_806570(qword_4F04C50, v3);
  result = (_QWORD *)qword_4F04C50;
  if ( *(_QWORD *)(qword_4F04C50 + 48LL) )
    return (_QWORD *)sub_8062F0(a2);
  return result;
}
