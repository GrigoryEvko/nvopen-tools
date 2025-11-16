// Function: sub_5CF720
// Address: 0x5cf720
//
_QWORD *__fastcall sub_5CF720(const __m128i *a1, const __m128i *a2)
{
  _QWORD *result; // rax
  _QWORD **v3; // rbx
  _QWORD *v4; // [rsp+8h] [rbp-18h] BYREF

  v4 = 0;
  if ( !a1 )
    return (_QWORD *)sub_5CF190(a2);
  result = (_QWORD *)sub_5CF190(a1);
  v4 = result;
  if ( a2 )
  {
    v3 = sub_5CB9F0(&v4);
    *v3 = (_QWORD *)sub_5CF190(a2);
    return v4;
  }
  return result;
}
