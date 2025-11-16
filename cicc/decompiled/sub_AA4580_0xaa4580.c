// Function: sub_AA4580
// Address: 0xaa4580
//
_QWORD *__fastcall sub_AA4580(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax

  result = *(_QWORD **)(a2 + 64);
  if ( !result )
  {
    result = (_QWORD *)sub_22077B0(24);
    if ( result )
    {
      result[2] = result + 1;
      result[1] = (unsigned __int64)(result + 1) | 4;
    }
    *result = a2;
    *(_QWORD *)(a2 + 64) = result;
  }
  return result;
}
