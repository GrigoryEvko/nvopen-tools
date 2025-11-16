// Function: sub_87E2B0
// Address: 0x87e2b0
//
_QWORD *__fastcall sub_87E2B0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rdx

  for ( result = a2; result; result = (_QWORD *)*result )
  {
    v3 = (_QWORD *)result[1];
    if ( v3 && *v3 == *a1 )
      break;
  }
  return result;
}
