// Function: sub_87E280
// Address: 0x87e280
//
_QWORD *__fastcall sub_87E280(_QWORD **a1)
{
  _QWORD *result; // rax

  while ( 1 )
  {
    result = *a1;
    if ( !*a1 )
      break;
    *a1 = (_QWORD *)*result;
    *result = qword_4F60018;
    qword_4F60018 = (__int64)result;
  }
  return result;
}
