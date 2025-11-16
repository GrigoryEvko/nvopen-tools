// Function: sub_6447E0
// Address: 0x6447e0
//
_QWORD *__fastcall sub_6447E0(__int64 a1)
{
  _QWORD *i; // rax
  _QWORD *result; // rax

  for ( i = *(_QWORD **)(a1 + 184); i; i = (_QWORD *)*i )
    i[6] = 0;
  for ( result = *(_QWORD **)(a1 + 200); result; result = (_QWORD *)*result )
    result[6] = 0;
  return result;
}
