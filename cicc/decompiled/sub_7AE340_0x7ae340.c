// Function: sub_7AE340
// Address: 0x7ae340
//
_QWORD *__fastcall sub_7AE340(__int64 a1)
{
  _QWORD *result; // rax

  for ( result = *(_QWORD **)(a1 + 8); result; result = (_QWORD *)*result )
    result[5] = result;
  return result;
}
