// Function: sub_12A6160
// Address: 0x12a6160
//
_QWORD *__fastcall sub_12A6160(__int64 a1)
{
  _QWORD *result; // rax

  result = *(_QWORD **)(a1 + 8);
  if ( !result )
  {
    result = (_QWORD *)sub_22077B0(8);
    if ( result )
      *result = &unk_49E69F0;
    *(_QWORD *)(a1 + 8) = result;
  }
  return result;
}
