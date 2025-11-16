// Function: sub_1607FA0
// Address: 0x1607fa0
//
_QWORD *__fastcall sub_1607FA0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result != -8 && *result != -16 )
      break;
    ++result;
  }
  return result;
}
