// Function: sub_1607F70
// Address: 0x1607f70
//
_QWORD *__fastcall sub_1607F70(__int64 a1)
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
