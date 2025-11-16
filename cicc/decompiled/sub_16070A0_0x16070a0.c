// Function: sub_16070A0
// Address: 0x16070a0
//
_QWORD *__fastcall sub_16070A0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result != -8 && *result != -16 )
      break;
    result += 2;
  }
  return result;
}
