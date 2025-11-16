// Function: sub_16070D0
// Address: 0x16070d0
//
_QWORD *__fastcall sub_16070D0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result != -4 && *result != -8 )
      break;
    result += 2;
  }
  return result;
}
