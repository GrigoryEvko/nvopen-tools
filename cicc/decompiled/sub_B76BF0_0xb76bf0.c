// Function: sub_B76BF0
// Address: 0xb76bf0
//
_QWORD *__fastcall sub_B76BF0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result != -4096 && *result != -8192 )
      break;
    ++result;
  }
  return result;
}
