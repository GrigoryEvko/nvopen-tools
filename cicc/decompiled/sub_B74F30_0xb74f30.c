// Function: sub_B74F30
// Address: 0xb74f30
//
_QWORD *__fastcall sub_B74F30(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result != -4096 && *result != -8192 )
      break;
    result += 2;
  }
  return result;
}
