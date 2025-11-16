// Function: sub_B76730
// Address: 0xb76730
//
_QWORD *__fastcall sub_B76730(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); i != result; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result != -4096 && *result != -8192 )
      break;
    ++result;
  }
  return result;
}
