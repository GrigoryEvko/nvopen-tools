// Function: sub_B8D830
// Address: 0xb8d830
//
_QWORD *__fastcall sub_B8D830(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result == -1 )
    {
      if ( result[2] != -1 )
        return result;
    }
    else if ( *result != -2 || result[2] != -2 )
    {
      return result;
    }
    result += 4;
  }
  return result;
}
