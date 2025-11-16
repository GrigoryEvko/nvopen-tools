// Function: sub_C25EF0
// Address: 0xc25ef0
//
_QWORD *__fastcall sub_C25EF0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result < 0xFFFFFFFFFFFFFFFELL )
      break;
    result += 2;
  }
  return result;
}
