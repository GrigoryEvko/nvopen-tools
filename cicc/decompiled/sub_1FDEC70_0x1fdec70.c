// Function: sub_1FDEC70
// Address: 0x1fdec70
//
_QWORD *__fastcall sub_1FDEC70(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  for ( i = *(_QWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( (*result & 4) != 0 )
      break;
    if ( (*result & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
      break;
    result += 2;
  }
  return result;
}
