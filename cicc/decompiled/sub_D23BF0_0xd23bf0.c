// Function: sub_D23BF0
// Address: 0xd23bf0
//
_QWORD *__fastcall sub_D23BF0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // r8

  result = *(_QWORD **)a1;
  v2 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v2 )
  {
    do
    {
      if ( (*result & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*result & 0xFFFFFFFFFFFFFFF8LL) )
        break;
      ++result;
    }
    while ( (_QWORD *)v2 != result );
  }
  return result;
}
