// Function: sub_1D464F0
// Address: 0x1d464f0
//
_QWORD *__fastcall sub_1D464F0(_QWORD **a1, __int64 *a2)
{
  __int64 v2; // rcx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v2 = *a2;
  result = (_QWORD *)**a1;
  for ( i = &result[*((unsigned int *)*a1 + 2)]; i != result; ++result )
  {
    while ( v2 != *result )
    {
      if ( i == ++result )
        return result;
    }
    *result = 0;
  }
  return result;
}
