// Function: sub_341E8E0
// Address: 0x341e8e0
//
_QWORD *__fastcall sub_341E8E0(_QWORD **a1, __int64 *a2)
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
