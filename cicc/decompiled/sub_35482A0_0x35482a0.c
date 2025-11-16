// Function: sub_35482A0
// Address: 0x35482a0
//
_DWORD *__fastcall sub_35482A0(__int64 a1)
{
  __int64 v1; // rdx
  _DWORD *result; // rax
  _DWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_DWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[22 * v1]; i != result; result += 22 )
  {
    if ( result )
      *result = 0x7FFFFFFF;
  }
  return result;
}
