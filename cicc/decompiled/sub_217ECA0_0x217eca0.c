// Function: sub_217ECA0
// Address: 0x217eca0
//
_DWORD *__fastcall sub_217ECA0(__int64 a1)
{
  _DWORD *result; // rax
  __int64 v2; // rdx
  _DWORD *i; // rdx

  result = *(_DWORD **)(a1 + 8);
  v2 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[2 * v2]; i != result; result += 2 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
