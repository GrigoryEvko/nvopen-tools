// Function: sub_37BEB30
// Address: 0x37beb30
//
_DWORD *__fastcall sub_37BEB30(__int64 a1)
{
  __int64 v1; // rcx
  _DWORD *result; // rax
  _DWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_DWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[28 * v1]; i != result; result += 28 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
