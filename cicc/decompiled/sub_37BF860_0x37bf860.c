// Function: sub_37BF860
// Address: 0x37bf860
//
_DWORD *__fastcall sub_37BF860(__int64 a1)
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
