// Function: sub_22B3440
// Address: 0x22b3440
//
_DWORD *__fastcall sub_22B3440(__int64 a1)
{
  __int64 v1; // rdx
  _DWORD *result; // rax
  _DWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_DWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[8 * v1]; i != result; result += 8 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
