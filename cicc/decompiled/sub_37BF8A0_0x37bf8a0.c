// Function: sub_37BF8A0
// Address: 0x37bf8a0
//
_QWORD *__fastcall sub_37BF8A0(__int64 a1)
{
  __int64 v1; // rcx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[7 * v1]; result != i; result += 7 )
  {
    if ( result )
    {
      *result = -4096;
      result[1] = -1;
      result[2] = -1;
    }
  }
  return result;
}
