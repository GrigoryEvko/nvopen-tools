// Function: sub_37BFB00
// Address: 0x37bfb00
//
_QWORD *__fastcall sub_37BFB00(__int64 a1)
{
  __int64 v1; // rcx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[17 * v1]; i != result; result += 17 )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
