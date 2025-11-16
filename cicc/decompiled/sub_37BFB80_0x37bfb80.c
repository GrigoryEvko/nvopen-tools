// Function: sub_37BFB80
// Address: 0x37bfb80
//
_QWORD *__fastcall sub_37BFB80(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  v2 = unk_5051170;
  result = *(_QWORD **)(a1 + 8);
  for ( i = &result[2 * v1]; result != i; result += 2 )
  {
    if ( result )
      *result = v2;
  }
  return result;
}
