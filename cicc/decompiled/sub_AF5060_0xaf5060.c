// Function: sub_AF5060
// Address: 0xaf5060
//
__int64 __fastcall sub_AF5060(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 result; // rax
  _QWORD *i; // r13

  v1 = *(_QWORD **)(a1 + 136);
  result = *(unsigned int *)(a1 + 144);
  for ( i = &v1[result]; i != v1; ++v1 )
  {
    if ( *v1 )
      result = sub_B96E90(v1, *v1, a1 & 0xFFFFFFFFFFFFFFFCLL | 1);
  }
  return result;
}
