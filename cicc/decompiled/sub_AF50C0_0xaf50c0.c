// Function: sub_AF50C0
// Address: 0xaf50c0
//
__int64 __fastcall sub_AF50C0(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 result; // rax
  _QWORD *i; // r12

  v1 = *(_QWORD **)(a1 + 136);
  result = *(unsigned int *)(a1 + 144);
  for ( i = &v1[result]; i != v1; ++v1 )
  {
    if ( *v1 )
      result = sub_B91220(v1);
  }
  return result;
}
