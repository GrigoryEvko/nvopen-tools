// Function: sub_B140D0
// Address: 0xb140d0
//
unsigned __int64 __fastcall sub_B140D0(__int64 a1)
{
  __int64 v1; // rbx
  unsigned __int64 result; // rax
  _QWORD *v4; // rdi
  unsigned __int64 *v5; // rcx
  unsigned __int64 v6; // rdx

  v1 = a1 + 8;
  result = *(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1 + 8 != result )
  {
    do
    {
      v4 = *(_QWORD **)(a1 + 16);
      v5 = (unsigned __int64 *)v4[1];
      v6 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
      *v5 = v6 | *v5 & 7;
      *(_QWORD *)(v6 + 8) = v5;
      *v4 &= 7uLL;
      v4[1] = 0;
      sub_B12320((__int64)v4);
      result = *(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL;
    }
    while ( v1 != result );
  }
  return result;
}
