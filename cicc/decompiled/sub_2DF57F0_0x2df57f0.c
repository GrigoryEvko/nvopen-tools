// Function: sub_2DF57F0
// Address: 0x2df57f0
//
_QWORD *__fastcall sub_2DF57F0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v3; // rsi
  _QWORD *result; // rax
  unsigned __int64 *v5; // r12

  v3 = (_QWORD *)(a2 & 0xFFFFFFFFFFFFFFC0LL);
  if ( !a3 )
  {
    v5 = v3 + 17;
    do
    {
      if ( *v5 )
        j_j___libc_free_0_0(*v5);
      v5 -= 3;
    }
    while ( v5 != v3 + 5 );
  }
  result = *(_QWORD **)(a1 + 168);
  *v3 = *result;
  *result = v3;
  return result;
}
