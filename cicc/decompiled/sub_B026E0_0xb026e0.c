// Function: sub_B026E0
// Address: 0xb026e0
//
_QWORD *__fastcall sub_B026E0(_QWORD **a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 *v3; // r12
  __int64 *v4; // rbx

  if ( !a2 )
    return 0;
  result = *a1;
  if ( a2 != 1 )
  {
    v3 = (__int64 *)&a1[a2];
    v4 = (__int64 *)(a1 + 1);
    if ( v3 != (__int64 *)(a1 + 1) )
    {
      do
      {
        result = sub_B026B0((__int64)result, *v4);
        if ( !result )
          break;
        ++v4;
      }
      while ( v3 != v4 );
    }
  }
  return result;
}
