// Function: sub_3535DB0
// Address: 0x3535db0
//
__int64 __fastcall sub_3535DB0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // rcx
  __int64 v8; // rdx
  _QWORD *v9; // rsi
  _QWORD *v10; // rax

  result = 0xFFFFFFFFFFFFFFFLL;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 <= 0xFFFFFFFFFFFFFFFLL )
    result = a3;
  if ( a3 > 0 )
  {
    v5 = result;
    while ( 1 )
    {
      v6 = 8 * v5;
      result = sub_2207800(8 * v5);
      v7 = result;
      if ( result )
        break;
      v5 >>= 1;
      if ( !v5 )
        return result;
    }
    v8 = *a2;
    v9 = (_QWORD *)(result + v6);
    v10 = (_QWORD *)(result + 8);
    *a2 = 0;
    *(v10 - 1) = v8;
    if ( v9 == v10 )
    {
      result = v7;
    }
    else
    {
      while ( 1 )
      {
        *v10++ = v8;
        *(v10 - 2) = 0;
        if ( v9 == v10 )
          break;
        v8 = *(v10 - 1);
      }
      v8 = *(_QWORD *)(v7 + v6 - 8);
      result = v7 + v6 - 8;
    }
    *(_QWORD *)result = 0;
    *a2 = v8;
    a1[2] = v7;
    a1[1] = v5;
  }
  return result;
}
