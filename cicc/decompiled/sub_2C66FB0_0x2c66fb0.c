// Function: sub_2C66FB0
// Address: 0x2c66fb0
//
__int64 __fastcall sub_2C66FB0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v6; // r13
  __int64 *v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 *v11; // rax

  result = 0xFFFFFFFFFFFFFFFLL;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 <= 0xFFFFFFFFFFFFFFFLL )
    result = a3;
  if ( a3 > 0 )
  {
    v4 = result;
    while ( 1 )
    {
      v6 = v4;
      result = sub_2207800(8 * v4);
      v7 = (__int64 *)result;
      if ( result )
        break;
      v4 >>= 1;
      if ( !v4 )
        return result;
    }
    v8 = result + v6 * 8;
    *(_QWORD *)result = *a2;
    v9 = result + 8;
    if ( (__int64 *)v8 == v7 + 1 )
    {
      v11 = v7;
    }
    else
    {
      do
      {
        v10 = *(_QWORD *)(v9 - 8);
        v9 += 8;
        *(_QWORD *)(v9 - 8) = v10;
      }
      while ( v8 != v9 );
      v11 = &v7[v6 - 1];
    }
    result = *v11;
    a1[2] = (__int64)v7;
    a1[1] = v4;
    *a2 = result;
  }
  return result;
}
