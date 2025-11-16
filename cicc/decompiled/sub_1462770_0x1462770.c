// Function: sub_1462770
// Address: 0x1462770
//
__int64 **__fastcall sub_1462770(
        __int64 **a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        _QWORD *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v10; // rsi
  __int64 **v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 **v14; // r12
  __int64 **v17; // [rsp+18h] [rbp-38h]

  v10 = a2 - (_QWORD)a1;
  v11 = a1;
  v12 = v10 >> 3;
  if ( v10 > 0 )
  {
    v17 = a1;
    do
    {
      while ( 1 )
      {
        v13 = v12 >> 1;
        v14 = &v17[v12 >> 1];
        if ( (int)sub_1462150(a7, a8, *a9, *v14, *a3, a10, 0) >= 0 )
          break;
        v12 = v12 - v13 - 1;
        v17 = v14 + 1;
        if ( v12 <= 0 )
          return v17;
      }
      v12 >>= 1;
    }
    while ( v13 > 0 );
    return v17;
  }
  return v11;
}
