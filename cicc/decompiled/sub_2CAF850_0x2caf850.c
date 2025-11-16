// Function: sub_2CAF850
// Address: 0x2caf850
//
__int64 __fastcall sub_2CAF850(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 v11; // rdx

  v4 = a2 - a1;
  v5 = a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 3);
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(a3 + 16);
        v9 = v7 >> 1;
        v10 = v5 + 8 * ((v7 >> 1) + (v7 & 0xFFFFFFFFFFFFFFFELL));
        if ( !v8 )
          break;
        v11 = *(_QWORD *)(v10 + 16);
        if ( !v11 || !(unsigned __int8)sub_B19DB0(a4, v8, v11) )
          break;
        v7 >>= 1;
        if ( v9 <= 0 )
          return v5;
      }
      v5 = v10 + 24;
      v7 = v7 - v9 - 1;
    }
    while ( v7 > 0 );
  }
  return v5;
}
