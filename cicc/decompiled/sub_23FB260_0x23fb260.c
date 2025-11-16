// Function: sub_23FB260
// Address: 0x23fb260
//
__int64 ***__fastcall sub_23FB260(__int64 ***a1, __int64 a2, __int64 ***a3)
{
  __int64 v3; // rsi
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 ***v7; // r14
  __int64 **v8; // r15
  __int64 ***v10; // [rsp+0h] [rbp-40h]
  unsigned int v11; // [rsp+Ch] [rbp-34h]

  v3 = a2 - (_QWORD)a1;
  v5 = v3 >> 3;
  v10 = a1;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v5 >> 1;
        v7 = &v10[v5 >> 1];
        v8 = *v7;
        v11 = sub_22DADF0(***a3);
        if ( v11 < (unsigned int)sub_22DADF0(**v8) )
          break;
        v5 = v5 - v6 - 1;
        v10 = v7 + 1;
        if ( v5 <= 0 )
          return v10;
      }
      v5 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v10;
}
