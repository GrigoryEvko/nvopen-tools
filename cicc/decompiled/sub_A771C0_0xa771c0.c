// Function: sub_A771C0
// Address: 0xa771c0
//
__int64 *__fastcall sub_A771C0(__int64 *a1, __int64 a2, int *a3)
{
  __int64 v3; // rsi
  __int64 v4; // r15
  __int64 *v5; // r12
  __int64 v6; // rbx
  __int64 *v7; // r14
  int v10; // [rsp+Ch] [rbp-44h]
  __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a2 - (_QWORD)a1;
  v4 = v3 >> 3;
  v5 = a1;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v4 >> 1;
        v7 = &v5[v4 >> 1];
        v10 = *a3;
        v11[0] = *v7;
        if ( sub_A71840((__int64)v11) || (int)sub_A71AE0(v11) >= v10 )
          break;
        v5 = v7 + 1;
        v4 = v4 - v6 - 1;
        if ( v4 <= 0 )
          return v5;
      }
      v4 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v5;
}
