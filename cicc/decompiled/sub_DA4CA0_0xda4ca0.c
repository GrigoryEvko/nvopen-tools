// Function: sub_DA4CA0
// Address: 0xda4ca0
//
unsigned __int64 *__fastcall sub_DA4CA0(unsigned __int64 *a1, __int64 a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  unsigned __int64 *v5; // r13
  __int64 v6; // rbx
  __int64 v8; // r15
  unsigned __int64 *v9; // r14
  __int64 v12; // [rsp+18h] [rbp-38h]

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 3;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        v9 = &v5[v6 >> 1];
        v12 = sub_DA4700(*(_QWORD **)a4, **(_QWORD **)(a4 + 8), *v9, *a3, *(_QWORD *)(a4 + 16), 0);
        if ( BYTE4(v12) )
        {
          if ( (int)v12 < 0 )
            break;
        }
        v6 >>= 1;
        if ( v8 <= 0 )
          return v5;
      }
      v5 = v9 + 1;
      v6 = v6 - v8 - 1;
    }
    while ( v6 > 0 );
  }
  return v5;
}
