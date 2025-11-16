// Function: sub_2B1C1A0
// Address: 0x2b1c1a0
//
unsigned int *__fastcall sub_2B1C1A0(
        unsigned int *a1,
        __int64 a2,
        unsigned int *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7)
{
  __int64 v7; // rsi
  unsigned int *v9; // r13
  __int64 v10; // rbx
  __int64 v11; // r12
  unsigned int *v12; // r15

  v7 = a2 - (_QWORD)a1;
  v9 = a1;
  v10 = v7 >> 2;
  if ( v7 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v11 = v10 >> 1;
        v12 = &v9[v10 >> 1];
        if ( !sub_2B1BC20(&a7, *v12, *a3) )
          break;
        v9 = v12 + 1;
        v10 = v10 - v11 - 1;
        if ( v10 <= 0 )
          return v9;
      }
      v10 >>= 1;
    }
    while ( v11 > 0 );
  }
  return v9;
}
