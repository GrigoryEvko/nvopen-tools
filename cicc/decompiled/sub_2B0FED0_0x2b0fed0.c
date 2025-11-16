// Function: sub_2B0FED0
// Address: 0x2b0fed0
//
void __fastcall sub_2B0FED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+8h] [rbp-38h]

  v8 = (a2 - a1) >> 6;
  v9 = a3 + a2 - a1;
  v18 = a2 - a1;
  v19 = v8;
  if ( a2 - a1 <= 384 )
  {
    sub_2B0F830(a1, a2, v8, a4, a5, a6);
  }
  else
  {
    v10 = a1;
    do
    {
      v11 = v10;
      v10 += 448;
      sub_2B0F830(v11, v10, v8, a4, a5, a6);
    }
    while ( a2 - v10 > 384 );
    sub_2B0F830(v10, a2, v8, a4, a5, a6);
    if ( v18 > 448 )
    {
      v14 = 7;
      do
      {
        sub_2B0FE20(a1, a2, a3, v14, v12, v13);
        v15 = 2 * v14;
        v14 *= 4;
        sub_2B0FE20(a3, v9, a1, v15, v16, v17);
      }
      while ( v19 > v14 );
    }
  }
}
