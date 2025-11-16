// Function: sub_2B11460
// Address: 0x2b11460
//
void __fastcall sub_2B11460(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9

  v7 = a2 - (_QWORD)a1;
  v10 = (__int64)a3 + v7;
  v11 = 0x8E38E38E38E38E39LL * (v7 >> 3);
  if ( v7 <= 432 )
  {
    sub_2B0F460((__int64)a1, a2, v11, a4, a5, a6);
  }
  else
  {
    v12 = (__int64)a1;
    do
    {
      v13 = v12;
      v12 += 504;
      sub_2B0F460(v13, v12, v11, a4, a5, a6);
    }
    while ( a2 - v12 > 432 );
    sub_2B0F460(v12, a2, v11, a4, a5, a6);
    if ( v7 > 504 )
    {
      v16 = 7;
      do
      {
        sub_2B11390(a1, a2, (__int64)a3, v16, v14, v15);
        v17 = 2 * v16;
        v16 *= 4;
        sub_2B11390(a3, v10, (__int64)a1, v17, v18, v19);
      }
      while ( (__int64)(0x8E38E38E38E38E39LL * (v7 >> 3)) > v16 );
    }
  }
}
