// Function: sub_192E9B0
// Address: 0x192e9b0
//
void __fastcall sub_192E9B0(char *a1, __int64 a2, char *a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rsi
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // r8
  int v15; // r9d
  __int64 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // r8
  int v19; // r9d

  v7 = a2 - (_QWORD)a1;
  v10 = (__int64)&a3[v7];
  v11 = 0x8E38E38E38E38E39LL * (v7 >> 3);
  if ( v7 <= 432 )
  {
    sub_192E490((__int64)a1, a2, v11, a4, a5, a6);
  }
  else
  {
    v12 = (__int64)a1;
    do
    {
      v13 = v12;
      v12 += 504;
      sub_192E490(v13, v12, v11, a4, a5, a6);
    }
    while ( a2 - v12 > 432 );
    sub_192E490(v12, a2, v11, a4, a5, a6);
    if ( v7 > 504 )
    {
      v16 = 7;
      do
      {
        sub_192E8E0(a1, a2, (__int64)a3, v16, v14, v15);
        v17 = 2 * v16;
        v16 *= 4;
        sub_192E8E0(a3, v10, (__int64)a1, v17, v18, v19);
      }
      while ( (__int64)(0x8E38E38E38E38E39LL * (v7 >> 3)) > v16 );
    }
  }
}
