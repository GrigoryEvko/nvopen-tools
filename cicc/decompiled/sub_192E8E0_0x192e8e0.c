// Function: sub_192E8E0
// Address: 0x192e8e0
//
__int64 __fastcall sub_192E8E0(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  char *v9; // rbx
  __int64 v10; // r14
  __int64 v11; // r15
  char *v12; // rdi

  v7 = 0x8E38E38E38E38E39LL * ((a2 - (__int64)a1) >> 3);
  v8 = 2 * a4;
  v9 = a1;
  if ( 2 * a4 <= v7 )
  {
    v10 = 72 * a4;
    v11 = 144 * a4;
    do
    {
      v12 = v9;
      v9 += v11;
      a3 = sub_192E6F0(v12, &v9[v10 - v11], (__int64)&v9[v10 - v11], (__int64)v9, a3, a6);
      v7 = 0x8E38E38E38E38E39LL * ((a2 - (__int64)v9) >> 3);
    }
    while ( v8 <= v7 );
  }
  if ( a4 <= v7 )
    v7 = a4;
  return sub_192E6F0(v9, &v9[72 * v7], (__int64)&v9[72 * v7], a2, a3, a6);
}
