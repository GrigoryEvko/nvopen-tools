// Function: sub_2F8B260
// Address: 0x2f8b260
//
__int64 __fastcall sub_2F8B260(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 *v12; // rdi

  v7 = 2 * a4;
  v8 = a1;
  v9 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - (__int64)a1) >> 3);
  if ( 2 * a4 <= v9 )
  {
    v10 = 176 * a4;
    v11 = 88 * a4;
    do
    {
      v12 = v8;
      v8 = (__int64 *)((char *)v8 + v10);
      a3 = sub_2F8B090(v12, (__int64 *)((char *)v8 + v11 - v10), (__int64)v8 + v11 - v10, (__int64)v8, a3, a6);
      v9 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - (__int64)v8) >> 3);
    }
    while ( v7 <= v9 );
  }
  if ( a4 <= v9 )
    v9 = a4;
  return sub_2F8B090(v8, &v8[11 * v9], (__int64)&v8[11 * v9], a2, a3, a6);
}
