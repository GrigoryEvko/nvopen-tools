// Function: sub_2048150
// Address: 0x2048150
//
_QWORD *__fastcall sub_2048150(__int64 a1, unsigned int a2, __int64 a3, double a4, double a5, __m128i a6)
{
  void *v7; // r12
  void *v8; // rax
  void *v9; // rbx
  _QWORD *v10; // r12
  __int64 v12; // r14
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+8h] [rbp-58h]
  char v17[8]; // [rsp+10h] [rbp-50h] BYREF
  void *v18; // [rsp+18h] [rbp-48h] BYREF
  __int64 v19; // [rsp+20h] [rbp-40h]

  v16 = 32;
  v15 = a2;
  v7 = sub_1698270();
  v8 = sub_16982C0();
  v9 = v8;
  if ( v7 == v8 )
    sub_169D060(&v18, (__int64)v8, &v15);
  else
    sub_169D050((__int64)&v18, v7, &v15);
  v10 = sub_1D36490(a1, (__int64)v17, a3, 9u, 0, 0, a4, a5, a6);
  if ( v18 == v9 )
  {
    v12 = v19;
    if ( v19 )
    {
      v13 = 32LL * *(_QWORD *)(v19 - 8);
      v14 = v19 + v13;
      if ( v19 != v19 + v13 )
      {
        do
        {
          v14 -= 32;
          sub_127D120((_QWORD *)(v14 + 8));
        }
        while ( v12 != v14 );
      }
      j_j_j___libc_free_0_0(v12 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v18);
  }
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return v10;
}
