// Function: sub_348C2B0
// Address: 0x348c2b0
//
__int64 __fastcall sub_348C2B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        int a6,
        __int128 a7,
        __int64 *a8,
        __int64 a9)
{
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // r13
  __int64 *v12; // r12
  __int64 *v13; // r14
  __int64 v14; // rsi
  __int64 *v16; // [rsp+18h] [rbp-70h] BYREF
  __int64 v17; // [rsp+20h] [rbp-68h]
  _BYTE v18[96]; // [rsp+28h] [rbp-60h] BYREF

  v16 = (__int64 *)v18;
  v17 = 0x500000000LL;
  v9 = sub_348B620(a1, a2, a3, a4, a5, a6, a7, (__int64)a8, a9, (__int64)&v16);
  if ( v9 )
  {
    v10 = v16;
    v11 = v9;
    v12 = &v16[(unsigned int)v17];
    if ( v12 != v16 )
    {
      v13 = v16;
      do
      {
        v14 = *v13++;
        sub_32C2500(a8, v14);
      }
      while ( v12 != v13 );
      v10 = v16;
    }
  }
  else
  {
    v10 = v16;
    v11 = 0;
  }
  if ( v10 != (__int64 *)v18 )
    _libc_free((unsigned __int64)v10);
  return v11;
}
