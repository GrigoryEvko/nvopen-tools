// Function: sub_348D320
// Address: 0x348d320
//
__int64 __fastcall sub_348D320(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // r13
  __int64 *v13; // r12
  __int64 *v14; // r14
  __int64 v15; // rsi
  __int64 *v17; // [rsp+18h] [rbp-80h] BYREF
  __int64 v18; // [rsp+20h] [rbp-78h]
  _BYTE v19[112]; // [rsp+28h] [rbp-70h] BYREF

  v17 = (__int64 *)v19;
  v18 = 0x700000000LL;
  v10 = sub_348C370(a1, a2, a3, a4, a5, a6, a7, a8, (__int64)a9, a10, (__int64)&v17);
  if ( v10 )
  {
    v11 = v17;
    v12 = v10;
    v13 = &v17[(unsigned int)v18];
    if ( v13 != v17 )
    {
      v14 = v17;
      do
      {
        v15 = *v14++;
        sub_32C2500(a9, v15);
      }
      while ( v13 != v14 );
      v11 = v17;
    }
  }
  else
  {
    v11 = v17;
    v12 = 0;
  }
  if ( v11 != (__int64 *)v19 )
    _libc_free((unsigned __int64)v11);
  return v12;
}
