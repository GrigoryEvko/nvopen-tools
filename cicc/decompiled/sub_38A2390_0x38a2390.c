// Function: sub_38A2390
// Address: 0x38a2390
//
__int64 __fastcall sub_38A2390(__int64 **a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  unsigned int v7; // r12d
  __int64 v8; // rax
  __int64 *v10; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v11; // [rsp+8h] [rbp-B8h]
  _BYTE v12[176]; // [rsp+10h] [rbp-B0h] BYREF

  v10 = (__int64 *)v12;
  v11 = 0x1000000000LL;
  v7 = sub_38A2250((__int64)a1, (__int64)&v10, a4, a5, a6);
  if ( !(_BYTE)v7 )
  {
    if ( a3 )
      v8 = sub_3887400(*a1, v10, (__int64 *)(unsigned int)v11);
    else
      v8 = sub_38873F0(*a1, v10, (__int64 *)(unsigned int)v11);
    *a2 = v8;
  }
  if ( v10 != (__int64 *)v12 )
    _libc_free((unsigned __int64)v10);
  return v7;
}
