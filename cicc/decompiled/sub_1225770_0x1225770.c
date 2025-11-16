// Function: sub_1225770
// Address: 0x1225770
//
__int64 __fastcall sub_1225770(__int64 **a1, _QWORD *a2, char a3)
{
  __int64 **v5; // rsi
  unsigned int v6; // r12d
  __int64 (__fastcall *v7)(__int64 *, __int64 *, __int64 *); // rax
  __int64 **v9; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v10; // [rsp+8h] [rbp-B8h]
  _BYTE v11[176]; // [rsp+10h] [rbp-B0h] BYREF

  v5 = (__int64 **)&v9;
  v9 = (__int64 **)v11;
  v10 = 0x1000000000LL;
  v6 = sub_1225600((__int64)a1, (__int64)&v9);
  if ( !(_BYTE)v6 )
  {
    if ( a3 )
      v7 = sub_1205530;
    else
      v7 = sub_1205520;
    v5 = v9;
    *a2 = v7(*a1, (__int64 *)v9, (__int64 *)(unsigned int)v10);
  }
  if ( v9 != (__int64 **)v11 )
    _libc_free(v9, v5);
  return v6;
}
