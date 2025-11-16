// Function: sub_13CAFA0
// Address: 0x13cafa0
//
__int64 __fastcall sub_13CAFA0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // [rsp+0h] [rbp-C0h] BYREF
  _BYTE *v5; // [rsp+8h] [rbp-B8h]
  _BYTE *v6; // [rsp+10h] [rbp-B0h]
  __int64 v7; // [rsp+18h] [rbp-A8h]
  int v8; // [rsp+20h] [rbp-A0h]
  _BYTE v9[144]; // [rsp+28h] [rbp-98h] BYREF

  v5 = v9;
  v4 = 0;
  v6 = v9;
  v7 = 16;
  v8 = 0;
  v2 = sub_13CA600(a1, a2, (__int64)&v4);
  if ( v6 != v5 )
    _libc_free((unsigned __int64)v6);
  return v2;
}
