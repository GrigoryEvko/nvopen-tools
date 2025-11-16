// Function: sub_1675920
// Address: 0x1675920
//
__int64 __fastcall sub_1675920(__int64 a1, _QWORD **a2)
{
  __int64 v2; // r12
  __int64 v4; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v5; // [rsp+8h] [rbp-78h]
  _BYTE *v6; // [rsp+10h] [rbp-70h]
  __int64 v7; // [rsp+18h] [rbp-68h]
  int v8; // [rsp+20h] [rbp-60h]
  _BYTE v9[80]; // [rsp+28h] [rbp-58h] BYREF

  v4 = 0;
  v5 = v9;
  v6 = v9;
  v7 = 8;
  v8 = 0;
  v2 = sub_1674800(a1, a2, (__int64)&v4);
  if ( v6 != v5 )
    _libc_free((unsigned __int64)v6);
  return v2;
}
