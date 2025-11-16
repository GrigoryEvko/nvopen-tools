// Function: sub_193F750
// Address: 0x193f750
//
__int64 __fastcall sub_193F750(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v3; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v4; // [rsp+8h] [rbp-78h]
  _QWORD *v5; // [rsp+10h] [rbp-70h]
  __int64 v6; // [rsp+18h] [rbp-68h]
  int v7; // [rsp+20h] [rbp-60h]
  _QWORD v8[10]; // [rsp+28h] [rbp-58h] BYREF

  v4 = v8;
  v5 = v8;
  v8[0] = a1;
  v6 = 0x100000008LL;
  v7 = 0;
  v3 = 1;
  v1 = sub_193F5F0(a1, (__int64)&v3, 0);
  if ( v5 != v4 )
    _libc_free((unsigned __int64)v5);
  return v1;
}
