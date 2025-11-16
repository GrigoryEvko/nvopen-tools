// Function: sub_33E34E0
// Address: 0x33e34e0
//
_QWORD *__fastcall sub_33E34E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  _QWORD *v6; // r12
  __int64 v8; // r9
  unsigned __int64 *v9; // r10
  int v10; // esi
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v18; // [rsp+10h] [rbp-D0h] BYREF
  int v19; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v20[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v21[176]; // [rsp+30h] [rbp-B0h] BYREF

  v6 = 0;
  if ( !(unsigned __int8)sub_33C7D40(a2) )
  {
    v10 = *(_DWORD *)(a2 + 24);
    v11 = *(_QWORD *)(a2 + 48);
    v20[0] = (unsigned __int64)v21;
    v20[1] = 0x2000000000LL;
    sub_33C9670((__int64)v20, v10, v11, v9, v8, v8);
    sub_33E2C00((__int64)v20, a2, v12, v13, v14, v15);
    v16 = *(_QWORD *)(a2 + 80);
    v18 = v16;
    if ( v16 )
      sub_B96E90((__int64)&v18, v16, 1);
    v19 = *(_DWORD *)(a2 + 72);
    v6 = sub_33CCCF0(a1, (__int64)v20, (__int64)&v18, a5);
    if ( v18 )
      sub_B91220((__int64)&v18, v18);
    if ( v6 )
      sub_33D00A0((__int64)v6, *(_DWORD *)(a2 + 28));
    if ( (_BYTE *)v20[0] != v21 )
      _libc_free(v20[0]);
  }
  return v6;
}
