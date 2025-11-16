// Function: sub_33E33B0
// Address: 0x33e33b0
//
_QWORD *__fastcall sub_33E33B0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7)
{
  _QWORD *v9; // r13
  __int64 v11; // r9
  unsigned __int64 v12; // r10
  unsigned __int64 v13; // rdx
  int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rsi
  __int64 v21; // [rsp+0h] [rbp-F0h] BYREF
  int v22; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v23[4]; // [rsp+10h] [rbp-E0h] BYREF
  unsigned __int64 v24[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v25[176]; // [rsp+40h] [rbp-B0h] BYREF

  v9 = 0;
  if ( !(unsigned __int8)sub_33C7D40(a2) )
  {
    v13 = *(_QWORD *)(a2 + 48);
    v14 = *(_DWORD *)(a2 + 24);
    v23[0] = a3;
    v23[1] = a4;
    v23[2] = v12;
    v23[3] = v11;
    v24[1] = 0x2000000000LL;
    v24[0] = (unsigned __int64)v25;
    sub_33C9670((__int64)v24, v14, v13, v23, 2, v11);
    sub_33E2C00((__int64)v24, a2, v15, v16, v17, v18);
    v19 = *(_QWORD *)(a2 + 80);
    v21 = v19;
    if ( v19 )
      sub_B96E90((__int64)&v21, v19, 1);
    v22 = *(_DWORD *)(a2 + 72);
    v9 = sub_33CCCF0(a1, (__int64)v24, (__int64)&v21, a7);
    if ( v21 )
      sub_B91220((__int64)&v21, v21);
    if ( v9 )
      sub_33D00A0((__int64)v9, *(_DWORD *)(a2 + 28));
    if ( (_BYTE *)v24[0] != v25 )
      _libc_free(v24[0]);
  }
  return v9;
}
