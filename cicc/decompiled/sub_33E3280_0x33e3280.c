// Function: sub_33E3280
// Address: 0x33e3280
//
_QWORD *__fastcall sub_33E3280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  _QWORD *v5; // r14
  __int64 v8; // r9
  unsigned __int64 v9; // r10
  unsigned __int64 v10; // r11
  unsigned __int64 v11; // rdx
  int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rsi
  unsigned __int64 v19[2]; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+20h] [rbp-D0h] BYREF
  int v21; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v22[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v23[176]; // [rsp+40h] [rbp-B0h] BYREF

  v5 = 0;
  if ( !(unsigned __int8)sub_33C7D40(a2) )
  {
    v11 = *(_QWORD *)(a2 + 48);
    v12 = *(_DWORD *)(a2 + 24);
    v22[0] = (unsigned __int64)v23;
    v19[0] = v9;
    v19[1] = v10;
    v22[1] = 0x2000000000LL;
    sub_33C9670((__int64)v22, v12, v11, v19, 1, v8);
    sub_33E2C00((__int64)v22, a2, v13, v14, v15, v16);
    v17 = *(_QWORD *)(a2 + 80);
    v20 = v17;
    if ( v17 )
      sub_B96E90((__int64)&v20, v17, 1);
    v21 = *(_DWORD *)(a2 + 72);
    v5 = sub_33CCCF0(a1, (__int64)v22, (__int64)&v20, a5);
    if ( v20 )
      sub_B91220((__int64)&v20, v20);
    if ( v5 )
      sub_33D00A0((__int64)v5, *(_DWORD *)(a2 + 28));
    if ( (_BYTE *)v22[0] != v23 )
      _libc_free(v22[0]);
  }
  return v5;
}
