// Function: sub_1D194A0
// Address: 0x1d194a0
//
_QWORD *__fastcall sub_1D194A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  _QWORD *v9; // r13
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // r13
  int v14; // esi
  __int64 v15; // rsi
  __int64 v17; // [rsp+10h] [rbp-F0h] BYREF
  int v18; // [rsp+18h] [rbp-E8h]
  __int64 v19; // [rsp+20h] [rbp-E0h]
  __int64 v20; // [rsp+28h] [rbp-D8h]
  __int64 v21; // [rsp+30h] [rbp-D0h]
  __int64 v22; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v23[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v24[176]; // [rsp+50h] [rbp-B0h] BYREF

  v9 = 0;
  if ( !(unsigned __int8)sub_1D12E90(a2) )
  {
    v19 = a3;
    v13 = *(_QWORD *)(a2 + 40);
    v14 = *(unsigned __int16 *)(a2 + 24);
    v23[0] = (unsigned __int64)v24;
    v21 = v12;
    v22 = v11;
    v23[1] = 0x2000000000LL;
    v20 = a4;
    sub_16BD430((__int64)v23, v14);
    sub_16BD4C0((__int64)v23, v13);
    sub_16BD4C0((__int64)v23, a3);
    sub_16BD430((__int64)v23, a4);
    sub_16BD4C0((__int64)v23, v21);
    sub_16BD430((__int64)v23, v22);
    sub_1D14B60((__int64)v23, a2);
    v15 = *(_QWORD *)(a2 + 72);
    v17 = v15;
    if ( v15 )
      sub_1623A60((__int64)&v17, v15, 2);
    v18 = *(_DWORD *)(a2 + 64);
    v9 = sub_1D17920(a1, (__int64)v23, (__int64)&v17, a7);
    if ( v17 )
      sub_161E7C0((__int64)&v17, v17);
    if ( v9 )
      sub_1D19330((__int64)v9, *(_WORD *)(a2 + 80));
    if ( (_BYTE *)v23[0] != v24 )
      _libc_free(v23[0]);
  }
  return v9;
}
