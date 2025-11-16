// Function: sub_1D19350
// Address: 0x1d19350
//
_QWORD *__fastcall sub_1D19350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  _QWORD *v7; // r12
  int v9; // r10d
  __int64 v10; // r8
  int v11; // esi
  __int64 v12; // rsi
  __int64 v14; // [rsp+8h] [rbp-E8h]
  int v15; // [rsp+18h] [rbp-D8h]
  __int64 v16; // [rsp+20h] [rbp-D0h] BYREF
  int v17; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v18[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v19[176]; // [rsp+40h] [rbp-B0h] BYREF

  v7 = 0;
  if ( !(unsigned __int8)sub_1D12E90(a2) )
  {
    v10 = *(_QWORD *)(a2 + 40);
    v11 = *(unsigned __int16 *)(a2 + 24);
    v18[0] = (unsigned __int64)v19;
    v15 = v9;
    v18[1] = 0x2000000000LL;
    v14 = v10;
    sub_16BD430((__int64)v18, v11);
    sub_16BD4C0((__int64)v18, v14);
    sub_16BD4C0((__int64)v18, a3);
    sub_16BD430((__int64)v18, v15);
    sub_1D14B60((__int64)v18, a2);
    v12 = *(_QWORD *)(a2 + 72);
    v16 = v12;
    if ( v12 )
      sub_1623A60((__int64)&v16, v12, 2);
    v17 = *(_DWORD *)(a2 + 64);
    v7 = sub_1D17920(a1, (__int64)v18, (__int64)&v16, a5);
    if ( v16 )
      sub_161E7C0((__int64)&v16, v16);
    if ( v7 )
      sub_1D19330((__int64)v7, *(_WORD *)(a2 + 80));
    if ( (_BYTE *)v18[0] != v19 )
      _libc_free(v18[0]);
  }
  return v7;
}
