// Function: sub_1D2A890
// Address: 0x1d2a890
//
_QWORD *__fastcall sub_1D2A890(_QWORD *a1, unsigned __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // r12
  __int64 v14; // rbx
  int v15; // r12d
  __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  __int64 v20; // [rsp+8h] [rbp-E8h]
  __int64 v21; // [rsp+8h] [rbp-E8h]
  __int64 *v22; // [rsp+10h] [rbp-E0h] BYREF
  unsigned __int8 *v23; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v24; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v26[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v27[176]; // [rsp+40h] [rbp-B0h] BYREF

  v24 = a4;
  v25 = a5;
  v26[0] = (unsigned __int64)v27;
  v26[1] = 0x2000000000LL;
  v20 = sub_1D29190((__int64)a1, 1u, 0, a4, a5, a6);
  sub_16BD430((__int64)v26, a2);
  sub_16BD4C0((__int64)v26, v20);
  sub_16BD4C0((__int64)v26, v24);
  sub_16BD430((__int64)v26, v25);
  sub_16BD4C0((__int64)v26, a6);
  v22 = 0;
  v7 = sub_1D17910((__int64)a1, (__int64)v26, (__int64 *)&v22);
  if ( v7 )
  {
    v12 = v7;
  }
  else
  {
    v14 = a1[26];
    v15 = *(_DWORD *)(a3 + 8);
    if ( v14 )
      a1[26] = *(_QWORD *)v14;
    else
      v14 = sub_145CBF0(a1 + 27, 112, 8);
    v16 = sub_1D274F0(1u, v8, v9, v10, v11);
    v17 = *(_QWORD *)a3;
    v23 = (unsigned __int8 *)v17;
    if ( v17 )
    {
      v21 = v16;
      sub_1623A60((__int64)&v23, v17, 2);
      v16 = v21;
    }
    *(_QWORD *)v14 = 0;
    v18 = v23;
    *(_QWORD *)(v14 + 8) = 0;
    *(_QWORD *)(v14 + 16) = 0;
    *(_WORD *)(v14 + 24) = 194;
    *(_DWORD *)(v14 + 28) = -1;
    *(_QWORD *)(v14 + 32) = 0;
    *(_QWORD *)(v14 + 40) = v16;
    *(_QWORD *)(v14 + 48) = 0;
    *(_QWORD *)(v14 + 56) = 0x100000000LL;
    *(_DWORD *)(v14 + 64) = v15;
    *(_QWORD *)(v14 + 72) = v18;
    if ( v18 )
      sub_1623210((__int64)&v23, v18, v14 + 72);
    *(_WORD *)(v14 + 80) &= 0xF000u;
    *(_WORD *)(v14 + 26) = 0;
    v12 = (_QWORD *)v14;
    *(_QWORD *)(v14 + 88) = a6;
    sub_1D23B60((__int64)a1, v14, (__int64)&v24, 1);
    sub_16BDA20(a1 + 40, (__int64 *)v14, v22);
    sub_1D172A0((__int64)a1, v14);
  }
  if ( (_BYTE *)v26[0] != v27 )
    _libc_free(v26[0]);
  return v12;
}
