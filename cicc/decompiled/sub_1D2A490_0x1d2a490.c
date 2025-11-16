// Function: sub_1D2A490
// Address: 0x1d2a490
//
_QWORD *__fastcall sub_1D2A490(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // r12
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 *v18; // rdx
  __int64 *v19; // [rsp+0h] [rbp-D0h] BYREF
  unsigned __int8 *v20; // [rsp+8h] [rbp-C8h] BYREF
  unsigned __int64 v21[2]; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE v22[176]; // [rsp+20h] [rbp-B0h] BYREF

  v21[0] = (unsigned __int64)v22;
  v21[1] = 0x2000000000LL;
  v7 = sub_1D29190((__int64)a1, 1u, 0, a4, a5, a6);
  sub_16BD430((__int64)v21, 5);
  sub_16BD4C0((__int64)v21, v7);
  sub_16BD4C0((__int64)v21, a2);
  v19 = 0;
  v8 = sub_1D17910((__int64)a1, (__int64)v21, (__int64 *)&v19);
  if ( v8 )
  {
    v13 = v8;
  }
  else
  {
    v15 = a1[26];
    if ( v15 )
      a1[26] = *(_QWORD *)v15;
    else
      v15 = sub_145CBF0(a1 + 27, 112, 8);
    v16 = sub_1D274F0(1u, v9, v10, v11, v12);
    v20 = 0;
    *(_QWORD *)v15 = 0;
    v17 = v20;
    *(_QWORD *)(v15 + 40) = v16;
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = 0;
    *(_WORD *)(v15 + 24) = 5;
    *(_DWORD *)(v15 + 28) = -1;
    *(_QWORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 48) = 0;
    *(_QWORD *)(v15 + 56) = 0x100000000LL;
    *(_DWORD *)(v15 + 64) = 0;
    *(_QWORD *)(v15 + 72) = v17;
    if ( v17 )
      sub_1623210((__int64)&v20, v17, v15 + 72);
    *(_QWORD *)(v15 + 88) = a2;
    v18 = v19;
    *(_WORD *)(v15 + 80) &= 0xF000u;
    v13 = (_QWORD *)v15;
    *(_WORD *)(v15 + 26) = 0;
    sub_16BDA20(a1 + 40, (__int64 *)v15, v18);
    sub_1D172A0((__int64)a1, v15);
  }
  if ( (_BYTE *)v21[0] != v22 )
    _libc_free(v21[0]);
  return v13;
}
