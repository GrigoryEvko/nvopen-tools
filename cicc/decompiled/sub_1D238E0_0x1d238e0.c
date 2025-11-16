// Function: sub_1D238E0
// Address: 0x1d238e0
//
__int64 __fastcall sub_1D238E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // r13
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rsi
  _QWORD *v15; // rsi
  __int64 v16; // r13
  _QWORD *v18; // rax
  _QWORD *v19; // r14
  char *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r13
  char *v23; // r12
  __int64 v24; // rax
  int v25; // eax
  __int64 *v26; // rdx
  __int64 v29; // [rsp+38h] [rbp-E8h]
  __int64 *v30; // [rsp+48h] [rbp-D8h] BYREF
  _QWORD v31[2]; // [rsp+50h] [rbp-D0h] BYREF
  unsigned __int64 v32[2]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE v33[176]; // [rsp+70h] [rbp-B0h] BYREF

  v10 = a2;
  v32[1] = 0x2000000000LL;
  v32[0] = (unsigned __int64)v33;
  sub_16BD430((__int64)v32, 4);
  a2 = (unsigned __int8)a2;
  if ( !(_BYTE)v10 )
    a2 = a3;
  sub_16BD4D0((__int64)v32, a2);
  v12 = (unsigned __int8)a4;
  if ( !(_BYTE)a4 )
    v12 = a5;
  sub_16BD4D0((__int64)v32, v12);
  v13 = (unsigned __int8)a7;
  if ( !(_BYTE)a7 )
    v13 = a8;
  sub_16BD4D0((__int64)v32, v13);
  v14 = (unsigned __int8)a9;
  if ( !(_BYTE)a9 )
    v14 = a10;
  sub_16BD4D0((__int64)v32, v14);
  v30 = 0;
  v15 = sub_16BDDE0(a1 + 672, (__int64)v32, (__int64 *)&v30);
  if ( !v15 )
  {
    v18 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 544), 64, 8);
    *v18 = v10;
    v19 = v18;
    v18[2] = a4;
    v18[1] = a3;
    v18[4] = a7;
    v18[3] = a5;
    v18[5] = a8;
    v18[6] = a9;
    v18[7] = a10;
    v20 = sub_16BD760((__int64)v32, (__int64 *)(a1 + 544));
    v22 = v21;
    v23 = v20;
    v24 = sub_145CBF0((__int64 *)(a1 + 544), 40, 16);
    v31[0] = v23;
    *(_QWORD *)v24 = 0;
    *(_QWORD *)(v24 + 8) = v23;
    *(_QWORD *)(v24 + 16) = v22;
    *(_QWORD *)(v24 + 24) = v19;
    *(_DWORD *)(v24 + 32) = 4;
    v31[1] = v22;
    v29 = v24;
    v25 = sub_16BDD90((__int64)v31);
    v26 = v30;
    *(_DWORD *)(v29 + 36) = v25;
    sub_16BDA20((__int64 *)(a1 + 672), (__int64 *)v29, v26);
    v15 = (_QWORD *)v29;
  }
  v16 = v15[3];
  if ( (_BYTE *)v32[0] != v33 )
    _libc_free(v32[0]);
  return v16;
}
