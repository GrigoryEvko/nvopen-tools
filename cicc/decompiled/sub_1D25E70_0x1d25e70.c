// Function: sub_1D25E70
// Address: 0x1d25e70
//
__int64 __fastcall sub_1D25E70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r13
  __int64 v10; // rsi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  __int64 v14; // r13
  _QWORD *v16; // rax
  char *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  char *v20; // r12
  __int64 v21; // rax
  int v22; // eax
  __int64 *v23; // rdx
  _QWORD *v26; // [rsp+28h] [rbp-E8h]
  __int64 v27; // [rsp+28h] [rbp-E8h]
  __int64 *v28; // [rsp+38h] [rbp-D8h] BYREF
  _QWORD v29[2]; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v30[2]; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE v31[176]; // [rsp+60h] [rbp-B0h] BYREF

  v8 = a2;
  v30[1] = 0x2000000000LL;
  v30[0] = (unsigned __int64)v31;
  sub_16BD430((__int64)v30, 3);
  a2 = (unsigned __int8)a2;
  if ( !(_BYTE)v8 )
    a2 = a3;
  sub_16BD4D0((__int64)v30, a2);
  v10 = (unsigned __int8)a4;
  if ( !(_BYTE)a4 )
    v10 = a5;
  sub_16BD4D0((__int64)v30, v10);
  v11 = (unsigned __int8)a7;
  if ( !(_BYTE)a7 )
    v11 = a8;
  sub_16BD4D0((__int64)v30, v11);
  v28 = 0;
  v12 = sub_16BDDE0(a1 + 672, (__int64)v30, (__int64 *)&v28);
  if ( v12 )
  {
    v13 = v12;
  }
  else
  {
    v16 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 544), 48, 8);
    *v16 = v8;
    v16[2] = a4;
    v16[1] = a3;
    v16[4] = a7;
    v16[3] = a5;
    v26 = v16;
    v16[5] = a8;
    v17 = sub_16BD760((__int64)v30, (__int64 *)(a1 + 544));
    v19 = v18;
    v20 = v17;
    v21 = sub_145CBF0((__int64 *)(a1 + 544), 40, 16);
    v29[0] = v20;
    *(_QWORD *)v21 = 0;
    *(_QWORD *)(v21 + 24) = v26;
    *(_QWORD *)(v21 + 8) = v20;
    *(_QWORD *)(v21 + 16) = v19;
    *(_DWORD *)(v21 + 32) = 3;
    v29[1] = v19;
    v27 = v21;
    v22 = sub_16BDD90((__int64)v29);
    v23 = v28;
    *(_DWORD *)(v27 + 36) = v22;
    sub_16BDA20((__int64 *)(a1 + 672), (__int64 *)v27, v23);
    v13 = (_QWORD *)v27;
  }
  v14 = v13[3];
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
  return v14;
}
