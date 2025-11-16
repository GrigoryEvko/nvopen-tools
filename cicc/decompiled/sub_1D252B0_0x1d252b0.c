// Function: sub_1D252B0
// Address: 0x1d252b0
//
__int64 __fastcall sub_1D252B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // r13
  _QWORD *v13; // r9
  __int64 v14; // rax
  char *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  char *v18; // r12
  __int64 v19; // rax
  int v20; // eax
  __int64 *v21; // rdx
  _QWORD *v24; // [rsp+18h] [rbp-E8h]
  __int64 v25; // [rsp+18h] [rbp-E8h]
  __int64 *v26; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD v27[2]; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 v28[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v29[176]; // [rsp+50h] [rbp-B0h] BYREF

  v6 = a2;
  v28[0] = (unsigned __int64)v29;
  v28[1] = 0x2000000000LL;
  sub_16BD430((__int64)v28, 2);
  a2 = (unsigned __int8)a2;
  if ( !(_BYTE)v6 )
    a2 = a3;
  sub_16BD4D0((__int64)v28, a2);
  v8 = (unsigned __int8)a4;
  if ( !(_BYTE)a4 )
    v8 = a5;
  sub_16BD4D0((__int64)v28, v8);
  v26 = 0;
  v9 = sub_16BDDE0(a1 + 672, (__int64)v28, (__int64 *)&v26);
  if ( v9 )
  {
    v10 = v9;
  }
  else
  {
    v13 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 544), 32, 8);
    *v13 = v6;
    v13[2] = a4;
    v13[1] = a3;
    v14 = a5;
    v24 = v13;
    v13[3] = v14;
    v15 = sub_16BD760((__int64)v28, (__int64 *)(a1 + 544));
    v17 = v16;
    v18 = v15;
    v19 = sub_145CBF0((__int64 *)(a1 + 544), 40, 16);
    v27[0] = v18;
    *(_QWORD *)v19 = 0;
    *(_QWORD *)(v19 + 24) = v24;
    *(_QWORD *)(v19 + 8) = v18;
    *(_QWORD *)(v19 + 16) = v17;
    *(_DWORD *)(v19 + 32) = 2;
    v27[1] = v17;
    v25 = v19;
    v20 = sub_16BDD90((__int64)v27);
    v21 = v26;
    *(_DWORD *)(v25 + 36) = v20;
    sub_16BDA20((__int64 *)(a1 + 672), (__int64 *)v25, v21);
    v10 = (_QWORD *)v25;
  }
  v11 = v10[3];
  if ( (_BYTE *)v28[0] != v29 )
    _libc_free(v28[0]);
  return v11;
}
