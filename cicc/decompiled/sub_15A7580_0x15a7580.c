// Function: sub_15A7580
// Address: 0x15a7580
//
__int64 __fastcall sub_15A7580(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r10
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r12
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-D0h]
  __int64 v26; // [rsp+8h] [rbp-C8h]
  _QWORD v27[4]; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE v28[16]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v29; // [rsp+40h] [rbp-90h]
  __int64 v30[16]; // [rsp+50h] [rbp-80h] BYREF

  v7 = a2;
  if ( !a1[4] )
  {
    v24 = sub_15E26F0(*a1, 38, 0, 0);
    v7 = a2;
    a1[4] = v24;
  }
  v25 = v7;
  sub_15A6B80((__int64)a1, a3);
  sub_15A6B80((__int64)a1, a4);
  v26 = a1[1];
  v15 = sub_1624210(v25, a4, v13, v14);
  v16 = sub_1628DA0(v26, v15);
  v17 = a1[1];
  v27[0] = v16;
  v18 = sub_1628DA0(v17, a3);
  v19 = a1[1];
  v27[1] = v18;
  v27[2] = sub_1628DA0(v19, a4);
  sub_15A51A0((__int64)v30, a5, a6, a7);
  v20 = a1[4];
  v21 = *(_QWORD *)(v20 + 24);
  v29 = 257;
  v22 = sub_1285290(v30, v21, v20, (int)v27, 3, (__int64)v28, 0);
  if ( v30[0] )
    sub_161E7C0(v30);
  return v22;
}
