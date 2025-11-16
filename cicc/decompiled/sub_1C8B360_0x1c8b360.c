// Function: sub_1C8B360
// Address: 0x1c8b360
//
unsigned __int64 __fastcall sub_1C8B360(
        _BYTE *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  unsigned __int64 result; // rax
  __int64 v28; // rsi
  __int64 *v29; // [rsp+0h] [rbp-F0h]
  _QWORD **v30; // [rsp+8h] [rbp-E8h]
  _QWORD v32[4]; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 *v33[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v34[4]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v35[5]; // [rsp+70h] [rbp-80h] BYREF
  int v36; // [rsp+98h] [rbp-58h]
  __int64 v37; // [rsp+A0h] [rbp-50h]
  __int64 v38; // [rsp+A8h] [rbp-48h]

  v16 = sub_16498A0((__int64)a2);
  v17 = (unsigned __int8 *)a2[6];
  v35[0] = 0;
  v35[3] = v16;
  v18 = (__int64)a2[5];
  v35[4] = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v35[1] = v18;
  v35[2] = (__int64)(a2 + 3);
  v33[0] = v17;
  if ( v17 )
  {
    sub_1623A60((__int64)v33, (__int64)v17, 2);
    if ( v35[0] )
      sub_161E7C0((__int64)v35, v35[0]);
    v35[0] = (__int64)v33[0];
    if ( v33[0] )
      sub_1623210((__int64)v33, v33[0], (__int64)v35);
    v18 = (__int64)a2[5];
  }
  v29 = *a2;
  v30 = *(_QWORD ***)(*(_QWORD *)(v18 + 56) + 40LL);
  v19 = sub_159CCF0(*v30, a5);
  v20 = (__int64)*(a2 - 6);
  v32[2] = v19;
  v32[0] = v20;
  v21 = (__int64)*(a2 - 3);
  v34[0] = v29;
  v32[1] = v21;
  v34[1] = v29;
  v34[2] = v29;
  v33[0] = (unsigned __int8 *)v34;
  v33[1] = (unsigned __int8 *)0x300000003LL;
  v22 = sub_1644EA0(v29, v34, 3, 0);
  v23 = sub_1632080((__int64)v30, a3, a4, v22, 0);
  if ( (_QWORD *)v33[0] != v34 )
    _libc_free((unsigned __int64)v33[0]);
  LOWORD(v34[0]) = 257;
  v24 = sub_1285290(v35, *(_QWORD *)(*(_QWORD *)v23 + 24LL), v23, (int)v32, 3, (__int64)v33, 0);
  sub_164D160((__int64)a2, v24, a6, a7, a8, a9, v25, v26, a12, a13);
  sub_15F20C0(a2);
  result = (unsigned __int64)a1;
  v28 = v35[0];
  *a1 = 1;
  if ( v28 )
    return sub_161E7C0((__int64)v35, v28);
  return result;
}
