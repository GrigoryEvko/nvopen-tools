// Function: sub_1C8BF90
// Address: 0x1c8bf90
//
unsigned __int64 __fastcall sub_1C8BF90(
        _BYTE *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // r11
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  unsigned __int64 result; // rax
  __int64 v27; // rsi
  __int64 v28; // [rsp+8h] [rbp-C8h]
  __int64 v30; // [rsp+28h] [rbp-A8h] BYREF
  unsigned __int8 *v31[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v32[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v33[5]; // [rsp+50h] [rbp-80h] BYREF
  int v34; // [rsp+78h] [rbp-58h]
  __int64 v35; // [rsp+80h] [rbp-50h]
  __int64 v36; // [rsp+88h] [rbp-48h]

  v15 = sub_16498A0((__int64)a2);
  v16 = (unsigned __int8 *)a2[6];
  v33[0] = 0;
  v33[3] = v15;
  v17 = (__int64)a2[5];
  v33[4] = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v33[1] = v17;
  v33[2] = (__int64)(a2 + 3);
  v31[0] = v16;
  if ( v16 )
  {
    sub_1623A60((__int64)v31, (__int64)v16, 2);
    if ( v33[0] )
      sub_161E7C0((__int64)v33, v33[0]);
    v33[0] = (__int64)v31[0];
    if ( v31[0] )
      sub_1623210((__int64)v31, v31[0], (__int64)v33);
    v17 = (__int64)a2[5];
  }
  v18 = *a2;
  v19 = *(_QWORD *)(*(_QWORD *)(v17 + 56) + 40LL);
  v20 = **(a2 - 3);
  v30 = (__int64)*(a2 - 3);
  v31[0] = (unsigned __int8 *)v32;
  v32[0] = v20;
  v28 = v19;
  v31[1] = (unsigned __int8 *)0x100000001LL;
  v21 = sub_1644EA0(v18, v32, 1, 0);
  v22 = sub_1632080(v28, a3, a4, v21, 0);
  if ( (_QWORD *)v31[0] != v32 )
    _libc_free((unsigned __int64)v31[0]);
  LOWORD(v32[0]) = 257;
  v23 = sub_1285290(v33, *(_QWORD *)(*(_QWORD *)v22 + 24LL), v22, (int)&v30, 1, (__int64)v31, 0);
  sub_164D160((__int64)a2, v23, a5, a6, a7, a8, v24, v25, a11, a12);
  sub_15F20C0(a2);
  result = (unsigned __int64)a1;
  v27 = v33[0];
  *a1 = 1;
  if ( v27 )
    return sub_161E7C0((__int64)v33, v27);
  return result;
}
