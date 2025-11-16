// Function: sub_1D65F00
// Address: 0x1d65f00
//
__int64 __fastcall sub_1D65F00(
        _QWORD *a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned int v10; // eax
  unsigned int v11; // ebx
  bool v12; // zf
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rcx
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 *v20; // r14
  _QWORD *v21; // r15
  __int64 v22; // rax
  _QWORD *v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // r11
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rdi
  double v30; // xmm4_8
  double v31; // xmm5_8
  __int64 v33; // [rsp+10h] [rbp-90h]
  _QWORD *v34; // [rsp+10h] [rbp-90h]
  _QWORD *v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  __int64 v37; // [rsp+20h] [rbp-80h] BYREF
  __int64 v38; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v39; // [rsp+30h] [rbp-70h] BYREF
  __int64 v40; // [rsp+38h] [rbp-68h] BYREF
  __int64 v41[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v42[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD **v43; // [rsp+60h] [rbp-40h]

  v42[0] = &v37;
  v42[1] = &v38;
  v43 = &v39;
  v10 = sub_1D65D80(v42, (__int64)a1);
  if ( (_BYTE)v10
    && (v11 = v10, v12 = *(_BYTE *)(*v39 + 8LL) == 11, v40 = *v39, v12)
    && (a1[5] == v39[5] || (v13 = v39[1]) != 0 && !*(_QWORD *)(v13 + 8)) )
  {
    v14 = (__int64 *)sub_15F2050((__int64)a1);
    v15 = sub_15E26F0(v14, 209, &v40, 1);
    v16 = v39;
    v17 = v15;
    v18 = v39[1];
    v35 = v39;
    if ( v18 )
    {
      if ( !*(_QWORD *)(v18 + 8) )
        v16 = a1;
      v35 = v16;
    }
    LOWORD(v43) = 259;
    v42[0] = "uadd.overflow";
    v41[0] = v37;
    v41[1] = v38;
    v33 = *(_QWORD *)(*(_QWORD *)v17 + 24LL);
    v19 = sub_1648AB0(72, 3u, 0);
    v20 = v19;
    if ( v19 )
    {
      sub_15F1EA0((__int64)v19, **(_QWORD **)(v33 + 16), 54, (__int64)(v19 - 9), 3, (__int64)v35);
      v20[7] = 0;
      sub_15F5B40((__int64)v20, v33, v17, v41, 2, (__int64)v42, 0, 0);
    }
    v42[0] = "uadd";
    LOWORD(v43) = 259;
    LODWORD(v41[0]) = 0;
    v21 = sub_1648A60(88, 1u);
    if ( v21 )
    {
      v22 = sub_15FB2A0(*v20, (unsigned int *)v41, 1);
      sub_15F1EA0((__int64)v21, v22, 62, (__int64)(v21 - 3), 1, (__int64)v35);
      sub_1593B40(v21 - 3, (__int64)v20);
      v21[7] = v21 + 9;
      v21[8] = 0x400000000LL;
      sub_15FB110((__int64)v21, v41, 1, (__int64)v42);
    }
    v42[0] = "overflow";
    LOWORD(v43) = 259;
    LODWORD(v41[0]) = 1;
    v23 = sub_1648A60(88, 1u);
    v26 = (__int64)v23;
    if ( v23 )
    {
      v34 = v23;
      v27 = sub_15FB2A0(*v20, (unsigned int *)v41, 1);
      v28 = (__int64)v35;
      v29 = (__int64)v34;
      v36 = (__int64)v34;
      v34 -= 3;
      sub_15F1EA0(v29, v27, 62, (__int64)v34, 1, v28);
      sub_1593B40(v34, (__int64)v20);
      *(_QWORD *)(v36 + 56) = v36 + 72;
      *(_QWORD *)(v36 + 64) = 0x400000000LL;
      sub_15FB110(v36, v41, 1, (__int64)v42);
      v26 = v36;
    }
    sub_164D160((__int64)a1, v26, a2, a3, a4, a5, v24, v25, a8, a9);
    sub_164D160((__int64)v39, (__int64)v21, a2, a3, a4, a5, v30, v31, a8, a9);
    sub_15F20C0(a1);
    sub_15F20C0(v39);
  }
  else
  {
    return 0;
  }
  return v11;
}
