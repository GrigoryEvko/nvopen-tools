// Function: sub_1948FD0
// Address: 0x1948fd0
//
__int64 __fastcall sub_1948FD0(
        __int64 a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // r12
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  __int64 v27; // [rsp+38h] [rbp-48h]
  __int64 v28; // [rsp+40h] [rbp-40h]

  v9 = (__int64 *)sub_157E9C0(**(_QWORD **)(a1 + 32));
  v21 = sub_1627350(v9, 0, 0, 0, 1);
  v24 = sub_161FF10(v9, "llvm.loop.unroll.disable", 0x18u);
  v22 = sub_1627350(v9, &v24, (__int64 *)1, 0, 1);
  v10 = sub_1643320(v9);
  v11 = sub_159C470(v10, 0, 0);
  v12 = sub_1624210(v11);
  v13 = sub_161FF10(v9, "llvm.loop.vectorize.enable", 0x1Au);
  v25 = v12;
  v24 = v13;
  v23 = sub_1627350(v9, &v24, (__int64 *)2, 0, 1);
  v24 = sub_161FF10(v9, "llvm.loop.licm_versioning.disable", 0x21u);
  v14 = sub_1627350(v9, &v24, (__int64 *)1, 0, 1);
  v15 = sub_161FF10(v9, "llvm.loop.distribute.enable", 0x1Bu);
  v25 = v12;
  v24 = v15;
  v16 = sub_1627350(v9, &v24, (__int64 *)2, 0, 1);
  v24 = v21;
  v25 = (_QWORD *)v22;
  v26 = v23;
  v27 = v14;
  v28 = v16;
  v17 = (unsigned __int8 *)sub_1627350(v9, &v24, (__int64 *)5, 0, 1);
  sub_1630830((__int64)v17, 0, v17, a2, a3, a4, a5, v18, v19, a8, a9);
  return sub_13FCC30(a1, (__int64)v17);
}
