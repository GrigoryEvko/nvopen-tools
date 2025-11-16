// Function: sub_1AA6B00
// Address: 0x1aa6b00
//
__int64 __fastcall sub_1AA6B00(
        __int64 a1,
        __int64 *a2,
        _QWORD *a3,
        _QWORD *a4,
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
  _QWORD *v13; // r13
  __int64 v14; // r15
  __int64 v15; // rbx
  _QWORD *v16; // rax
  __int64 v17; // r14
  __int64 v18; // rcx
  _QWORD *v19; // rax
  __int64 v20; // r13
  _QWORD *v21; // rax
  _QWORD *v22; // rbx
  __int64 *v23; // r8
  __int64 v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rbx
  __int64 *v27; // r15
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // r12
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v34; // rsi
  unsigned __int8 *v35; // rsi
  __int64 v36; // rsi
  unsigned __int8 *v37; // rsi
  __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+8h] [rbp-88h]
  __int64 *v43; // [rsp+28h] [rbp-68h]
  unsigned __int64 v44; // [rsp+30h] [rbp-60h]
  __int64 v46[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v47; // [rsp+50h] [rbp-40h]

  v13 = (_QWORD *)a2[5];
  v47 = 257;
  v14 = sub_157FBF0(v13, a2 + 3, (__int64)v46);
  v44 = sub_157EBA0((__int64)v13);
  v15 = sub_157E9C0((__int64)v13);
  v38 = v13[7];
  v47 = 257;
  v16 = (_QWORD *)sub_22077B0(64);
  v17 = (__int64)v16;
  if ( v16 )
    sub_157FB60(v16, v15, (__int64)v46, v38, v14);
  v18 = v13[7];
  v47 = 257;
  v39 = v18;
  v19 = (_QWORD *)sub_22077B0(64);
  v20 = (__int64)v19;
  if ( v19 )
    sub_157FB60(v19, v15, (__int64)v46, v39, v14);
  v21 = sub_1648A60(56, 1u);
  v22 = v21;
  if ( v21 )
    sub_15F8590((__int64)v21, v14, v17);
  v23 = v22 + 6;
  *a3 = v22;
  v24 = a2[6];
  v46[0] = v24;
  if ( !v24 )
  {
    if ( v23 == v46 )
      goto LABEL_11;
    v36 = v22[6];
    if ( !v36 )
      goto LABEL_11;
LABEL_25:
    v43 = v23;
    sub_161E7C0((__int64)v23, v36);
    v23 = v43;
    goto LABEL_26;
  }
  sub_1623A60((__int64)v46, v24, 2);
  v23 = v22 + 6;
  if ( v22 + 6 == v46 )
  {
    if ( v46[0] )
      sub_161E7C0((__int64)v46, v46[0]);
    goto LABEL_11;
  }
  v36 = v22[6];
  if ( v36 )
    goto LABEL_25;
LABEL_26:
  v37 = (unsigned __int8 *)v46[0];
  v22[6] = v46[0];
  if ( v37 )
    sub_1623210((__int64)v46, v37, (__int64)v23);
LABEL_11:
  v25 = sub_1648A60(56, 1u);
  v26 = v25;
  if ( v25 )
    sub_15F8590((__int64)v25, v14, v20);
  v27 = v26 + 6;
  *a4 = v26;
  v28 = a2[6];
  v46[0] = v28;
  if ( !v28 )
  {
    if ( v27 == v46 )
      goto LABEL_17;
    v34 = v26[6];
    if ( !v34 )
      goto LABEL_17;
LABEL_21:
    sub_161E7C0((__int64)(v26 + 6), v34);
    goto LABEL_22;
  }
  sub_1623A60((__int64)v46, v28, 2);
  if ( v27 == v46 )
  {
    if ( v46[0] )
      sub_161E7C0((__int64)v46, v46[0]);
    goto LABEL_17;
  }
  v34 = v26[6];
  if ( v34 )
    goto LABEL_21;
LABEL_22:
  v35 = (unsigned __int8 *)v46[0];
  v26[6] = v46[0];
  if ( v35 )
    sub_1623210((__int64)v46, v35, (__int64)(v26 + 6));
LABEL_17:
  v29 = sub_1648A60(56, 3u);
  v30 = v29;
  if ( v29 )
    sub_15F83E0((__int64)v29, v17, v20, a1, 0);
  sub_1625C10((__int64)v30, 2, a5);
  return sub_1AA6530(v44, v30, a6, a7, a8, a9, v31, v32, a12, a13);
}
