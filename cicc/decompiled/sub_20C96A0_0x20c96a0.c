// Function: sub_20C96A0
// Address: 0x20c96a0
//
unsigned __int64 __fastcall sub_20C96A0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        unsigned int a4,
        __int64 (__fastcall *a5)(__int64, unsigned __int8 **, __int64),
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // r15
  _QWORD *v20; // rax
  _QWORD *v21; // rdi
  _QWORD *v22; // rax
  _QWORD *v23; // r15
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdx
  unsigned __int8 *v28; // rsi
  __int64 v29; // r15
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // r15
  _QWORD *v37; // rax
  _QWORD *v38; // rbx
  __int64 v39; // rsi
  __int64 v40; // rax
  double v41; // xmm4_8
  double v42; // xmm5_8
  __int64 v43; // rsi
  __int64 v44; // rdx
  unsigned __int8 *v45; // rsi
  __int64 *v46; // rdx
  unsigned __int8 *v47; // rsi
  unsigned __int8 *v48; // rsi
  unsigned __int64 result; // rax
  _QWORD *v50; // rax
  __int64 v51; // r9
  _QWORD **v52; // rax
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // r9
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rdx
  unsigned __int8 *v60; // rsi
  __int64 v61; // [rsp+10h] [rbp-110h]
  __int64 *v62; // [rsp+10h] [rbp-110h]
  __int64 v65; // [rsp+20h] [rbp-100h]
  __int64 v67; // [rsp+28h] [rbp-F8h]
  _QWORD *v69; // [rsp+30h] [rbp-F0h]
  __int64 *v70; // [rsp+30h] [rbp-F0h]
  __int64 v71; // [rsp+30h] [rbp-F0h]
  _QWORD *v72; // [rsp+38h] [rbp-E8h]
  __int64 v73; // [rsp+38h] [rbp-E8h]
  __int64 v74; // [rsp+38h] [rbp-E8h]
  _QWORD *v75; // [rsp+40h] [rbp-E0h]
  _QWORD *v76; // [rsp+48h] [rbp-D8h]
  __int64 *v77; // [rsp+48h] [rbp-D8h]
  unsigned __int8 *v78; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v79[2]; // [rsp+60h] [rbp-C0h] BYREF
  char v80; // [rsp+70h] [rbp-B0h]
  char v81; // [rsp+71h] [rbp-AFh]
  unsigned __int8 *v82[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v83; // [rsp+90h] [rbp-90h]
  unsigned __int8 *v84; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v85; // [rsp+A8h] [rbp-78h]
  __int64 *v86; // [rsp+B0h] [rbp-70h]
  _QWORD *v87; // [rsp+B8h] [rbp-68h]
  __int64 v88; // [rsp+C0h] [rbp-60h]
  int v89; // [rsp+C8h] [rbp-58h]
  __int64 v90; // [rsp+D0h] [rbp-50h]
  __int64 v91; // [rsp+D8h] [rbp-48h]

  v16 = sub_16498A0((__int64)a2);
  v17 = (unsigned __int8 *)a2[6];
  v84 = 0;
  v87 = (_QWORD *)v16;
  v18 = (_QWORD *)a2[5];
  v88 = 0;
  v85 = v18;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v86 = a2 + 3;
  v82[0] = v17;
  if ( v17 )
  {
    sub_1623A60((__int64)v82, (__int64)v17, 2);
    if ( v84 )
      sub_161E7C0((__int64)&v84, (__int64)v84);
    v84 = v82[0];
    if ( v82[0] )
      sub_1623210((__int64)v82, v82[0], (__int64)&v84);
  }
  v19 = v85;
  v72 = v87;
  v61 = v85[7];
  v82[0] = "atomicrmw.end";
  v83 = 259;
  v75 = (_QWORD *)sub_157FBF0(v85, v86, (__int64)v82);
  v82[0] = "atomicrmw.start";
  v83 = 259;
  v20 = (_QWORD *)sub_22077B0(64);
  v76 = v20;
  if ( v20 )
    sub_157FB60(v20, (__int64)v72, (__int64)v82, v61, (__int64)v75);
  v21 = (_QWORD *)((v19[5] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v19[5] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v21 = 0;
  sub_15F20C0(v21);
  v83 = 257;
  v85 = v19;
  v86 = v19 + 5;
  v22 = sub_1648A60(56, 1u);
  v23 = v22;
  if ( v22 )
    sub_15F8320((__int64)v22, (__int64)v76, 0);
  if ( v85 )
  {
    v62 = v86;
    sub_157E9D0((__int64)(v85 + 5), (__int64)v23);
    v24 = *v62;
    v25 = v23[3] & 7LL;
    v23[4] = v62;
    v24 &= 0xFFFFFFFFFFFFFFF8LL;
    v23[3] = v24 | v25;
    *(_QWORD *)(v24 + 8) = v23 + 3;
    *v62 = *v62 & 7 | (unsigned __int64)(v23 + 3);
  }
  sub_164B780((__int64)v23, (__int64 *)v82);
  if ( v84 )
  {
    v79[0] = (__int64)v84;
    sub_1623A60((__int64)v79, (__int64)v84, 2);
    v26 = v23[6];
    v27 = (__int64)(v23 + 6);
    if ( v26 )
    {
      sub_161E7C0((__int64)(v23 + 6), v26);
      v27 = (__int64)(v23 + 6);
    }
    v28 = (unsigned __int8 *)v79[0];
    v23[6] = v79[0];
    if ( v28 )
      sub_1623210((__int64)v79, v28, v27);
  }
  v29 = a1;
  v30 = *(_QWORD *)(a1 + 160);
  v85 = v76;
  v86 = v76 + 5;
  v67 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 **, __int64, _QWORD))(*(_QWORD *)v30 + 608LL))(
          v30,
          &v84,
          a3,
          a4);
  v31 = a5(a6, &v84, v67);
  v32 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int8 **, __int64, __int64, _QWORD))(**(_QWORD **)(v29 + 160)
                                                                                        + 616LL))(
          *(_QWORD *)(v29 + 160),
          &v84,
          v31,
          a3,
          a4);
  v81 = 1;
  v33 = v32;
  v80 = 3;
  v79[0] = (__int64)"tryagain";
  v34 = sub_1644900(v72, 0x20u);
  v35 = sub_159C470(v34, 0, 0);
  if ( *(_BYTE *)(v33 + 16) > 0x10u || *(_BYTE *)(v35 + 16) > 0x10u )
  {
    v73 = v35;
    v83 = 257;
    v50 = sub_1648A60(56, 2u);
    v51 = v73;
    v36 = v50;
    if ( v50 )
    {
      v74 = (__int64)v50;
      v52 = *(_QWORD ***)v33;
      if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) == 16 )
      {
        v65 = v51;
        v69 = v52[4];
        v53 = (__int64 *)sub_1643320(*v52);
        v54 = (__int64)sub_16463B0(v53, (unsigned int)v69);
        v55 = v65;
      }
      else
      {
        v71 = v51;
        v54 = sub_1643320(*v52);
        v55 = v71;
      }
      sub_15FEC10((__int64)v36, v54, 51, 33, v33, v55, (__int64)v82, 0);
    }
    else
    {
      v74 = 0;
    }
    if ( v85 )
    {
      v70 = v86;
      sub_157E9D0((__int64)(v85 + 5), (__int64)v36);
      v56 = *v70;
      v57 = v36[3] & 7LL;
      v36[4] = v70;
      v56 &= 0xFFFFFFFFFFFFFFF8LL;
      v36[3] = v56 | v57;
      *(_QWORD *)(v56 + 8) = v36 + 3;
      *v70 = *v70 & 7 | (unsigned __int64)(v36 + 3);
    }
    sub_164B780(v74, v79);
    if ( v84 )
    {
      v78 = v84;
      sub_1623A60((__int64)&v78, (__int64)v84, 2);
      v58 = v36[6];
      v59 = (__int64)(v36 + 6);
      if ( v58 )
      {
        sub_161E7C0((__int64)(v36 + 6), v58);
        v59 = (__int64)(v36 + 6);
      }
      v60 = v78;
      v36[6] = v78;
      if ( v60 )
        sub_1623210((__int64)&v78, v60, v59);
    }
  }
  else
  {
    v36 = (_QWORD *)sub_15A37B0(0x21u, (_QWORD *)v33, (_QWORD *)v35, 0);
  }
  v83 = 257;
  v37 = sub_1648A60(56, 3u);
  v38 = v37;
  if ( v37 )
    sub_15F83E0((__int64)v37, (__int64)v76, (__int64)v75, (__int64)v36, 0);
  if ( v85 )
  {
    v77 = v86;
    sub_157E9D0((__int64)(v85 + 5), (__int64)v38);
    v39 = *v77;
    v40 = v38[3] & 7LL;
    v38[4] = v77;
    v39 &= 0xFFFFFFFFFFFFFFF8LL;
    v38[3] = v39 | v40;
    *(_QWORD *)(v39 + 8) = v38 + 3;
    *v77 = *v77 & 7 | (unsigned __int64)(v38 + 3);
  }
  sub_164B780((__int64)v38, (__int64 *)v82);
  if ( v84 )
  {
    v79[0] = (__int64)v84;
    sub_1623A60((__int64)v79, (__int64)v84, 2);
    v43 = v38[6];
    v44 = (__int64)(v38 + 6);
    if ( v43 )
    {
      sub_161E7C0((__int64)(v38 + 6), v43);
      v44 = (__int64)(v38 + 6);
    }
    v45 = (unsigned __int8 *)v79[0];
    v38[6] = v79[0];
    if ( v45 )
      sub_1623210((__int64)v79, v45, v44);
  }
  v46 = (__int64 *)v75[6];
  v85 = v75;
  v86 = v46;
  if ( v46 != v75 + 5 )
  {
    if ( !v46 )
      BUG();
    v47 = (unsigned __int8 *)v46[3];
    v82[0] = v47;
    if ( v47 )
    {
      sub_1623A60((__int64)v82, (__int64)v47, 2);
      v48 = v84;
      if ( !v84 )
        goto LABEL_36;
    }
    else
    {
      v48 = v84;
      if ( !v84 )
        goto LABEL_38;
    }
    sub_161E7C0((__int64)&v84, (__int64)v48);
LABEL_36:
    v84 = v82[0];
    if ( v82[0] )
      sub_1623210((__int64)v82, v82[0], (__int64)&v84);
  }
LABEL_38:
  sub_164D160((__int64)a2, v67, a7, a8, a9, a10, v41, v42, a13, a14);
  result = sub_15F20C0(a2);
  if ( v84 )
    return sub_161E7C0((__int64)&v84, (__int64)v84);
  return result;
}
