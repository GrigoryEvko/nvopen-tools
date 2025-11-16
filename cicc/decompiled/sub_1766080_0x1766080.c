// Function: sub_1766080
// Address: 0x1766080
//
__int64 __fastcall sub_1766080(
        __int64 *a1,
        __int64 a2,
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
  __int64 v14; // r15
  int v15; // eax
  _BYTE *v17; // rdi
  int v18; // r14d
  unsigned __int8 v19; // al
  bool v20; // al
  __int64 v21; // r8
  char v22; // al
  _QWORD *v23; // rcx
  __int64 *v24; // rdx
  int v25; // eax
  __int64 v26; // rax
  unsigned int v27; // ebx
  int v28; // eax
  int v29; // eax
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rdx
  void *v34; // rdx
  __int64 v35; // rdi
  unsigned __int8 *v36; // rax
  __int64 v37; // rsi
  bool v38; // al
  __int64 v39; // rcx
  __int64 v40; // rax
  bool v41; // al
  __int64 v42; // r8
  char v43; // al
  __int64 v44; // rax
  __int64 *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  bool v48; // al
  int v49; // eax
  __int64 *v50; // rdi
  bool v51; // al
  __int64 v52; // rsi
  __int64 v53; // rax
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 v56; // rax
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // rax
  __int64 v60; // [rsp+0h] [rbp-A0h]
  __int64 v61; // [rsp+0h] [rbp-A0h]
  __int64 v62; // [rsp+8h] [rbp-98h]
  __int64 v63; // [rsp+8h] [rbp-98h]
  __int64 v64; // [rsp+8h] [rbp-98h]
  __int64 v65; // [rsp+8h] [rbp-98h]
  __int64 v66; // [rsp+8h] [rbp-98h]
  __int64 v67; // [rsp+8h] [rbp-98h]
  __int64 v68; // [rsp+10h] [rbp-90h]
  __int64 *v69; // [rsp+10h] [rbp-90h]
  __int64 v70; // [rsp+10h] [rbp-90h]
  _QWORD *v71; // [rsp+10h] [rbp-90h]
  __int64 v72; // [rsp+10h] [rbp-90h]
  __int64 *v73; // [rsp+10h] [rbp-90h]
  __int64 v74; // [rsp+10h] [rbp-90h]
  __int64 v75; // [rsp+18h] [rbp-88h]
  _QWORD *v76; // [rsp+18h] [rbp-88h]
  __int64 v77; // [rsp+18h] [rbp-88h]
  __int64 v78; // [rsp+18h] [rbp-88h]
  __int64 v79; // [rsp+18h] [rbp-88h]
  _QWORD *v80; // [rsp+18h] [rbp-88h]
  __int64 v81; // [rsp+18h] [rbp-88h]
  _QWORD *v82; // [rsp+18h] [rbp-88h]
  __int64 v83; // [rsp+18h] [rbp-88h]
  bool v84; // [rsp+18h] [rbp-88h]
  __int64 v85; // [rsp+20h] [rbp-80h]
  __int64 v86; // [rsp+20h] [rbp-80h]
  __int64 v87; // [rsp+20h] [rbp-80h]
  __int64 *v88; // [rsp+20h] [rbp-80h]
  __int64 v89; // [rsp+20h] [rbp-80h]
  __int64 v90; // [rsp+28h] [rbp-78h]
  __int64 v91; // [rsp+28h] [rbp-78h]
  __int64 v92; // [rsp+28h] [rbp-78h]
  __int64 v93; // [rsp+28h] [rbp-78h]
  __int64 v94[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v95; // [rsp+40h] [rbp-60h] BYREF
  int v96; // [rsp+48h] [rbp-58h]
  __int64 v97; // [rsp+50h] [rbp-50h] BYREF
  int v98; // [rsp+58h] [rbp-48h]
  __int16 v99; // [rsp+60h] [rbp-40h]

  v14 = *(_QWORD *)(a3 - 48);
  v15 = *(unsigned __int8 *)(v14 + 16);
  if ( (unsigned __int8)(v15 - 47) > 2u )
    return 0;
  v17 = *(_BYTE **)(v14 - 24);
  v18 = v15 - 24;
  v19 = v17[16];
  if ( v19 == 13 )
  {
    v85 = (__int64)(v17 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) != 16 )
      goto LABEL_12;
    v79 = a5;
    if ( v19 > 0x10u )
      goto LABEL_12;
    v87 = a4;
    v91 = a3;
    v40 = sub_15A1020(v17, *(_QWORD *)v17, a3, a4);
    a3 = v91;
    a4 = v87;
    if ( !v40 || *(_BYTE *)(v40 + 16) != 13 )
      goto LABEL_12;
    a5 = v79;
    v85 = v40 + 24;
  }
  if ( v18 == 23 )
  {
    v72 = a5;
    v81 = a4;
    v92 = a3;
    v41 = sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF);
    a3 = v92;
    a4 = v81;
    v42 = v72;
    if ( v41 )
    {
      v48 = sub_13D0200((__int64 *)v72, *(_DWORD *)(v72 + 8) - 1);
      a3 = v92;
      a4 = v81;
      if ( v48 )
        goto LABEL_12;
      v49 = *(_DWORD *)(v81 + 8);
      v50 = (__int64 *)v81;
      v83 = v92;
      v93 = a4;
      v51 = sub_13D0200(v50, v49 - 1);
      a4 = v93;
      a3 = v83;
      if ( v51 )
        goto LABEL_12;
      v42 = v72;
    }
    v66 = v42;
    v73 = (__int64 *)a3;
    v82 = (_QWORD *)a4;
    sub_13A38D0((__int64)v94, a4);
    sub_16A81B0((__int64)v94, v85);
    sub_13A38D0((__int64)&v95, (__int64)v94);
    sub_16A7E20((__int64)&v95, v85);
    v43 = sub_1455820((__int64)&v95, v82);
    v23 = v82;
    v24 = v73;
    if ( !v43 )
      goto LABEL_8;
    v44 = sub_15A1070(*v73, (__int64)v94);
    sub_1593B40((_QWORD *)(a2 - 24), v44);
    sub_13A38D0((__int64)&v97, v66);
    sub_16A81B0((__int64)&v97, v85);
    v45 = v73;
    goto LABEL_36;
  }
  if ( v18 != 25 )
    goto LABEL_6;
  v60 = a4;
  v65 = a3;
  v80 = (_QWORD *)a5;
  sub_13A38D0((__int64)&v95, a5);
  sub_16A7E20((__int64)&v95, v85);
  sub_13A38D0((__int64)&v97, (__int64)&v95);
  sub_16A81B0((__int64)&v97, v85);
  v71 = v80;
  LOBYTE(v80) = sub_1455820((__int64)&v97, v80);
  sub_135E100(&v97);
  sub_135E100(&v95);
  a5 = (__int64)v71;
  a3 = v65;
  a4 = v60;
  if ( (_BYTE)v80 )
  {
LABEL_6:
    v62 = a5;
    v68 = a4;
    v75 = a3;
    v20 = sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF);
    a3 = v75;
    a4 = v68;
    v21 = v62;
    if ( v20 )
    {
      v37 = v62;
      v64 = v68;
      v70 = v75;
      v78 = v21;
      sub_13A38D0((__int64)&v95, v37);
      sub_16A7E20((__int64)&v95, v85);
      v38 = sub_13D0200(&v95, v96 - 1);
      v39 = v64;
      if ( v38 )
      {
        sub_135E100(&v95);
        a3 = v70;
        a4 = v64;
        goto LABEL_12;
      }
      v52 = v64;
      v61 = v78;
      v67 = v70;
      v74 = v39;
      sub_13A38D0((__int64)&v97, v52);
      sub_16A7E20((__int64)&v97, v85);
      v84 = sub_13D0200(&v97, v98 - 1);
      sub_135E100(&v97);
      sub_135E100(&v95);
      a4 = v74;
      a3 = v67;
      v21 = v61;
      if ( v84 )
        goto LABEL_12;
    }
    v63 = v21;
    v69 = (__int64 *)a3;
    v76 = (_QWORD *)a4;
    sub_13A38D0((__int64)v94, a4);
    sub_16A7E20((__int64)v94, v85);
    sub_13A38D0((__int64)&v95, (__int64)v94);
    sub_16A81B0((__int64)&v95, v85);
    v22 = sub_1455820((__int64)&v95, v76);
    v23 = v76;
    v24 = v69;
    if ( !v22 )
    {
LABEL_8:
      v25 = *(unsigned __int16 *)(a2 + 18);
      BYTE1(v25) &= ~0x80u;
      if ( v25 == 32 )
      {
        v56 = sub_15A0640(*(_QWORD *)a2);
        v47 = sub_170E100(a1, a2, v56, a6, a7, a8, a9, v57, v58, a12, a13);
      }
      else
      {
        if ( v25 != 33 )
        {
          v77 = (__int64)v23;
          v86 = (__int64)v24;
          sub_135E100(&v95);
          sub_135E100(v94);
          a3 = v86;
          a4 = v77;
          goto LABEL_12;
        }
        v53 = sub_15A0600(*(_QWORD *)a2);
        v47 = sub_170E100(a1, a2, v53, a6, a7, a8, a9, v54, v55, a12, a13);
      }
LABEL_37:
      v89 = v47;
      sub_135E100(&v95);
      sub_135E100(v94);
      return v89;
    }
    v59 = sub_15A1070(*v69, (__int64)v94);
    sub_1593B40((_QWORD *)(a2 - 24), v59);
    sub_13A38D0((__int64)&v97, v63);
    sub_16A7E20((__int64)&v97, v85);
    v45 = v69;
LABEL_36:
    v88 = v45;
    v46 = sub_15A1070(*v45, (__int64)&v97);
    sub_1593B40(v88 - 3, v46);
    sub_1593B40(v88 - 6, *(_QWORD *)(v14 - 48));
    sub_170B990(*a1, v14);
    sub_135E100(&v97);
    v47 = a2;
    goto LABEL_37;
  }
LABEL_12:
  v26 = *(_QWORD *)(v14 + 8);
  if ( !v26 || *(_QWORD *)(v26 + 8) )
    return 0;
  v27 = *(_DWORD *)(a4 + 8);
  if ( v27 <= 0x40 )
  {
    if ( !*(_QWORD *)a4 )
      goto LABEL_16;
    return 0;
  }
  v90 = a3;
  v28 = sub_16A57B0(a4);
  a3 = v90;
  if ( v27 != v28 )
    return 0;
LABEL_16:
  v29 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v29) &= ~0x80u;
  if ( (unsigned int)(v29 - 32) > 1 || *(_BYTE *)(v14 + 16) == 49 || *(_BYTE *)(*(_QWORD *)(v14 - 48) + 16LL) <= 0x10u )
    return 0;
  v31 = *(_QWORD *)(a3 - 24);
  v32 = a1[1];
  v99 = 257;
  v33 = *(_QWORD *)(v14 - 24);
  if ( v18 == 23 )
    v34 = sub_172C310(v32, v31, v33, &v97, 0, *(double *)a6.m128_u64, a7, a8);
  else
    v34 = sub_173DC60(v32, v31, v33, &v97, 0, 0, *(double *)a6.m128_u64, a7, a8);
  v35 = a1[1];
  v99 = 257;
  v36 = sub_1729500(v35, *(unsigned __int8 **)(v14 - 48), (__int64)v34, &v97, *(double *)a6.m128_u64, a7, a8);
  sub_1593B40((_QWORD *)(a2 - 48), (__int64)v36);
  return a2;
}
