// Function: sub_17960B0
// Address: 0x17960b0
//
__int64 __fastcall sub_17960B0(
        __int64 *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        _BYTE *a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        int a15,
        _BYTE *a16)
{
  __int64 ***v19; // rbx
  __int64 ***v20; // r13
  int v21; // r9d
  __int64 v22; // r13
  const char *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 ***v28; // rax
  __int64 v29; // rbx
  __int64 ***v30; // r14
  __int64 v31; // r13
  _QWORD *v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // r13
  __int64 v37; // rbx
  __int64 v38; // r14
  _QWORD *v39; // rax
  double v40; // xmm4_8
  double v41; // xmm5_8
  __int64 v42; // rsi
  __int64 v43; // r13
  __int64 v44; // r14
  _QWORD *v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  unsigned __int8 v48; // al
  __int64 v49; // r10
  unsigned __int8 v50; // al
  unsigned __int64 *v51; // rsi
  int v52; // eax
  __int64 v53; // rcx
  __int64 v54; // r14
  _QWORD *v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rcx
  __int64 v58; // r8
  int v59; // r9d
  __int64 v60; // r13
  int v61; // eax
  _QWORD *v62; // r14
  __int64 v63; // r13
  int v64; // eax
  _QWORD *v65; // rax
  unsigned __int8 *v66; // rax
  __int64 v67; // rbx
  __int64 v68; // r14
  _QWORD *v69; // rax
  __int64 v70; // r13
  __int64 v71; // r14
  _QWORD *v72; // rax
  __int64 v73; // rax
  int v74; // eax
  int v75; // eax
  int v76; // eax
  __int64 v77; // rax
  __int64 v78; // rdi
  unsigned __int8 *v79; // rax
  __int64 v80; // rdi
  unsigned __int8 *v81; // rax
  __int64 v82; // rdi
  unsigned __int8 *v83; // rax
  _BYTE *v84; // [rsp+8h] [rbp-98h]
  __int64 v85; // [rsp+10h] [rbp-90h]
  __int64 v86; // [rsp+10h] [rbp-90h]
  __int64 v87; // [rsp+10h] [rbp-90h]
  __int64 v88; // [rsp+18h] [rbp-88h]
  __int64 v89; // [rsp+18h] [rbp-88h]
  __int64 v90; // [rsp+18h] [rbp-88h]
  __int64 v91; // [rsp+18h] [rbp-88h]
  _BYTE *v92; // [rsp+18h] [rbp-88h]
  __int64 v93; // [rsp+18h] [rbp-88h]
  __int64 v94; // [rsp+18h] [rbp-88h]
  __int64 v95; // [rsp+18h] [rbp-88h]
  __int64 v96; // [rsp+18h] [rbp-88h]
  char v97; // [rsp+2Fh] [rbp-71h] BYREF
  unsigned __int8 *v98; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int8 *v99; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v100[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v101[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v102; // [rsp+60h] [rbp-40h]
  int v103; // [rsp+B0h] [rbp+10h]
  int v104; // [rsp+B0h] [rbp+10h]
  int v105; // [rsp+B0h] [rbp+10h]
  int v106; // [rsp+B0h] [rbp+10h]
  int v107; // [rsp+B0h] [rbp+10h]

  v19 = (__int64 ***)a2;
  v20 = (__int64 ***)a16;
  v21 = a15;
  if ( a16 == (_BYTE *)a4 || a16 == a5 )
  {
    if ( a3 == a15 && (unsigned int)(a3 - 7) > 1 && a3 )
    {
      v43 = *(_QWORD *)(a6 + 8);
      if ( !v43 )
        return 0;
      v44 = *a1;
      do
      {
        v45 = sub_1648700(v43);
        sub_170B990(v44, (__int64)v45);
        v43 = *(_QWORD *)(v43 + 8);
      }
      while ( v43 );
      goto LABEL_35;
    }
    if ( a3 == 1 && a15 == 3 || a3 == 3 && a15 == 1 || a3 == 2 && a15 == 4 || a3 == 4 && a15 == 2 )
    {
      v37 = *(_QWORD *)(a6 + 8);
      if ( !v37 )
        return 0;
      v38 = *a1;
      do
      {
        v39 = sub_1648700(v37);
        sub_170B990(v38, (__int64)v39);
        v37 = *(_QWORD *)(v37 + 8);
      }
      while ( v37 );
LABEL_23:
      if ( v20 == (__int64 ***)a6 )
        v20 = (__int64 ***)sub_1599EF0(*v20);
      v42 = (__int64)v20;
      v35 = a6;
      sub_164D160(a6, v42, a7, a8, a9, a10, v40, v41, a13, a14);
      return v35;
    }
  }
  if ( a3 != a15 )
  {
LABEL_4:
    if ( a3 == 7 && v21 == 8 || a3 == 8 && v21 == 7 )
    {
      v22 = a1[1];
      v23 = sub_1649960((__int64)v19);
      v24 = (__int64)*(v19 - 6);
      v25 = (__int64)*(v19 - 9);
      v100[1] = v26;
      v102 = 261;
      v27 = (__int64)*(v19 - 3);
      v100[0] = v23;
      v101[0] = (__int64)v100;
      v28 = (__int64 ***)sub_1707C10(v22, v25, v27, v24, v101, (__int64)v19);
      v29 = *(_QWORD *)(a6 + 8);
      v30 = v28;
      if ( v29 )
      {
        v31 = *a1;
        do
        {
          v32 = sub_1648700(v29);
          sub_170B990(v31, (__int64)v32);
          v29 = *(_QWORD *)(v29 + 8);
        }
        while ( v29 );
        if ( v30 == (__int64 ***)a6 )
          v30 = (__int64 ***)sub_1599EF0(*v30);
        v35 = a6;
        sub_164D160(a6, (__int64)v30, a7, a8, a9, a10, v33, v34, a13, a14);
        return v35;
      }
      return 0;
    }
    v88 = (__int64)a5;
    v97 = 0;
    if ( (unsigned int)(a3 - 7) <= 1 )
      return 0;
    if ( !a3 )
      return 0;
    if ( (unsigned int)(v21 - 7) <= 1 )
      return 0;
    v103 = v21;
    if ( !v21 )
      return 0;
    v85 = a4;
    if ( !(unsigned __int8)sub_1791020(a4, (__int64)&v98, (unsigned __int64)&v97, a4)
      || !(unsigned __int8)sub_1791020(v88, (__int64)&v99, (unsigned __int64)&v97, v56)
      || !(unsigned __int8)sub_1791020((__int64)a16, (__int64)v100, (unsigned __int64)&v97, v57)
      || !v97 )
    {
      return 0;
    }
    v58 = v88;
    v59 = v103;
    if ( !v98 )
    {
      v82 = a1[1];
      v102 = 257;
      v83 = sub_171CA90(v82, v85, v101, *(double *)a7.m128_u64, a8, a9);
      v59 = v103;
      v58 = v88;
      v98 = v83;
    }
    if ( !v99 )
    {
      v80 = a1[1];
      v107 = v59;
      v102 = 257;
      v81 = sub_171CA90(v80, v58, v101, *(double *)a7.m128_u64, a8, a9);
      v59 = v107;
      v99 = v81;
    }
    if ( !v100[0] )
    {
      v78 = a1[1];
      v106 = v59;
      v102 = 257;
      v79 = sub_171CA90(v78, (__int64)a16, v101, *(double *)a7.m128_u64, a8, a9);
      v59 = v106;
      v100[0] = v79;
    }
    v60 = (__int64)v98;
    v104 = v59;
    v90 = (__int64)v99;
    v61 = sub_14AEB50(a3);
    v62 = sub_1791FA0(a1[1], v61, v60, v90);
    v63 = a1[1];
    v91 = v100[0];
    v102 = 257;
    v64 = sub_14AEB50(v104);
    v65 = sub_1791FA0(a1[1], v64, (__int64)v62, v91);
    v66 = sub_171CA90(v63, (__int64)v65, v101, *(double *)a7.m128_u64, a8, a9);
    v67 = *(_QWORD *)(a6 + 8);
    v20 = (__int64 ***)v66;
    if ( !v67 )
      return 0;
    v68 = *a1;
    do
    {
      v69 = sub_1648700(v67);
      sub_170B990(v68, (__int64)v69);
      v67 = *(_QWORD *)(v67 + 8);
    }
    while ( v67 );
    goto LABEL_23;
  }
  v48 = a5[16];
  v49 = (__int64)(a5 + 24);
  if ( v48 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a5 + 8LL) != 16 )
      goto LABEL_70;
    if ( v48 > 0x10u )
      goto LABEL_70;
    v86 = a4;
    v92 = a5;
    v73 = sub_15A1020(a5, a2, *(_QWORD *)a5, a4);
    a5 = v92;
    a4 = v86;
    v21 = a15;
    if ( !v73 || *(_BYTE *)(v73 + 16) != 13 )
      goto LABEL_70;
    v49 = v73 + 24;
  }
  v50 = a16[16];
  v51 = (unsigned __int64 *)(a16 + 24);
  if ( v50 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a16 + 8LL) != 16 )
      goto LABEL_70;
    if ( v50 > 0x10u )
      goto LABEL_70;
    v105 = v21;
    v84 = a5;
    v87 = a4;
    v96 = v49;
    v77 = sub_15A1020(a16, (__int64)v51, *(_QWORD *)a16, a4);
    a4 = v87;
    a5 = v84;
    v21 = v105;
    if ( !v77 || *(_BYTE *)(v77 + 16) != 13 )
      goto LABEL_70;
    v49 = v96;
    v51 = (unsigned __int64 *)(v77 + 24);
  }
  switch ( a3 )
  {
    case 2:
      v93 = a4;
      v74 = sub_16A9900(v49, v51);
      v53 = v93;
      if ( v74 <= 0 )
        goto LABEL_44;
      goto LABEL_81;
    case 1:
      v94 = a4;
      v75 = sub_16AEA10(v49, (__int64)v51);
      v53 = v94;
      if ( v75 <= 0 )
        goto LABEL_44;
      goto LABEL_81;
    case 4:
      v89 = a4;
      v52 = sub_16A9900(v49, v51);
      v53 = v89;
      if ( v52 >= 0 )
      {
LABEL_44:
        v35 = *(_QWORD *)(a6 + 8);
        if ( !v35 )
          return v35;
        v54 = *a1;
        do
        {
          v55 = sub_1648700(v35);
          sub_170B990(v54, (__int64)v55);
          v35 = *(_QWORD *)(v35 + 8);
        }
        while ( v35 );
        goto LABEL_35;
      }
LABEL_81:
      v35 = a6;
      sub_1648780(a6, (__int64)v19, v53);
      return v35;
    case 3:
      v95 = a4;
      v76 = sub_16AEA10(v49, (__int64)v51);
      v53 = v95;
      if ( v76 >= 0 )
        goto LABEL_44;
      goto LABEL_81;
  }
LABEL_70:
  if ( (unsigned int)(a3 - 7) > 1 )
    goto LABEL_4;
  v70 = *(_QWORD *)(a6 + 8);
  if ( !v70 )
    return 0;
  v71 = *a1;
  do
  {
    v72 = sub_1648700(v70);
    sub_170B990(v71, (__int64)v72);
    v70 = *(_QWORD *)(v70 + 8);
  }
  while ( v70 );
LABEL_35:
  if ( v19 == (__int64 ***)a6 )
    v19 = (__int64 ***)sub_1599EF0(*v19);
  v35 = a6;
  sub_164D160(a6, (__int64)v19, a7, a8, a9, a10, v46, v47, a13, a14);
  return v35;
}
