// Function: sub_1D60900
// Address: 0x1d60900
//
__int64 __fastcall sub_1D60900(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        int a14)
{
  __int64 v14; // r13
  __int64 v16; // r14
  __int64 *v17; // rdx
  bool v18; // al
  __int64 v19; // rcx
  _BYTE *v20; // r8
  __int64 *v21; // rdi
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 (*v27)(); // rax
  char v28; // al
  unsigned __int64 v29; // rdi
  _BYTE *v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rax
  bool v33; // al
  bool v34; // al
  __int64 v35; // r15
  __int64 *v36; // rsi
  _QWORD *v37; // rdi
  _QWORD *v38; // rax
  __int64 v39; // r15
  __int64 v40; // r14
  __int64 *v41; // r12
  __int64 v42; // r13
  __int64 v43; // r15
  _QWORD *v44; // r10
  _QWORD *v45; // rax
  __int64 v46; // r9
  __int64 v47; // rcx
  __int64 v48; // r9
  __int64 *v49; // r14
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 *v52; // r15
  __int64 *v53; // rbx
  __int64 v54; // rsi
  unsigned __int64 v55; // r8
  __int64 *v56; // rax
  __int64 v57; // rbx
  __int64 *v58; // rbx
  __int64 *v59; // rax
  __int64 *v60; // r13
  __int64 v61; // r9
  __int64 v62; // rax
  __int64 v63; // r14
  __int64 v64; // r15
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  double v73; // xmm4_8
  double v74; // xmm5_8
  __int64 *v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r14
  _QWORD *v78; // rax
  __int64 v79; // rcx
  __int64 v80; // r15
  _QWORD *v81; // rax
  char v82; // al
  __int64 v83; // rcx
  _QWORD *v84; // rax
  __int64 v85; // r15
  _QWORD *v86; // rdi
  _QWORD *v87; // rax
  unsigned int v88; // r15d
  unsigned int v89; // eax
  __int64 v90; // [rsp+8h] [rbp-118h]
  __int64 v91; // [rsp+8h] [rbp-118h]
  __int64 v92; // [rsp+18h] [rbp-108h]
  __int64 v93; // [rsp+18h] [rbp-108h]
  unsigned __int8 v94; // [rsp+20h] [rbp-100h]
  __int64 v95; // [rsp+20h] [rbp-100h]
  unsigned __int8 v96; // [rsp+20h] [rbp-100h]
  _QWORD *v97; // [rsp+28h] [rbp-F8h]
  __int64 v98; // [rsp+38h] [rbp-E8h]
  __int64 v99; // [rsp+40h] [rbp-E0h]
  __int64 v100; // [rsp+48h] [rbp-D8h]
  __int64 *v101; // [rsp+48h] [rbp-D8h]
  __int64 v102; // [rsp+48h] [rbp-D8h]
  _QWORD *v103; // [rsp+48h] [rbp-D8h]
  __int64 v104; // [rsp+48h] [rbp-D8h]
  __int64 v105; // [rsp+48h] [rbp-D8h]
  __int64 v106; // [rsp+48h] [rbp-D8h]
  __int64 v107; // [rsp+48h] [rbp-D8h]
  __int64 v108; // [rsp+48h] [rbp-D8h]
  _BYTE *v109; // [rsp+48h] [rbp-D8h]
  _BYTE *v110; // [rsp+50h] [rbp-D0h]
  __int64 v111; // [rsp+50h] [rbp-D0h]
  _QWORD *v112; // [rsp+50h] [rbp-D0h]
  __int64 v113; // [rsp+50h] [rbp-D0h]
  __int64 v114; // [rsp+50h] [rbp-D0h]
  __int64 v115; // [rsp+58h] [rbp-C8h]
  __int64 **v116; // [rsp+58h] [rbp-C8h]
  _QWORD *v117; // [rsp+58h] [rbp-C8h]
  __int64 v118; // [rsp+58h] [rbp-C8h]
  __int64 v119[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v120; // [rsp+70h] [rbp-B0h]
  __int64 *v121; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v122; // [rsp+88h] [rbp-98h]
  _QWORD v123[2]; // [rsp+90h] [rbp-90h] BYREF
  const char *v124; // [rsp+A0h] [rbp-80h] BYREF
  __int64 *v125; // [rsp+A8h] [rbp-78h]
  __int64 *v126; // [rsp+B0h] [rbp-70h]
  __int64 v127; // [rsp+B8h] [rbp-68h]
  int v128; // [rsp+C0h] [rbp-60h]
  _BYTE v129[88]; // [rsp+C8h] [rbp-58h] BYREF

  v121 = v123;
  v123[0] = a2;
  v122 = 0x200000001LL;
  if ( !a2 )
    BUG();
  v14 = *(_QWORD *)(a2 + 32);
  v16 = a2;
  v17 = v123;
  if ( v14 != *(_QWORD *)(a2 + 40) + 40LL )
  {
    v23 = 1;
    do
    {
      if ( !v14 )
        BUG();
      v24 = v14 - 24;
      if ( *(_BYTE *)(v14 - 8) != 79 || *(_QWORD *)(a2 - 72) != *(_QWORD *)(v14 - 96) )
        break;
      if ( HIDWORD(v122) <= (unsigned int)v23 )
      {
        sub_16CD150((__int64)&v121, v123, 0, 8, v24, a14);
        v23 = (unsigned int)v122;
        v24 = v14 - 24;
      }
      v121[v23] = v24;
      v25 = *(_QWORD *)(a2 + 40);
      v23 = (unsigned int)(v122 + 1);
      LODWORD(v122) = v122 + 1;
      v14 = *(_QWORD *)(v14 + 8);
    }
    while ( v14 != v25 + 40 );
    v17 = &v121[v23 - 1];
  }
  v115 = *v17;
  *(_QWORD *)(a1 + 232) = *(_QWORD *)(*v17 + 32);
  v18 = sub_1642F90(**(_QWORD **)(a2 - 72), 1);
  v19 = v115;
  if ( byte_4FC3140 )
    goto LABEL_6;
  if ( *(_BYTE *)(a1 + 897) )
    goto LABEL_6;
  v20 = *(_BYTE **)(a1 + 176);
  LOBYTE(v14) = v18 && v20 != 0;
  if ( !(_BYTE)v14 )
    goto LABEL_6;
  if ( *(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0 )
  {
    v26 = sub_1625790(a2, 15);
    v19 = v115;
    if ( v26 )
      goto LABEL_6;
    v20 = *(_BYTE **)(a1 + 176);
  }
  v27 = *(__int64 (**)())(*(_QWORD *)v20 + 56LL);
  if ( v27 != sub_1D5A360 )
  {
    v118 = v19;
    v82 = ((__int64 (__fastcall *)(_BYTE *, bool))v27)(v20, *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16);
    v19 = v118;
    if ( !v82 )
      goto LABEL_30;
    v20 = *(_BYTE **)(a1 + 176);
  }
  v100 = v19;
  v110 = v20;
  if ( !v20[81536] )
    goto LABEL_6;
  v116 = *(__int64 ***)(a1 + 192);
  v28 = sub_1625AE0(a2, v119, &v124);
  v19 = v100;
  if ( !v28 )
    goto LABEL_105;
  v29 = (unsigned __int64)v124;
  v30 = v110;
  if ( !&v124[v119[0]] )
    goto LABEL_105;
  v114 = v100;
  if ( v119[0] >= (unsigned __int64)v124 )
    v29 = v119[0];
  v109 = v30;
  v88 = sub_16AF730(v29, (unsigned __int64)&v124[v119[0]]);
  v89 = (*(__int64 (__fastcall **)(_BYTE *))(*(_QWORD *)v109 + 104LL))(v109);
  v19 = v114;
  if ( v89 >= v88 )
  {
LABEL_105:
    v31 = *(_QWORD *)(a2 - 72);
    v111 = v19;
    if ( (unsigned __int8)(*(_BYTE *)(v31 + 16) - 75) > 1u
      || (v32 = *(_QWORD *)(v31 + 8)) == 0
      || *(_QWORD *)(v32 + 8)
      || (v33 = sub_1D5B6E0(v116, *(_QWORD *)(a2 - 48)), v19 = v111, !v33)
      && (v34 = sub_1D5B6E0(v116, *(_QWORD *)(a2 - 24)), v19 = v111, !v34) )
    {
LABEL_6:
      v21 = v121;
      LODWORD(v14) = 0;
      goto LABEL_7;
    }
  }
LABEL_30:
  *(_BYTE *)(a1 + 896) = 1;
  v35 = *(_QWORD *)(a2 + 40);
  v36 = *(__int64 **)(v19 + 32);
  v37 = *(_QWORD **)(v16 + 40);
  v97 = v37;
  v124 = "select.end";
  LOWORD(v126) = 259;
  v98 = sub_157FBF0(v37, v36, (__int64)&v124);
  v38 = (_QWORD *)sub_157EBA0(v35);
  sub_15F20C0(v38);
  v101 = &v121[(unsigned int)v122];
  if ( v121 == v101 )
  {
    v117 = 0;
LABEL_89:
    v83 = *(_QWORD *)(v98 + 56);
    LOWORD(v126) = 259;
    v108 = v83;
    v124 = "select.false";
    v113 = sub_16498A0(v16);
    v84 = (_QWORD *)sub_22077B0(64);
    v85 = (__int64)v84;
    if ( v84 )
      sub_157FB60(v84, v113, (__int64)&v124, v108, v98);
    v86 = sub_1648A60(56, 1u);
    if ( v86 )
      sub_15F8590((__int64)v86, v98, v85);
    v87 = v117;
    v117 = (_QWORD *)v85;
    v112 = v87;
    goto LABEL_40;
  }
  v117 = 0;
  v39 = 0;
  v112 = 0;
  v94 = v14;
  v92 = v16;
  v40 = 0;
  v41 = v121;
  do
  {
    v42 = *v41;
    if ( sub_1D5B6E0(*(__int64 ***)(a1 + 192), *(_QWORD *)(*v41 - 48)) )
    {
      if ( !v112 )
      {
        v76 = *(_QWORD *)(v98 + 56);
        LOWORD(v126) = 259;
        v90 = v76;
        v124 = "select.true.sink";
        v77 = sub_16498A0(v42);
        v112 = (_QWORD *)sub_22077B0(64);
        if ( v112 )
          sub_157FB60(v112, v77, (__int64)&v124, v90, v98);
        v78 = sub_1648A60(56, 1u);
        v40 = (__int64)v78;
        if ( v78 )
          sub_15F8590((__int64)v78, v98, (__int64)v112);
      }
      sub_15F22F0(*(_QWORD **)(v42 - 48), v40);
    }
    if ( sub_1D5B6E0(*(__int64 ***)(a1 + 192), *(_QWORD *)(v42 - 24)) )
    {
      if ( !v117 )
      {
        v79 = *(_QWORD *)(v98 + 56);
        LOWORD(v126) = 259;
        v91 = v79;
        v124 = "select.false.sink";
        v80 = sub_16498A0(v42);
        v117 = (_QWORD *)sub_22077B0(64);
        if ( v117 )
          sub_157FB60(v117, v80, (__int64)&v124, v91, v98);
        v81 = sub_1648A60(56, 1u);
        v39 = (__int64)v81;
        if ( v81 )
          sub_15F8590((__int64)v81, v98, (__int64)v117);
      }
      sub_15F22F0(*(_QWORD **)(v42 - 24), v39);
    }
    ++v41;
  }
  while ( v101 != v41 );
  LODWORD(v14) = v94;
  v16 = v92;
  if ( v112 == v117 )
    goto LABEL_89;
LABEL_40:
  v43 = (__int64)v112;
  if ( v112 )
  {
    v44 = v117;
    if ( !v117 )
    {
      v44 = (_QWORD *)v98;
      v43 = (__int64)v112;
      v117 = v97;
    }
  }
  else
  {
    v44 = v117;
    v43 = v98;
    v112 = v97;
  }
  v95 = (__int64)v44;
  sub_17CE510((__int64)&v124, v16, 0, 0, 0);
  v102 = *(_QWORD *)(v16 - 72);
  v45 = sub_1648A60(56, 3u);
  v46 = (__int64)v45;
  if ( v45 )
  {
    v47 = v102;
    v103 = v45;
    sub_15F83E0((__int64)v45, v43, v95, v47, 0);
    v46 = (__int64)v103;
  }
  v104 = v46;
  v119[0] = 0xF00000002LL;
  v119[1] = 14;
  sub_15F4370(v46, v16, (int *)v119, 4);
  v48 = v104;
  v120 = 257;
  if ( v125 )
  {
    v49 = v126;
    sub_157E9D0((__int64)(v125 + 5), v104);
    v48 = v104;
    v50 = *v49;
    v51 = *(_QWORD *)(v104 + 24);
    *(_QWORD *)(v104 + 32) = v49;
    v50 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v104 + 24) = v50 | v51 & 7;
    *(_QWORD *)(v50 + 8) = v104 + 24;
    *v49 = *v49 & 7 | (v104 + 24);
  }
  v105 = v48;
  sub_164B780(v48, v119);
  sub_12A86E0((__int64 *)&v124, v105);
  sub_17CD270((__int64 *)&v124);
  v21 = v121;
  v124 = 0;
  v125 = (__int64 *)v129;
  v126 = (__int64 *)v129;
  v127 = 2;
  v52 = &v121[(unsigned int)v122];
  v128 = 0;
  if ( v52 == v121 )
  {
    *(_QWORD *)(a1 + 232) = v97 + 5;
  }
  else
  {
    v106 = a1;
    v53 = v121;
    do
    {
      v54 = *v53++;
      sub_1412190((__int64)&v124, v54);
      v55 = (unsigned __int64)v126;
      v56 = v125;
    }
    while ( v52 != v53 );
    v57 = v106;
    if ( v121 != &v121[(unsigned int)v122] )
    {
      v93 = v106;
      v58 = &v121[(unsigned int)v122];
      v96 = v14;
      while ( 1 )
      {
        v60 = (__int64 *)*(v58 - 1);
        v61 = *(_QWORD *)(v98 + 48);
        v120 = 257;
        if ( v61 )
          v61 -= 24;
        v107 = *v60;
        v99 = v61;
        v62 = sub_1648B60(64);
        v63 = v62;
        if ( v62 )
        {
          v64 = v62;
          sub_15F1EA0(v62, v107, 53, 0, 0, v99);
          *(_DWORD *)(v63 + 56) = 2;
          sub_164B780(v63, v119);
          sub_1648880(v63, *(_DWORD *)(v63 + 56), 1);
        }
        else
        {
          v64 = 0;
        }
        sub_164B7C0(v64, (__int64)v60);
        v65 = sub_1D5B040((__int64)v60, 1, (__int64)&v124);
        sub_1704F80(v63, v65, (__int64)v112, v66, v67, v68);
        v69 = sub_1D5B040((__int64)v60, 0, (__int64)&v124);
        sub_1704F80(v63, v69, (__int64)v117, v70, v71, v72);
        sub_164D160((__int64)v60, v63, a3, a4, a5, a6, v73, v74, a9, a10);
        sub_15F20C0(v60);
        v59 = v125;
        if ( v126 == v125 )
        {
          v75 = &v125[HIDWORD(v127)];
          if ( v125 == v75 )
          {
LABEL_69:
            v59 = &v125[HIDWORD(v127)];
          }
          else
          {
            while ( v60 != (__int64 *)*v59 )
            {
              if ( v75 == ++v59 )
                goto LABEL_69;
            }
          }
          goto LABEL_64;
        }
        v59 = sub_16CC9F0((__int64)&v124, (__int64)v60);
        if ( v60 == (__int64 *)*v59 )
          break;
        if ( v126 == v125 )
        {
          v59 = &v126[HIDWORD(v127)];
          v75 = v59;
          goto LABEL_64;
        }
LABEL_54:
        if ( v121 == --v58 )
        {
          LODWORD(v14) = v96;
          v57 = v93;
          v55 = (unsigned __int64)v126;
          v56 = v125;
          goto LABEL_72;
        }
      }
      if ( v126 == v125 )
        v75 = &v126[HIDWORD(v127)];
      else
        v75 = &v126[(unsigned int)v127];
LABEL_64:
      if ( v75 != v59 )
      {
        *v59 = -2;
        ++v128;
      }
      goto LABEL_54;
    }
LABEL_72:
    *(_QWORD *)(v57 + 232) = v97 + 5;
    if ( (__int64 *)v55 != v56 )
      _libc_free(v55);
    v21 = v121;
  }
LABEL_7:
  if ( v21 != v123 )
    _libc_free((unsigned __int64)v21);
  return (unsigned int)v14;
}
