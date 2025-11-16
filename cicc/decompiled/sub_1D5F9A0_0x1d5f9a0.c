// Function: sub_1D5F9A0
// Address: 0x1d5f9a0
//
__int64 __fastcall sub_1D5F9A0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        _DWORD *a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        _QWORD *a15,
        unsigned __int8 a16)
{
  __int64 v16; // r12
  __int64 *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rcx
  int v23; // edi
  unsigned int v24; // r8d
  __int64 *v25; // rax
  __int64 v26; // r10
  __int64 v27; // r8
  unsigned __int64 v28; // r15
  unsigned int v29; // edx
  unsigned __int8 *v30; // rax
  __int64 v31; // r8
  __int64 **v32; // rbx
  _QWORD *v33; // rax
  double v34; // xmm4_8
  double v35; // xmm5_8
  _QWORD *v36; // r13
  __int64 **v37; // rax
  unsigned int v38; // eax
  _QWORD *v39; // rdx
  unsigned __int8 *v40; // r13
  __int64 v41; // rbx
  unsigned __int8 *v42; // rsi
  unsigned __int8 **v43; // r8
  int v44; // r9d
  double v45; // xmm4_8
  double v46; // xmm5_8
  __int64 v47; // rax
  __int64 v48; // r10
  __int64 **v49; // r11
  char v50; // al
  const void **v51; // rsi
  unsigned int v52; // edx
  __int64 v53; // rax
  _QWORD *v54; // rax
  double v55; // xmm4_8
  double v56; // xmm5_8
  _QWORD *v57; // r13
  __int64 v58; // rax
  unsigned __int8 *v59; // rsi
  __int64 v60; // rax
  unsigned __int8 *v61; // r9
  unsigned int v62; // eax
  _QWORD *v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rsi
  _QWORD *v66; // rax
  _QWORD *v67; // r8
  __int64 v68; // rax
  bool v69; // zf
  unsigned __int64 v70; // rax
  unsigned int v71; // eax
  _QWORD *v72; // rdx
  __int64 v74; // rax
  __int64 v75; // rcx
  _QWORD *v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  unsigned __int8 *v79; // rsi
  __int64 **v80; // r11
  unsigned __int8 *v81; // rsi
  __int64 v82; // rax
  unsigned int v83; // eax
  _QWORD *v84; // rdx
  __int64 **v85; // rdx
  unsigned __int8 *v86; // rdi
  int v87; // ecx
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // r10
  __int64 v91; // rcx
  __int64 v92; // rdi
  __int64 v93; // rax
  __int64 v94; // rsi
  __int64 v95; // rdx
  unsigned __int8 *v96; // rsi
  int v97; // r13d
  unsigned __int8 *v98; // r11
  int v99; // edi
  __int64 v100; // r9
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rsi
  __int64 v104; // rdx
  unsigned __int8 *v105; // rsi
  int v106; // eax
  int v107; // r9d
  __int64 v108; // [rsp+8h] [rbp-118h]
  __int64 **v109; // [rsp+10h] [rbp-110h]
  __int64 **v110; // [rsp+10h] [rbp-110h]
  __int64 *v111; // [rsp+10h] [rbp-110h]
  __int64 v112; // [rsp+10h] [rbp-110h]
  __int64 v113; // [rsp+10h] [rbp-110h]
  __int64 v114; // [rsp+10h] [rbp-110h]
  __int64 v115; // [rsp+10h] [rbp-110h]
  __int64 v116; // [rsp+18h] [rbp-108h]
  __int64 v117; // [rsp+18h] [rbp-108h]
  __int64 v118; // [rsp+18h] [rbp-108h]
  __int64 v119; // [rsp+18h] [rbp-108h]
  __int64 v120; // [rsp+18h] [rbp-108h]
  __int64 v121; // [rsp+18h] [rbp-108h]
  __int64 v122; // [rsp+18h] [rbp-108h]
  __int64 v123; // [rsp+18h] [rbp-108h]
  __int64 v124; // [rsp+18h] [rbp-108h]
  __int64 *v125; // [rsp+18h] [rbp-108h]
  __int64 v126; // [rsp+18h] [rbp-108h]
  _QWORD *v129; // [rsp+28h] [rbp-F8h]
  __int64 v130; // [rsp+28h] [rbp-F8h]
  _QWORD *v131; // [rsp+28h] [rbp-F8h]
  _QWORD *v132; // [rsp+28h] [rbp-F8h]
  __int64 v133; // [rsp+28h] [rbp-F8h]
  int v135; // [rsp+40h] [rbp-E0h]
  __int64 **v136; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v137; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v138; // [rsp+40h] [rbp-E0h]
  __int64 v139; // [rsp+40h] [rbp-E0h]
  __int64 v140; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v141; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v142; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v144; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v145[2]; // [rsp+60h] [rbp-C0h] BYREF
  char v146; // [rsp+70h] [rbp-B0h]
  char v147; // [rsp+71h] [rbp-AFh]
  unsigned __int8 *v148[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v149; // [rsp+90h] [rbp-90h]
  unsigned __int8 *v150; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v151; // [rsp+A8h] [rbp-78h]
  __int64 *v152; // [rsp+B0h] [rbp-70h]
  __int64 v153; // [rsp+B8h] [rbp-68h]
  __int64 v154; // [rsp+C0h] [rbp-60h]
  int v155; // [rsp+C8h] [rbp-58h]
  __int64 v156; // [rsp+D0h] [rbp-50h]
  __int64 v157; // [rsp+D8h] [rbp-48h]

  v16 = (__int64)a2;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v18 = *(__int64 **)(a1 - 8);
  else
    v18 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v19 = *v18;
  *a4 = 0;
  v20 = *(_QWORD *)(v19 + 8);
  if ( v20 && !*(_QWORD *)(v20 + 8) )
    goto LABEL_5;
  v136 = *(__int64 ***)v19;
  v54 = (_QWORD *)sub_22077B0(24);
  v57 = v54;
  if ( v54 )
  {
    v54[1] = a1;
    *v54 = off_49855B8;
    v150 = 0;
    v153 = sub_16498A0(a1);
    v58 = *(_QWORD *)(a1 + 40);
    v154 = 0;
    v155 = 0;
    v151 = v58;
    v59 = *(unsigned __int8 **)(a1 + 48);
    v156 = 0;
    v157 = 0;
    v152 = (__int64 *)(a1 + 24);
    v148[0] = v59;
    if ( v59 )
    {
      sub_1623A60((__int64)v148, (__int64)v59, 2);
      if ( v150 )
        sub_161E7C0((__int64)&v150, (__int64)v150);
      v150 = v148[0];
      if ( v148[0] )
        sub_1623210((__int64)v148, v148[0], (__int64)&v150);
    }
    v147 = 1;
    v145[0] = (__int64)"promoted";
    v146 = 3;
    if ( v136 == *(__int64 ***)a1 )
    {
      a2 = v150;
      v61 = (unsigned __int8 *)a1;
LABEL_42:
      v57[2] = v61;
      if ( a2 )
      {
        sub_161E7C0((__int64)&v150, (__int64)a2);
        v61 = (unsigned __int8 *)v57[2];
      }
      goto LABEL_44;
    }
    if ( *(_BYTE *)(a1 + 16) <= 0x10u )
    {
      v60 = sub_15A46C0(36, (__int64 ***)a1, v136, 0);
      a2 = v150;
      v61 = (unsigned __int8 *)v60;
      goto LABEL_42;
    }
    v149 = 257;
    v100 = sub_15FDBD0(36, a1, (__int64)v136, (__int64)v148, 0);
    if ( v151 )
    {
      v139 = v100;
      v125 = v152;
      sub_157E9D0(v151 + 40, v100);
      v100 = v139;
      v101 = *(_QWORD *)(v139 + 24);
      v102 = *v125;
      *(_QWORD *)(v139 + 32) = v125;
      v102 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v139 + 24) = v102 | v101 & 7;
      *(_QWORD *)(v102 + 8) = v139 + 24;
      *v125 = *v125 & 7 | (v139 + 24);
    }
    v140 = v100;
    sub_164B780(v100, v145);
    a2 = v150;
    v61 = (unsigned __int8 *)v140;
    if ( v150 )
    {
      v144 = v150;
      sub_1623A60((__int64)&v144, (__int64)v150, 2);
      v61 = (unsigned __int8 *)v140;
      v103 = *(_QWORD *)(v140 + 48);
      v104 = v140 + 48;
      if ( v103 )
      {
        sub_161E7C0(v140 + 48, v103);
        v61 = (unsigned __int8 *)v140;
        v104 = v140 + 48;
      }
      v105 = v144;
      *((_QWORD *)v61 + 6) = v144;
      if ( v105 )
      {
        v141 = v61;
        sub_1623210((__int64)&v144, v105, v104);
        v61 = v141;
      }
      a2 = v150;
      goto LABEL_42;
    }
    v57[2] = v140;
  }
  else
  {
    v61 = (unsigned __int8 *)MEMORY[0x10];
  }
LABEL_44:
  v62 = *(_DWORD *)(v16 + 8);
  if ( v62 >= *(_DWORD *)(v16 + 12) )
  {
    v142 = v61;
    sub_1D5B850(v16, (__int64)a2);
    v62 = *(_DWORD *)(v16 + 8);
    v61 = v142;
  }
  v63 = (_QWORD *)(*(_QWORD *)v16 + 8LL * v62);
  if ( v63 )
  {
    *v63 = v57;
    ++*(_DWORD *)(v16 + 8);
  }
  else
  {
    v138 = v61;
    *(_DWORD *)(v16 + 8) = v62 + 1;
    (*(void (__fastcall **)(_QWORD *))(*v57 + 8LL))(v57);
    v61 = v138;
  }
  if ( v61[16] > 0x17u )
  {
    v150 = v61;
    v137 = v61;
    sub_15F2300(v61, v19);
    v61 = v137;
    if ( a6 )
    {
      sub_14EF3D0(a6, &v150);
      v61 = v137;
    }
  }
  sub_1D5B980(v16, v19, (__int64)v61, a7, a8, a9, a10, v55, v56, a13, a14);
  sub_1D5BD90(v16, a1, 0, v19);
LABEL_5:
  v21 = *(unsigned int *)(a3 + 24);
  v148[0] = (unsigned __int8 *)v19;
  v22 = *(_QWORD *)(a3 + 8);
  if ( !(_DWORD)v21 )
  {
    v85 = *(__int64 ***)v19;
    ++*(_QWORD *)a3;
    v28 = (unsigned __int64)v85 & 0xFFFFFFFFFFFFFFF9LL | (2LL * a16);
    goto LABEL_100;
  }
  v23 = v21 - 1;
  v24 = (v21 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v25 = (__int64 *)(v22 + 16LL * v24);
  v26 = *v25;
  if ( v19 == *v25 )
  {
LABEL_7:
    if ( v25 == (__int64 *)(v22 + 16LL * (unsigned int)v21) )
    {
      v27 = 2LL * a16;
    }
    else
    {
      v27 = 4;
      if ( a16 == ((v25[1] >> 1) & 3) )
        goto LABEL_12;
    }
    v28 = v27 | *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF9LL;
  }
  else
  {
    v106 = 1;
    while ( v26 != -8 )
    {
      v107 = v106 + 1;
      v24 = v23 & (v106 + v24);
      v25 = (__int64 *)(v22 + 16LL * v24);
      v26 = *v25;
      if ( v19 == *v25 )
        goto LABEL_7;
      v106 = v107;
    }
    v28 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF9LL | (2LL * a16);
  }
  v29 = v23 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v30 = (unsigned __int8 *)(v22 + 16LL * v29);
  v31 = *(_QWORD *)v30;
  if ( v19 != *(_QWORD *)v30 )
  {
    v97 = 1;
    v98 = 0;
    while ( v31 != -8 )
    {
      if ( v31 == -16 && !v98 )
        v98 = v30;
      v29 = v23 & (v97 + v29);
      v30 = (unsigned __int8 *)(v22 + 16LL * v29);
      v31 = *(_QWORD *)v30;
      if ( v19 == *(_QWORD *)v30 )
        goto LABEL_11;
      ++v97;
    }
    v99 = *(_DWORD *)(a3 + 16);
    if ( v98 )
      v30 = v98;
    ++*(_QWORD *)a3;
    v87 = v99 + 1;
    if ( 4 * (v99 + 1) < (unsigned int)(3 * v21) )
    {
      v86 = (unsigned __int8 *)v19;
      if ( (int)v21 - *(_DWORD *)(a3 + 20) - v87 > (unsigned int)v21 >> 3 )
        goto LABEL_102;
      goto LABEL_101;
    }
LABEL_100:
    LODWORD(v21) = 2 * v21;
LABEL_101:
    sub_1D5BEF0(a3, v21);
    sub_1D5A510(a3, (__int64 *)v148, &v150);
    v21 = *(unsigned int *)(a3 + 16);
    v30 = v150;
    v86 = v148[0];
    v87 = v21 + 1;
LABEL_102:
    *(_DWORD *)(a3 + 16) = v87;
    if ( *(_QWORD *)v30 != -8 )
      --*(_DWORD *)(a3 + 20);
    *(_QWORD *)v30 = v86;
    *((_QWORD *)v30 + 1) = 0;
  }
LABEL_11:
  *((_QWORD *)v30 + 1) = v28;
LABEL_12:
  v32 = *(__int64 ***)a1;
  v33 = (_QWORD *)sub_22077B0(24);
  v36 = v33;
  if ( v33 )
  {
    v33[1] = v19;
    *v33 = off_4985648;
    v37 = *(__int64 ***)v19;
    *(_QWORD *)v19 = v32;
    v36[2] = v37;
  }
  v38 = *(_DWORD *)(v16 + 8);
  if ( v38 >= *(_DWORD *)(v16 + 12) )
  {
    sub_1D5B850(v16, v21);
    v38 = *(_DWORD *)(v16 + 8);
  }
  v39 = (_QWORD *)(*(_QWORD *)v16 + 8LL * v38);
  if ( v39 )
  {
    *v39 = v36;
    ++*(_DWORD *)(v16 + 8);
  }
  else
  {
    *(_DWORD *)(v16 + 8) = v38 + 1;
    if ( v36 )
      (*(void (__fastcall **)(_QWORD *))(*v36 + 8LL))(v36);
  }
  v40 = (unsigned __int8 *)a1;
  v41 = 0;
  v42 = (unsigned __int8 *)a1;
  sub_1D5B980(v16, a1, v19, a7, a8, a9, a10, v34, v35, a13, a14);
  v135 = *(_DWORD *)(v19 + 20) & 0xFFFFFFF;
  if ( !v135 )
    goto LABEL_123;
  do
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v19 + 23) & 0x40) != 0 )
        v47 = *(_QWORD *)(v19 - 8);
      else
        v47 = v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
      v48 = *(_QWORD *)(v47 + 24 * v41);
      v49 = *(__int64 ***)a1;
      if ( *(_QWORD *)v48 == *(_QWORD *)a1 || *(_BYTE *)(v19 + 16) == 79 && !(_DWORD)v41 )
        goto LABEL_30;
      v50 = *(_BYTE *)(v48 + 16);
      if ( v50 == 13 )
      {
        v51 = (const void **)(v48 + 24);
        v52 = *((_DWORD *)v49 + 2) >> 8;
        if ( a16 )
          sub_16A5B10((__int64)&v150, v51, v52);
        else
          sub_16A5C50((__int64)&v150, v51, v52);
        v53 = sub_15A1070(*(_QWORD *)a1, (__int64)&v150);
        v42 = (unsigned __int8 *)v19;
        sub_1D5BD90(v16, v19, v41, v53);
        if ( (unsigned int)v151 > 0x40 && v150 )
          j_j___libc_free_0_0(v150);
        goto LABEL_30;
      }
      if ( v50 == 9 )
      {
        v88 = sub_1599EF0(*(__int64 ***)a1);
        v42 = (unsigned __int8 *)v19;
        sub_1D5BD90(v16, v19, v41, v88);
        goto LABEL_30;
      }
      if ( v40 )
        goto LABEL_54;
      if ( !a16 )
      {
        v130 = v48;
        v74 = sub_1D5BAE0(v16, (unsigned __int8 *)a1, v48, v49);
        v48 = v130;
        v75 = v74;
        goto LABEL_79;
      }
      v109 = *(__int64 ***)a1;
      v116 = v48;
      v76 = (_QWORD *)sub_22077B0(24);
      v48 = v116;
      v131 = v76;
      if ( v76 )
      {
        v76[1] = a1;
        *v76 = off_49855E8;
        v77 = sub_16498A0(a1);
        v150 = 0;
        v153 = v77;
        v48 = v116;
        v78 = *(_QWORD *)(a1 + 40);
        v79 = *(unsigned __int8 **)(a1 + 48);
        v154 = 0;
        v155 = 0;
        v80 = v109;
        v151 = v78;
        v156 = 0;
        v157 = 0;
        v152 = (__int64 *)(a1 + 24);
        v148[0] = v79;
        if ( v79 )
        {
          sub_1623A60((__int64)v148, (__int64)v79, 2);
          v43 = v148;
          v48 = v116;
          v80 = v109;
          if ( v150 )
          {
            sub_161E7C0((__int64)&v150, (__int64)v150);
            v81 = v148[0];
            v43 = v148;
            v80 = v109;
            v48 = v116;
          }
          else
          {
            v81 = v148[0];
          }
          v150 = v81;
          if ( v81 )
          {
            v110 = v80;
            v117 = v48;
            sub_1623210((__int64)v148, v81, (__int64)&v150);
            v48 = v117;
            v80 = v110;
          }
        }
        v147 = 1;
        v145[0] = (__int64)"promoted";
        v146 = 3;
        if ( v80 == *(__int64 ***)v48 )
        {
          v42 = v150;
          v75 = v48;
LABEL_91:
          v131[2] = v75;
          if ( v42 )
          {
            v119 = v48;
            sub_161E7C0((__int64)&v150, (__int64)v42);
            v48 = v119;
            v75 = v131[2];
          }
          goto LABEL_93;
        }
        if ( *(_BYTE *)(v48 + 16) <= 0x10u )
        {
          v118 = v48;
          v82 = sub_15A46C0(38, (__int64 ***)v48, v80, 0);
          v42 = v150;
          v48 = v118;
          v75 = v82;
          goto LABEL_91;
        }
        v120 = v48;
        v149 = 257;
        v89 = sub_15FDBD0(38, v48, (__int64)v80, (__int64)v148, 0);
        v90 = v120;
        v91 = v89;
        if ( v151 )
        {
          v108 = v120;
          v121 = v89;
          v111 = v152;
          sub_157E9D0(v151 + 40, v89);
          v91 = v121;
          v90 = v108;
          v92 = *v111;
          v93 = *(_QWORD *)(v121 + 24);
          *(_QWORD *)(v121 + 32) = v111;
          v92 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v121 + 24) = v92 | v93 & 7;
          *(_QWORD *)(v92 + 8) = v121 + 24;
          *v111 = *v111 & 7 | (v121 + 24);
        }
        v112 = v90;
        v122 = v91;
        sub_164B780(v91, v145);
        v42 = v150;
        v75 = v122;
        v48 = v112;
        if ( v150 )
        {
          v144 = v150;
          sub_1623A60((__int64)&v144, (__int64)v150, 2);
          v75 = v122;
          v43 = &v144;
          v48 = v112;
          v94 = *(_QWORD *)(v122 + 48);
          v95 = v122 + 48;
          if ( v94 )
          {
            sub_161E7C0(v122 + 48, v94);
            v43 = &v144;
            v48 = v112;
            v75 = v122;
            v95 = v122 + 48;
          }
          v96 = v144;
          *(_QWORD *)(v75 + 48) = v144;
          if ( v96 )
          {
            v113 = v48;
            v123 = v75;
            sub_1623210((__int64)&v144, v96, v95);
            v75 = v123;
            v48 = v113;
          }
          v42 = v150;
          goto LABEL_91;
        }
        v131[2] = v122;
      }
      else
      {
        v75 = MEMORY[0x10];
      }
LABEL_93:
      v83 = *(_DWORD *)(v16 + 8);
      if ( v83 >= *(_DWORD *)(v16 + 12) )
      {
        v115 = v48;
        v126 = v75;
        sub_1D5B850(v16, (__int64)v42);
        v83 = *(_DWORD *)(v16 + 8);
        v48 = v115;
        v75 = v126;
      }
      v84 = (_QWORD *)(*(_QWORD *)v16 + 8LL * v83);
      if ( v84 )
      {
        *v84 = v131;
        ++*(_DWORD *)(v16 + 8);
        if ( *(_BYTE *)(v75 + 16) > 0x17u )
          break;
        goto LABEL_97;
      }
      v114 = v48;
      *(_DWORD *)(v16 + 8) = v83 + 1;
      v124 = v75;
      (*(void (__fastcall **)(_QWORD *))(*v131 + 8LL))(v131);
      v75 = v124;
      v48 = v114;
LABEL_79:
      if ( *(_BYTE *)(v75 + 16) > 0x17u )
        break;
LABEL_97:
      v42 = (unsigned __int8 *)v19;
      sub_1D5BD90(v16, v19, v41, v75);
LABEL_30:
      if ( v135 == (_DWORD)++v41 )
        goto LABEL_69;
    }
    v40 = (unsigned __int8 *)v75;
LABEL_54:
    if ( a5 )
    {
      v64 = *(unsigned int *)(a5 + 8);
      if ( (unsigned int)v64 >= *(_DWORD *)(a5 + 12) )
      {
        v133 = v48;
        sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, (int)v43, v44);
        v48 = v133;
        v64 = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v64) = v40;
      ++*(_DWORD *)(a5 + 8);
    }
    v65 = (__int64)v40;
    sub_1D5BD90(v16, (__int64)v40, 0, v48);
    v66 = (_QWORD *)sub_22077B0(32);
    v67 = v66;
    if ( v66 )
    {
      v66[1] = v40;
      *v66 = off_4985528;
      v68 = *((_QWORD *)v40 + 5);
      v69 = v40 + 24 == *(unsigned __int8 **)(v68 + 48);
      *((_BYTE *)v67 + 24) = v40 + 24 != *(unsigned __int8 **)(v68 + 48);
      if ( v69 )
      {
        v67[2] = v68;
      }
      else
      {
        v70 = 0;
        if ( (*((_QWORD *)v40 + 3) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          v70 = (*((_QWORD *)v40 + 3) & 0xFFFFFFFFFFFFFFF8LL) - 24;
        v67[2] = v70;
      }
      v65 = v19;
      v129 = v67;
      sub_15F22F0(v40, v19);
      v67 = v129;
    }
    v71 = *(_DWORD *)(v16 + 8);
    if ( v71 >= *(_DWORD *)(v16 + 12) )
    {
      v132 = v67;
      sub_1D5B850(v16, v65);
      v71 = *(_DWORD *)(v16 + 8);
      v67 = v132;
    }
    v72 = (_QWORD *)(*(_QWORD *)v16 + 8LL * v71);
    if ( v72 )
    {
      *v72 = v67;
      ++*(_DWORD *)(v16 + 8);
    }
    else
    {
      *(_DWORD *)(v16 + 8) = v71 + 1;
      if ( v67 )
        (*(void (__fastcall **)(_QWORD *))(*v67 + 8LL))(v67);
    }
    sub_1D5BD90(v16, v19, v41, (__int64)v40);
    v42 = v40;
    ++v41;
    v40 = 0;
    *a4 += (unsigned __int8)sub_1D5EF60(a15, v42) ^ 1;
  }
  while ( v135 != (_DWORD)v41 );
LABEL_69:
  if ( (unsigned __int8 *)a1 == v40 )
LABEL_123:
    sub_1D5C680(v16, a1, 0, a7, a8, a9, a10, v45, v46, a13, a14);
  return v19;
}
