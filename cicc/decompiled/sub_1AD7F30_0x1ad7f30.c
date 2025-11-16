// Function: sub_1AD7F30
// Address: 0x1ad7f30
//
void __fastcall sub_1AD7F30(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r9
  __int64 v16; // rax
  char v17; // di
  unsigned int v18; // esi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r15
  int v27; // edx
  unsigned __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rbx
  unsigned __int64 v31; // rax
  _QWORD *v32; // r13
  __int64 v33; // r14
  _QWORD *v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r13
  __int64 v39; // r9
  __int64 *v40; // r14
  __int64 v41; // r12
  __int64 *v42; // rbx
  __int64 v43; // rsi
  __int64 v44; // r15
  __int64 v45; // rsi
  int v46; // eax
  __int64 v47; // rax
  int v48; // edx
  __int64 v49; // rdx
  _QWORD *v50; // rax
  __int64 v51; // rdi
  unsigned __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 *v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rdx
  int v60; // r14d
  int v61; // r14d
  __int64 **v62; // rax
  __int64 v63; // rax
  __int64 v64; // r13
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 *v67; // rdx
  __int64 *v68; // rax
  __int64 *v69; // r14
  __int64 *v70; // rbx
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  int v77; // esi
  __int64 v78; // rdx
  __int64 *v79; // rax
  __int64 v80; // rcx
  __int64 v81; // rsi
  double v82; // xmm4_8
  double v83; // xmm5_8
  __int64 v84; // rdx
  __int64 v85; // r9
  __int64 *v86; // r15
  __int64 v87; // r13
  __int64 v88; // r8
  __int64 *v89; // r14
  __int64 v90; // rdx
  __int64 *v91; // rax
  __int64 v92; // rdi
  unsigned __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rcx
  __int64 v98; // r12
  int v99; // eax
  __int64 v100; // rax
  int v101; // edx
  __int64 v102; // rax
  int v103; // esi
  __int64 *v104; // r15
  unsigned int v105; // eax
  __int64 *v106; // r13
  __int64 v107; // rdx
  __int64 *v108; // rax
  __int64 v109; // r14
  __int64 v110; // r13
  __int64 v111; // rdi
  __int64 v112; // rsi
  __int64 v113; // rdx
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 *v116; // r15
  __int64 v117; // rbx
  __int64 *v118; // r14
  __int64 v119; // r13
  __int64 v120; // rcx
  __int64 v121; // r12
  __int64 v122; // rcx
  int v123; // eax
  __int64 v124; // rax
  int v125; // edx
  __int64 v126; // rdx
  _QWORD *v127; // rax
  unsigned __int64 v128; // rdx
  __int64 v129; // rdx
  __int64 v130; // rdx
  __int64 v131; // rax
  int v132; // r9d
  __int64 *v133; // r8
  int v134; // edx
  int v135; // r14d
  __int64 *v136; // r10
  int v137; // ecx
  __int64 v138; // rdi
  __int64 v139; // [rsp+0h] [rbp-120h]
  __int64 v141; // [rsp+18h] [rbp-108h]
  __int64 v142; // [rsp+20h] [rbp-100h]
  __int64 v143; // [rsp+20h] [rbp-100h]
  __int64 v144; // [rsp+28h] [rbp-F8h]
  __int64 *v145; // [rsp+28h] [rbp-F8h]
  __int64 v146; // [rsp+28h] [rbp-F8h]
  __int64 v147; // [rsp+28h] [rbp-F8h]
  __int64 v148; // [rsp+28h] [rbp-F8h]
  __int64 v149; // [rsp+30h] [rbp-F0h]
  __int64 v150; // [rsp+38h] [rbp-E8h]
  __int64 v151; // [rsp+40h] [rbp-E0h]
  __int64 v152; // [rsp+40h] [rbp-E0h]
  __int64 v153; // [rsp+40h] [rbp-E0h]
  __int64 v154; // [rsp+48h] [rbp-D8h]
  __int64 v155[2]; // [rsp+50h] [rbp-D0h] BYREF
  _QWORD v156[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v157; // [rsp+70h] [rbp-B0h]
  __int64 v158; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v159; // [rsp+88h] [rbp-98h]
  __int64 v160; // [rsp+90h] [rbp-90h]
  unsigned int v161; // [rsp+98h] [rbp-88h]
  __int64 *v162; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v163; // [rsp+A8h] [rbp-78h]
  _BYTE v164[112]; // [rsp+B0h] [rbp-70h] BYREF

  v12 = *(_QWORD *)(a1 - 24);
  v149 = *(_QWORD *)(a2 + 56);
  v13 = *(_QWORD *)(v12 + 48);
  v14 = v12 + 40;
  v162 = (__int64 *)v164;
  v163 = 0x800000000LL;
  v150 = v12;
  v141 = *(_QWORD *)(a1 + 40);
  while ( v14 != v13 )
  {
    if ( !v13 )
      BUG();
    if ( *(_BYTE *)(v13 - 8) != 77 )
      break;
    v15 = v13 - 24;
    v16 = 0x17FFFFFFE8LL;
    v17 = *(_BYTE *)(v13 - 1) & 0x40;
    v18 = *(_DWORD *)(v13 - 4) & 0xFFFFFFF;
    if ( v18 )
    {
      v19 = 24LL * *(unsigned int *)(v13 + 32) + 8;
      v20 = 0;
      do
      {
        v21 = v15 - 24LL * v18;
        if ( v17 )
          v21 = *(_QWORD *)(v13 - 32);
        if ( v141 == *(_QWORD *)(v21 + v19) )
        {
          v16 = 24 * v20;
          goto LABEL_11;
        }
        ++v20;
        v19 += 8;
      }
      while ( v18 != (_DWORD)v20 );
      v16 = 0x17FFFFFFE8LL;
    }
LABEL_11:
    if ( v17 )
    {
      v22 = *(_QWORD *)(v13 - 32);
      v23 = *(_QWORD *)(v22 + v16);
      v24 = (unsigned int)v163;
      if ( (unsigned int)v163 >= HIDWORD(v163) )
        goto LABEL_95;
    }
    else
    {
      v22 = v15 - 24LL * v18;
      v23 = *(_QWORD *)(v22 + v16);
      v24 = (unsigned int)v163;
      if ( (unsigned int)v163 >= HIDWORD(v163) )
      {
LABEL_95:
        v154 = v23;
        sub_16CD150((__int64)&v162, v164, 0, 8, v23, v22);
        v24 = (unsigned int)v163;
        v23 = v154;
      }
    }
    v162[v24] = v23;
    LODWORD(v163) = v163 + 1;
    v13 = *(_QWORD *)(v13 + 8);
  }
  v158 = 0;
  v25 = a2 + 24;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v139 = v25;
  if ( v25 == v149 + 72 )
    goto LABEL_89;
  do
  {
    v30 = v25 - 24;
    if ( !v25 )
      v30 = 0;
    v31 = sub_157EBA0(v30);
    v32 = (_QWORD *)v31;
    if ( *(_BYTE *)(v31 + 16) == 32 && (*(_BYTE *)(v31 + 18) & 1) == 0 )
    {
      v33 = *(_QWORD *)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
      v34 = sub_1648A60(56, 2u);
      if ( v34 )
        sub_15F76D0((__int64)v34, v33, v150, 2u, (__int64)v32);
      sub_15F20C0(v32);
      v38 = *(_QWORD *)(v150 + 48);
      v39 = (__int64)&v162[(unsigned int)v163];
      if ( v162 != (__int64 *)v39 )
      {
        v144 = v33;
        v40 = &v162[(unsigned int)v163];
        v142 = v25;
        v41 = v30;
        v42 = v162;
        do
        {
          v44 = *v42;
          if ( !v38 )
            BUG();
          v45 = v38 - 24;
          v46 = *(_DWORD *)(v38 - 4) & 0xFFFFFFF;
          if ( v46 == *(_DWORD *)(v38 + 32) )
          {
            sub_15F55D0(v38 - 24, v45, v35, v36, v37, v39);
            v45 = v38 - 24;
            v46 = *(_DWORD *)(v38 - 4) & 0xFFFFFFF;
          }
          v47 = (v46 + 1) & 0xFFFFFFF;
          v48 = v47 | *(_DWORD *)(v38 - 4) & 0xF0000000;
          *(_DWORD *)(v38 - 4) = v48;
          if ( (v48 & 0x40000000) != 0 )
            v49 = *(_QWORD *)(v38 - 32);
          else
            v49 = v45 - 24 * v47;
          v50 = (_QWORD *)(v49 + 24LL * (unsigned int)(v47 - 1));
          if ( *v50 )
          {
            v51 = v50[1];
            v52 = v50[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v52 = v51;
            if ( v51 )
            {
              v37 = *(_QWORD *)(v51 + 16) & 3LL;
              *(_QWORD *)(v51 + 16) = v37 | v52;
            }
          }
          *v50 = v44;
          if ( v44 )
          {
            v53 = *(_QWORD *)(v44 + 8);
            v37 = v44 + 8;
            v50[1] = v53;
            if ( v53 )
              *(_QWORD *)(v53 + 16) = (unsigned __int64)(v50 + 1) | *(_QWORD *)(v53 + 16) & 3LL;
            v50[2] = v37 | v50[2] & 3LL;
            *(_QWORD *)(v44 + 8) = v50;
          }
          v54 = *(_DWORD *)(v38 - 4) & 0xFFFFFFF;
          v55 = (unsigned int)(v54 - 1);
          if ( (*(_BYTE *)(v38 - 1) & 0x40) != 0 )
            v43 = *(_QWORD *)(v38 - 32);
          else
            v43 = v45 - 24 * v54;
          ++v42;
          v35 = 3LL * *(unsigned int *)(v38 + 32);
          *(_QWORD *)(v43 + 8 * v55 + 24LL * *(unsigned int *)(v38 + 32) + 8) = v41;
          v38 = *(_QWORD *)(v38 + 8);
        }
        while ( v40 != v42 );
        v30 = v41;
        v33 = v144;
        v25 = v142;
      }
      v102 = sub_15E0530(v149);
      v103 = v161;
      v155[0] = v33;
      v104 = (__int64 *)v102;
      if ( v161 )
      {
        v105 = (v161 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v106 = (__int64 *)(v159 + 16LL * v105);
        v107 = *v106;
        if ( v33 == *v106 )
        {
LABEL_87:
          v106[1] = sub_1594470(v104);
          goto LABEL_16;
        }
        v132 = 1;
        v133 = 0;
        while ( v107 != -8 )
        {
          if ( v107 == -16 && !v133 )
            v133 = v106;
          v105 = (v161 - 1) & (v132 + v105);
          v106 = (__int64 *)(v159 + 16LL * v105);
          v107 = *v106;
          if ( v33 == *v106 )
            goto LABEL_87;
          ++v132;
        }
        if ( v133 )
          v106 = v133;
        ++v158;
        v134 = v160 + 1;
        if ( 4 * ((int)v160 + 1) < 3 * v161 )
        {
          if ( v161 - HIDWORD(v160) - v134 > v161 >> 3 )
          {
LABEL_134:
            LODWORD(v160) = v134;
            if ( *v106 != -8 )
              --HIDWORD(v160);
            *v106 = v33;
            v106[1] = 0;
            goto LABEL_87;
          }
LABEL_139:
          sub_19566A0((__int64)&v158, v103);
          sub_1954890((__int64)&v158, v155, v156);
          v106 = (__int64 *)v156[0];
          v33 = v155[0];
          v134 = v160 + 1;
          goto LABEL_134;
        }
      }
      else
      {
        ++v158;
      }
      v103 = 2 * v161;
      goto LABEL_139;
    }
LABEL_16:
    v26 = sub_157ED20(v30);
    v27 = *(unsigned __int8 *)(v26 + 16);
    v28 = (unsigned int)(v27 - 34);
    if ( (unsigned int)v28 > 0x36 )
      goto LABEL_18;
    v29 = 0x40018000000001LL;
    if ( !_bittest64(&v29, v28) || (_BYTE)v27 != 34 || (*(_BYTE *)(v26 + 18) & 1) != 0 )
      goto LABEL_18;
    if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
    {
      v56 = *(__int64 **)(v26 - 8);
      v57 = *v56;
      if ( *(_BYTE *)(*v56 + 16) <= 0x17u )
        goto LABEL_93;
    }
    else
    {
      v57 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v57 + 16) <= 0x17u )
      {
LABEL_93:
        v108 = (__int64 *)sub_15E0530(v149);
        v151 = sub_1594470(v108);
        goto LABEL_50;
      }
    }
    v58 = sub_1AD6830(v57, (__int64)&v158);
    v151 = v58;
    if ( v58 && *(_BYTE *)(v58 + 16) != 16 )
      goto LABEL_18;
LABEL_50:
    v155[0] = (__int64)sub_1649960(v26);
    v157 = 261;
    v155[1] = v59;
    v156[0] = v155;
    v60 = *(_DWORD *)(v26 + 20);
    if ( (*(_BYTE *)(v26 + 18) & 1) != 0 )
    {
      v61 = (v60 & 0xFFFFFFF) - 2;
      if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
      {
LABEL_52:
        v62 = *(__int64 ***)(v26 - 8);
        goto LABEL_53;
      }
    }
    else
    {
      v61 = (v60 & 0xFFFFFFF) - 1;
      if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
        goto LABEL_52;
    }
    v62 = (__int64 **)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
LABEL_53:
    v145 = *v62;
    v63 = sub_1648B60(64);
    v64 = v63;
    if ( v63 )
      sub_15F7B50(v63, v145, v150, v61, (__int64)v156, v26);
    if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
    {
      v65 = *(_QWORD *)(v26 - 8);
      v66 = v65 + 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF);
    }
    else
    {
      v66 = v26;
      v65 = v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF);
    }
    v67 = (__int64 *)(v65 + 24);
    v68 = (__int64 *)(v65 + 48);
    if ( (*(_BYTE *)(v26 + 18) & 1) != 0 )
      v67 = v68;
    if ( v67 != (__int64 *)v66 )
    {
      v146 = v30;
      v69 = (__int64 *)v66;
      v70 = v67;
      do
      {
        v71 = *v70;
        v70 += 3;
        v72 = sub_15A5110(v71);
        sub_15F7DB0(v64, v72, v73, v74, v75, v76);
      }
      while ( v69 != v70 );
      v30 = v146;
    }
    v77 = v161;
    v155[0] = v64;
    if ( !v161 )
    {
      ++v158;
      goto LABEL_150;
    }
    LODWORD(v78) = (v161 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
    v79 = (__int64 *)(v159 + 16LL * (unsigned int)v78);
    v80 = *v79;
    if ( *v79 != v64 )
    {
      v135 = 1;
      v136 = 0;
      while ( v80 != -8 )
      {
        if ( v80 == -16 && !v136 )
          v136 = v79;
        v78 = (v161 - 1) & ((_DWORD)v78 + v135);
        v79 = (__int64 *)(v159 + 16 * v78);
        v80 = *v79;
        if ( v64 == *v79 )
          goto LABEL_65;
        ++v135;
      }
      if ( v136 )
        v79 = v136;
      ++v158;
      v137 = v160 + 1;
      if ( 4 * ((int)v160 + 1) < 3 * v161 )
      {
        v138 = v64;
        if ( v161 - HIDWORD(v160) - v137 > v161 >> 3 )
        {
LABEL_146:
          LODWORD(v160) = v137;
          if ( *v79 != -8 )
            --HIDWORD(v160);
          *v79 = v138;
          v79[1] = 0;
          goto LABEL_65;
        }
LABEL_151:
        sub_19566A0((__int64)&v158, v77);
        sub_1954890((__int64)&v158, v155, v156);
        v79 = (__int64 *)v156[0];
        v138 = v155[0];
        v137 = v160 + 1;
        goto LABEL_146;
      }
LABEL_150:
      v77 = 2 * v161;
      goto LABEL_151;
    }
LABEL_65:
    v79[1] = v151;
    if ( v64 )
    {
      sub_164B7C0(v64, v26);
      v81 = v64;
      sub_164D160(v26, v64, a4, a5, a6, a7, v82, v83, a10, a11);
      sub_15F20C0((_QWORD *)v26);
      v86 = v162;
      v87 = *(_QWORD *)(v150 + 48);
      v88 = (__int64)&v162[(unsigned int)v163];
      if ( v162 != (__int64 *)v88 )
      {
        v147 = v25;
        v89 = &v162[(unsigned int)v163];
        do
        {
          v97 = *v86;
          if ( !v87 )
            BUG();
          v98 = v87 - 24;
          v99 = *(_DWORD *)(v87 - 4) & 0xFFFFFFF;
          if ( v99 == *(_DWORD *)(v87 + 32) )
          {
            v153 = *v86;
            sub_15F55D0(v87 - 24, v81, v84, v97, v88, v85);
            v97 = v153;
            v99 = *(_DWORD *)(v87 - 4) & 0xFFFFFFF;
          }
          v100 = (v99 + 1) & 0xFFFFFFF;
          v101 = v100 | *(_DWORD *)(v87 - 4) & 0xF0000000;
          *(_DWORD *)(v87 - 4) = v101;
          if ( (v101 & 0x40000000) != 0 )
            v90 = *(_QWORD *)(v87 - 32);
          else
            v90 = v98 - 24 * v100;
          v91 = (__int64 *)(v90 + 24LL * (unsigned int)(v100 - 1));
          if ( *v91 )
          {
            v92 = v91[1];
            v93 = v91[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v93 = v92;
            if ( v92 )
            {
              v88 = *(_QWORD *)(v92 + 16) & 3LL;
              *(_QWORD *)(v92 + 16) = v88 | v93;
            }
          }
          *v91 = v97;
          if ( v97 )
          {
            v94 = *(_QWORD *)(v97 + 8);
            v88 = v97 + 8;
            v91[1] = v94;
            if ( v94 )
            {
              v85 = (__int64)(v91 + 1);
              *(_QWORD *)(v94 + 16) = (unsigned __int64)(v91 + 1) | *(_QWORD *)(v94 + 16) & 3LL;
            }
            v91[2] = v88 | v91[2] & 3;
            *(_QWORD *)(v97 + 8) = v91;
          }
          v95 = *(_DWORD *)(v87 - 4) & 0xFFFFFFF;
          v96 = (unsigned int)(v95 - 1);
          if ( (*(_BYTE *)(v87 - 1) & 0x40) != 0 )
            v81 = *(_QWORD *)(v87 - 32);
          else
            v81 = v98 - 24 * v95;
          ++v86;
          v84 = 3LL * *(unsigned int *)(v87 + 32);
          *(_QWORD *)(v81 + 8 * v96 + 24LL * *(unsigned int *)(v87 + 32) + 8) = v30;
          v87 = *(_QWORD *)(v87 + 8);
        }
        while ( v89 != v86 );
        v25 = v147;
      }
    }
LABEL_18:
    v25 = *(_QWORD *)(v25 + 8);
  }
  while ( v149 + 72 != v25 );
  v152 = v25;
  if ( *a3 )
  {
    v109 = v150;
    v110 = v139;
    do
    {
      v111 = v110 - 24;
      v112 = v109;
      if ( !v110 )
        v111 = 0;
      v115 = sub_1AD6F20(v111, v109, (__int64)&v158);
      if ( v115 )
      {
        v116 = v162;
        v117 = *(_QWORD *)(v109 + 48);
        if ( v162 != &v162[(unsigned int)v163] )
        {
          v148 = v109;
          v118 = &v162[(unsigned int)v163];
          v143 = v110;
          v119 = v115;
          do
          {
            v121 = *v116;
            if ( !v117 )
              BUG();
            v122 = v117 - 24;
            v123 = *(_DWORD *)(v117 - 4) & 0xFFFFFFF;
            if ( v123 == *(_DWORD *)(v117 + 32) )
            {
              sub_15F55D0(v117 - 24, v112, v113, v122, v114, v115);
              v122 = v117 - 24;
              v123 = *(_DWORD *)(v117 - 4) & 0xFFFFFFF;
            }
            v124 = (v123 + 1) & 0xFFFFFFF;
            v112 = (unsigned int)(v124 - 1);
            v125 = v124 | *(_DWORD *)(v117 - 4) & 0xF0000000;
            *(_DWORD *)(v117 - 4) = v125;
            if ( (v125 & 0x40000000) != 0 )
              v126 = *(_QWORD *)(v117 - 32);
            else
              v126 = v122 - 24 * v124;
            v127 = (_QWORD *)(v126 + 24LL * (unsigned int)v112);
            if ( *v127 )
            {
              v112 = v127[1];
              v128 = v127[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v128 = v112;
              if ( v112 )
                *(_QWORD *)(v112 + 16) = *(_QWORD *)(v112 + 16) & 3LL | v128;
            }
            *v127 = v121;
            if ( v121 )
            {
              v129 = *(_QWORD *)(v121 + 8);
              v127[1] = v129;
              if ( v129 )
              {
                v114 = (__int64)(v127 + 1);
                v112 = (unsigned __int64)(v127 + 1) | *(_QWORD *)(v129 + 16) & 3LL;
                *(_QWORD *)(v129 + 16) = v112;
              }
              v127[2] = (v121 + 8) | v127[2] & 3LL;
              *(_QWORD *)(v121 + 8) = v127;
            }
            v130 = *(_DWORD *)(v117 - 4) & 0xFFFFFFF;
            v131 = (unsigned int)(v130 - 1);
            if ( (*(_BYTE *)(v117 - 1) & 0x40) != 0 )
              v120 = *(_QWORD *)(v117 - 32);
            else
              v120 = v122 - 24 * v130;
            ++v116;
            v113 = 3LL * *(unsigned int *)(v117 + 32);
            *(_QWORD *)(v120 + 8 * v131 + 24LL * *(unsigned int *)(v117 + 32) + 8) = v119;
            v117 = *(_QWORD *)(v117 + 8);
          }
          while ( v118 != v116 );
          v109 = v148;
          v110 = v143;
        }
      }
      v110 = *(_QWORD *)(v110 + 8);
    }
    while ( v152 != v110 );
  }
LABEL_89:
  sub_157F2D0(v150, v141, 0);
  j___libc_free_0(v159);
  if ( v162 != (__int64 *)v164 )
    _libc_free((unsigned __int64)v162);
}
