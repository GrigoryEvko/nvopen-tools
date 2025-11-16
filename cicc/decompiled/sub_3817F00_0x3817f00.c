// Function: sub_3817F00
// Address: 0x3817f00
//
void __fastcall sub_3817F00(__int64 a1, __int64 a2, _QWORD **a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rax
  unsigned int v12; // r15d
  __int64 v13; // rax
  __int16 v14; // dx
  __int16 *v15; // rax
  __int16 v16; // dx
  __int64 v17; // rdx
  unsigned int v18; // r15d
  __int64 v19; // rdx
  unsigned __int64 v20; // r10
  int v21; // eax
  unsigned int v22; // edx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // r8
  _QWORD *v25; // r13
  int v26; // eax
  __int128 v27; // rax
  __int64 v28; // r9
  int v29; // edx
  _QWORD *v30; // r12
  __int128 v31; // rax
  __int64 v32; // r9
  int v33; // edx
  int v34; // eax
  _QWORD *v35; // r12
  __int128 v36; // rax
  __int64 v37; // r9
  int v38; // edx
  unsigned __int64 v39; // rax
  int v40; // eax
  _QWORD *v41; // r13
  unsigned int v42; // eax
  int v43; // eax
  __int128 v44; // rax
  __int64 v45; // r9
  __int128 v46; // rax
  __int128 v47; // rax
  __int64 v48; // r9
  __int128 v49; // rax
  __int64 v50; // r9
  int v51; // edx
  _QWORD *v52; // r14
  __int128 v53; // rax
  __int64 v54; // r9
  int v55; // edx
  __int64 v56; // rcx
  unsigned int v57; // edx
  _QWORD *v58; // r12
  __int128 v59; // rax
  __int64 v60; // r9
  int v61; // edx
  unsigned __int64 v62; // rdx
  unsigned int v63; // r15d
  _QWORD *v64; // rax
  unsigned __int64 v65; // rax
  _QWORD *v66; // r15
  int v67; // eax
  __int128 v68; // rax
  __int64 v69; // r9
  int v70; // edx
  int v71; // edx
  unsigned __int8 *v72; // rax
  unsigned __int64 v73; // r10
  int v74; // edx
  _QWORD *v75; // r14
  int v76; // eax
  __int128 v77; // rax
  __int64 v78; // r9
  int v79; // edx
  int v80; // eax
  unsigned int v81; // edx
  unsigned __int8 *v82; // rax
  int v83; // edx
  _QWORD *v84; // r13
  __int128 v85; // rax
  __int64 v86; // r9
  unsigned __int8 *v87; // rax
  __int64 v88; // r10
  int v89; // edx
  unsigned int v90; // eax
  _QWORD *v91; // r13
  unsigned __int64 v92; // rdx
  unsigned __int64 v93; // rdx
  bool v94; // zf
  unsigned __int64 v95; // rax
  unsigned int v96; // eax
  int v97; // eax
  __int128 v98; // rax
  __int64 v99; // r9
  __int128 v100; // rax
  _QWORD *v101; // r14
  __int128 v102; // rax
  __int64 v103; // r9
  __int128 v104; // rax
  __int64 v105; // r9
  int v106; // edx
  unsigned __int64 v107; // rax
  _QWORD *v108; // r13
  unsigned int v109; // eax
  int v110; // eax
  __int128 v111; // rax
  __int64 v112; // r9
  __int128 v113; // rax
  __int128 v114; // rax
  __int64 v115; // r9
  __int128 v116; // rax
  __int64 v117; // r9
  int v118; // edx
  _QWORD *v119; // r14
  __int128 v120; // rax
  __int64 v121; // r9
  int v122; // edx
  unsigned __int64 v123; // rax
  int v124; // edx
  __int64 v125; // rcx
  __int64 v126; // r8
  int v127; // edx
  unsigned __int64 v128; // rax
  unsigned __int64 v129; // rsi
  unsigned __int64 v130; // rdx
  _QWORD *v131; // [rsp+8h] [rbp-208h]
  unsigned int v132; // [rsp+10h] [rbp-200h]
  __int128 v133; // [rsp+10h] [rbp-200h]
  unsigned __int64 v134; // [rsp+10h] [rbp-200h]
  unsigned int v135; // [rsp+10h] [rbp-200h]
  unsigned __int64 v136; // [rsp+10h] [rbp-200h]
  __int128 v137; // [rsp+10h] [rbp-200h]
  int v138; // [rsp+20h] [rbp-1F0h]
  unsigned int v139; // [rsp+20h] [rbp-1F0h]
  __int64 v140; // [rsp+20h] [rbp-1F0h]
  _QWORD *v141; // [rsp+20h] [rbp-1F0h]
  unsigned int v142; // [rsp+20h] [rbp-1F0h]
  unsigned __int64 v143; // [rsp+20h] [rbp-1F0h]
  unsigned int v144; // [rsp+20h] [rbp-1F0h]
  unsigned __int64 v145; // [rsp+20h] [rbp-1F0h]
  __int64 v146; // [rsp+20h] [rbp-1F0h]
  __int128 v147; // [rsp+20h] [rbp-1F0h]
  unsigned int v148; // [rsp+20h] [rbp-1F0h]
  __int64 v149; // [rsp+20h] [rbp-1F0h]
  _QWORD *v150; // [rsp+20h] [rbp-1F0h]
  _QWORD *v151; // [rsp+30h] [rbp-1E0h]
  unsigned __int8 *v153; // [rsp+A0h] [rbp-170h]
  __int64 v154; // [rsp+170h] [rbp-A0h] BYREF
  int v155; // [rsp+178h] [rbp-98h]
  __int128 v156; // [rsp+180h] [rbp-90h] BYREF
  __int128 v157; // [rsp+190h] [rbp-80h] BYREF
  unsigned int v158; // [rsp+1A0h] [rbp-70h] BYREF
  __int64 v159; // [rsp+1A8h] [rbp-68h]
  unsigned __int64 v160; // [rsp+1B0h] [rbp-60h] BYREF
  unsigned int v161; // [rsp+1B8h] [rbp-58h]
  __int64 v162; // [rsp+1C0h] [rbp-50h] BYREF
  __int64 v163; // [rsp+1C8h] [rbp-48h]
  __int64 v164; // [rsp+1D0h] [rbp-40h] BYREF
  __int64 v165; // [rsp+1D8h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v154 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v154, v9, 1);
  v10 = *(_DWORD *)(a2 + 72);
  DWORD2(v156) = 0;
  DWORD2(v157) = 0;
  v155 = v10;
  v11 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)&v156 = 0;
  *(_QWORD *)&v157 = 0;
  sub_375E510(a1, *(_QWORD *)v11, *(_QWORD *)(v11 + 8), (__int64)&v156, (__int64)&v157);
  v12 = *((_DWORD *)a3 + 2);
  if ( v12 <= 0x40 )
  {
    if ( !*a3 )
      goto LABEL_5;
  }
  else if ( v12 == (unsigned int)sub_C444A0((__int64)a3) )
  {
LABEL_5:
    *(_QWORD *)a4 = v156;
    *(_DWORD *)(a4 + 8) = DWORD2(v156);
    *(_QWORD *)a5 = v157;
    *(_DWORD *)(a5 + 8) = DWORD2(v157);
    goto LABEL_6;
  }
  v13 = *(_QWORD *)(v156 + 48) + 16LL * DWORD2(v156);
  v14 = *(_WORD *)v13;
  v159 = *(_QWORD *)(v13 + 8);
  v15 = *(__int16 **)(a2 + 48);
  LOWORD(v158) = v14;
  v16 = *v15;
  v163 = *((_QWORD *)v15 + 1);
  LOWORD(v162) = v16;
  v164 = sub_2D5B750((unsigned __int16 *)&v162);
  v165 = v17;
  v18 = sub_CA1930(&v164);
  v164 = sub_2D5B750((unsigned __int16 *)&v158);
  v165 = v19;
  LODWORD(v20) = sub_CA1930(&v164);
  v21 = *(_DWORD *)(a2 + 24);
  if ( v21 == 190 )
  {
    v62 = v18;
    v63 = *((_DWORD *)a3 + 2);
    if ( v63 > 0x40 )
    {
      v134 = v62;
      v142 = v20;
      if ( v63 - (unsigned int)sub_C444A0((__int64)a3) > 0x40 )
        goto LABEL_43;
      v64 = (_QWORD *)**a3;
      if ( v134 <= (unsigned __int64)v64 )
        goto LABEL_43;
      v20 = v142;
      if ( v142 >= (unsigned __int64)v64 )
        goto LABEL_45;
    }
    else
    {
      v64 = *a3;
      if ( v62 <= (unsigned __int64)*a3 )
        goto LABEL_43;
      v20 = (unsigned int)v20;
      if ( (unsigned __int64)v64 <= (unsigned int)v20 )
      {
LABEL_45:
        if ( v64 == (_QWORD *)v20 )
        {
          *(_QWORD *)a4 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v154, v158, v159, 0, a6, 0);
          *(_DWORD *)(a4 + 8) = v124;
          *(_QWORD *)a5 = v156;
          *(_DWORD *)(a5 + 8) = DWORD2(v156);
          goto LABEL_6;
        }
        v84 = *(_QWORD **)(a1 + 8);
        v145 = v20;
        *(_QWORD *)&v85 = sub_3400EC0((__int64)v84, (__int64)a3, v158, v159, (__int64)&v154, a6);
        v87 = sub_3406EB0(v84, 0xBEu, (__int64)&v154, v158, v159, v86, v156, v85);
        v88 = v145;
        *(_QWORD *)a4 = v87;
        *(_DWORD *)(a4 + 8) = v89;
        v91 = *(_QWORD **)(a1 + 8);
        v161 = *((_DWORD *)a3 + 2);
        v90 = v161;
        if ( v161 > 0x40 )
        {
          sub_C43780((__int64)&v160, (const void **)a3);
          v90 = v161;
          v88 = v145;
          if ( v161 > 0x40 )
          {
            sub_C43D10((__int64)&v160);
            v88 = v145;
LABEL_51:
            v146 = v88;
            sub_C46250((__int64)&v160);
            v96 = v161;
            v161 = 0;
            LODWORD(v163) = v96;
            v162 = v160;
            sub_C46A40((__int64)&v162, v146);
            v97 = v163;
            LODWORD(v163) = 0;
            LODWORD(v165) = v97;
            v164 = v162;
            *(_QWORD *)&v98 = sub_3400EC0((__int64)v91, (__int64)&v164, v158, v159, (__int64)&v154, a6);
            *(_QWORD *)&v100 = sub_3406EB0(v91, 0xC0u, (__int64)&v154, v158, v159, v99, v156, v98);
            v101 = *(_QWORD **)(a1 + 8);
            v147 = v100;
            *(_QWORD *)&v102 = sub_3400EC0((__int64)v101, (__int64)a3, v158, v159, (__int64)&v154, a6);
            *(_QWORD *)&v104 = sub_3406EB0(v101, 0xBEu, (__int64)&v154, v158, v159, v103, v157, v102);
            *(_QWORD *)a5 = sub_3406EB0(v91, 0xBBu, (__int64)&v154, v158, v159, v105, v104, v147);
            *(_DWORD *)(a5 + 8) = v106;
            sub_969240(&v164);
            sub_969240(&v162);
            sub_969240((__int64 *)&v160);
            goto LABEL_6;
          }
          v92 = v160;
        }
        else
        {
          v92 = (unsigned __int64)*a3;
        }
        v93 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v90) & ~v92;
        v94 = v90 == 0;
        v95 = 0;
        if ( !v94 )
          v95 = v93;
        v160 = v95;
        goto LABEL_51;
      }
    }
    v143 = v20;
    v72 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v154, v158, v159, 0, a6, 0);
    v73 = v143;
    *(_QWORD *)a4 = v72;
    *(_DWORD *)(a4 + 8) = v74;
    v75 = *(_QWORD **)(a1 + 8);
    LODWORD(v163) = *((_DWORD *)a3 + 2);
    if ( (unsigned int)v163 > 0x40 )
    {
      sub_C43780((__int64)&v162, (const void **)a3);
      v73 = v143;
    }
    else
    {
      v162 = (__int64)*a3;
    }
    sub_C46F20((__int64)&v162, v73);
    v76 = v163;
    LODWORD(v163) = 0;
    LODWORD(v165) = v76;
    v164 = v162;
    *(_QWORD *)&v77 = sub_3400EC0((__int64)v75, (__int64)&v164, v158, v159, (__int64)&v154, a6);
    *(_QWORD *)a5 = sub_3406EB0(v75, 0xBEu, (__int64)&v154, v158, v159, v78, v156, v77);
    *(_DWORD *)(a5 + 8) = v79;
    sub_969240(&v164);
    sub_969240(&v162);
    goto LABEL_6;
  }
  v22 = *((_DWORD *)a3 + 2);
  if ( v21 != 192 )
  {
    if ( v22 > 0x40 )
    {
      v132 = *((_DWORD *)a3 + 2);
      v139 = v20;
      v34 = sub_C444A0((__int64)a3);
      LODWORD(v20) = v139;
      if ( v132 - v34 <= 0x40 )
      {
        v39 = **a3;
        if ( v18 > v39 )
        {
          if ( v139 < v39 )
          {
            LODWORD(v163) = v132;
            v25 = *(_QWORD **)(a1 + 8);
            sub_C43780((__int64)&v162, (const void **)a3);
            LODWORD(v20) = v139;
            v24 = v139;
            goto LABEL_16;
          }
          v131 = *a3;
          v40 = sub_C444A0((__int64)a3);
          LODWORD(v20) = v139;
          if ( v132 - v40 > 0x40 || v139 != *v131 )
          {
            v161 = v132;
            v41 = *(_QWORD **)(a1 + 8);
            sub_C43780((__int64)&v160, (const void **)a3);
            v22 = v161;
            v24 = v139;
            if ( v161 > 0x40 )
            {
              sub_C43D10((__int64)&v160);
              v24 = v139;
LABEL_25:
              v140 = v24;
              sub_C46250((__int64)&v160);
              v42 = v161;
              v161 = 0;
              LODWORD(v163) = v42;
              v162 = v160;
              sub_C46A40((__int64)&v162, v140);
              v43 = v163;
              LODWORD(v163) = 0;
              LODWORD(v165) = v43;
              v164 = v162;
              *(_QWORD *)&v44 = sub_3400EC0((__int64)v41, (__int64)&v164, v158, v159, (__int64)&v154, a6);
              *(_QWORD *)&v46 = sub_3406EB0(v41, 0xBEu, (__int64)&v154, v158, v159, v45, v157, v44);
              v141 = *(_QWORD **)(a1 + 8);
              v133 = v46;
              *(_QWORD *)&v47 = sub_3400EC0((__int64)v141, (__int64)a3, v158, v159, (__int64)&v154, a6);
              *(_QWORD *)&v49 = sub_3406EB0(v141, 0xC0u, (__int64)&v154, v158, v159, v48, v156, v47);
              *(_QWORD *)a4 = sub_3406EB0(v41, 0xBBu, (__int64)&v154, v158, v159, v50, v49, v133);
              *(_DWORD *)(a4 + 8) = v51;
              sub_969240(&v164);
              sub_969240(&v162);
              sub_969240((__int64 *)&v160);
              v52 = *(_QWORD **)(a1 + 8);
              *(_QWORD *)&v53 = sub_3400EC0((__int64)v52, (__int64)a3, v158, v159, (__int64)&v154, a6);
              *(_QWORD *)a5 = sub_3406EB0(v52, 0xBFu, (__int64)&v154, v158, v159, v54, v157, v53);
              *(_DWORD *)(a5 + 8) = v55;
              goto LABEL_19;
            }
            v23 = v160;
LABEL_60:
            v123 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v23;
            if ( !v22 )
              v123 = 0;
            v160 = v123;
            goto LABEL_25;
          }
LABEL_27:
          v56 = v159;
          v57 = v158;
          *(_QWORD *)a4 = v157;
          *(_DWORD *)(a4 + 8) = DWORD2(v157);
          v58 = *(_QWORD **)(a1 + 8);
          *(_QWORD *)&v59 = sub_3400E40((__int64)v58, (unsigned int)(v20 - 1), v57, v56, (__int64)&v154, a6);
          *(_QWORD *)a5 = sub_3406EB0(v58, 0xBFu, (__int64)&v154, v158, v159, v60, v157, v59);
          *(_DWORD *)(a5 + 8) = v61;
          goto LABEL_19;
        }
      }
    }
    else
    {
      v23 = (unsigned __int64)*a3;
      if ( v18 > (unsigned __int64)*a3 )
      {
        v24 = (unsigned int)v20;
        if ( (unsigned int)v20 < v23 )
        {
          LODWORD(v163) = *((_DWORD *)a3 + 2);
          v25 = *(_QWORD **)(a1 + 8);
          v162 = v23;
LABEL_16:
          v138 = v20;
          sub_C46F20((__int64)&v162, v24);
          v26 = v163;
          LODWORD(v163) = 0;
          LODWORD(v165) = v26;
          v164 = v162;
          *(_QWORD *)&v27 = sub_3400EC0((__int64)v25, (__int64)&v164, v158, v159, (__int64)&v154, a6);
          *(_QWORD *)a4 = sub_3406EB0(v25, 0xBFu, (__int64)&v154, v158, v159, v28, v157, v27);
          *(_DWORD *)(a4 + 8) = v29;
          sub_969240(&v164);
          sub_969240(&v162);
          v30 = *(_QWORD **)(a1 + 8);
          *(_QWORD *)&v31 = sub_3400E40((__int64)v30, (unsigned int)(v138 - 1), v158, v159, (__int64)&v154, a6);
          *(_QWORD *)a5 = sub_3406EB0(v30, 0xBFu, (__int64)&v154, v158, v159, v32, v157, v31);
          *(_DWORD *)(a5 + 8) = v33;
LABEL_19:
          sub_9C6650(&v154);
          return;
        }
        if ( (unsigned int)v20 != v23 )
        {
          v161 = *((_DWORD *)a3 + 2);
          v41 = *(_QWORD **)(a1 + 8);
          goto LABEL_60;
        }
        goto LABEL_27;
      }
    }
    v35 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v36 = sub_3400E40((__int64)v35, (unsigned int)(v20 - 1), v158, v159, (__int64)&v154, a6);
    v153 = sub_3406EB0(v35, 0xBFu, (__int64)&v154, v158, v159, v37, v157, v36);
    *(_QWORD *)a4 = v153;
    *(_DWORD *)(a4 + 8) = v38;
    *(_QWORD *)a5 = v153;
    *(_DWORD *)(a5 + 8) = *(_DWORD *)(a4 + 8);
    goto LABEL_19;
  }
  if ( v22 > 0x40 )
  {
    v135 = *((_DWORD *)a3 + 2);
    v144 = v20;
    v80 = sub_C444A0((__int64)a3);
    v81 = v135;
    if ( v135 - v80 <= 0x40 )
    {
      v107 = **a3;
      if ( v18 > v107 )
      {
        if ( v107 > v144 )
        {
          v66 = *(_QWORD **)(a1 + 8);
          LODWORD(v163) = v135;
          sub_C43780((__int64)&v162, (const void **)a3);
          v20 = v144;
          goto LABEL_35;
        }
        v151 = *a3;
        v136 = v144;
        v148 = v81;
        if ( v81 - (unsigned int)sub_C444A0((__int64)a3) > 0x40 || v136 != *v151 )
        {
          v161 = v148;
          v108 = *(_QWORD **)(a1 + 8);
          sub_C43780((__int64)&v160, (const void **)a3);
          v22 = v161;
          v20 = v136;
          if ( v161 > 0x40 )
          {
            sub_C43D10((__int64)&v160);
            v20 = v136;
LABEL_57:
            v149 = v20;
            sub_C46250((__int64)&v160);
            v109 = v161;
            v161 = 0;
            LODWORD(v163) = v109;
            v162 = v160;
            sub_C46A40((__int64)&v162, v149);
            v110 = v163;
            LODWORD(v163) = 0;
            LODWORD(v165) = v110;
            v164 = v162;
            *(_QWORD *)&v111 = sub_3400EC0((__int64)v108, (__int64)&v164, v158, v159, (__int64)&v154, a6);
            *(_QWORD *)&v113 = sub_3406EB0(v108, 0xBEu, (__int64)&v154, v158, v159, v112, v157, v111);
            v150 = *(_QWORD **)(a1 + 8);
            v137 = v113;
            *(_QWORD *)&v114 = sub_3400EC0((__int64)v150, (__int64)a3, v158, v159, (__int64)&v154, a6);
            *(_QWORD *)&v116 = sub_3406EB0(v150, 0xC0u, (__int64)&v154, v158, v159, v115, v156, v114);
            *(_QWORD *)a4 = sub_3406EB0(v108, 0xBBu, (__int64)&v154, v158, v159, v117, v116, v137);
            *(_DWORD *)(a4 + 8) = v118;
            sub_969240(&v164);
            sub_969240(&v162);
            sub_969240((__int64 *)&v160);
            v119 = *(_QWORD **)(a1 + 8);
            *(_QWORD *)&v120 = sub_3400EC0((__int64)v119, (__int64)a3, v158, v159, (__int64)&v154, a6);
            *(_QWORD *)a5 = sub_3406EB0(v119, 0xC0u, (__int64)&v154, v158, v159, v121, v157, v120);
            *(_DWORD *)(a5 + 8) = v122;
            goto LABEL_6;
          }
          v65 = v160;
LABEL_71:
          v128 = ~v65;
          v129 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
          v94 = v22 == 0;
          v130 = 0;
          if ( !v94 )
            v130 = v129;
          v160 = v130 & v128;
          goto LABEL_57;
        }
        goto LABEL_66;
      }
    }
LABEL_43:
    v82 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v154, v158, v159, 0, a6, 0);
    *(_QWORD *)a5 = v82;
    *(_DWORD *)(a5 + 8) = v83;
    *(_QWORD *)a4 = v82;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(a5 + 8);
    goto LABEL_6;
  }
  v65 = (unsigned __int64)*a3;
  if ( v18 <= (unsigned __int64)*a3 )
    goto LABEL_43;
  v20 = (unsigned int)v20;
  if ( v65 > (unsigned int)v20 )
  {
    LODWORD(v163) = *((_DWORD *)a3 + 2);
    v66 = *(_QWORD **)(a1 + 8);
    v162 = v65;
LABEL_35:
    sub_C46F20((__int64)&v162, v20);
    v67 = v163;
    LODWORD(v163) = 0;
    LODWORD(v165) = v67;
    v164 = v162;
    *(_QWORD *)&v68 = sub_3400EC0((__int64)v66, (__int64)&v164, v158, v159, (__int64)&v154, a6);
    *(_QWORD *)a4 = sub_3406EB0(v66, 0xC0u, (__int64)&v154, v158, v159, v69, v157, v68);
    *(_DWORD *)(a4 + 8) = v70;
    sub_969240(&v164);
    sub_969240(&v162);
    *(_QWORD *)a5 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v154, v158, v159, 0, a6, 0);
    *(_DWORD *)(a5 + 8) = v71;
    goto LABEL_6;
  }
  if ( v65 != (unsigned int)v20 )
  {
    v161 = *((_DWORD *)a3 + 2);
    v108 = *(_QWORD **)(a1 + 8);
    goto LABEL_71;
  }
LABEL_66:
  v125 = v158;
  v126 = v159;
  *(_QWORD *)a4 = v157;
  *(_DWORD *)(a4 + 8) = DWORD2(v157);
  *(_QWORD *)a5 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v154, v125, v126, 0, a6, 0);
  *(_DWORD *)(a5 + 8) = v127;
LABEL_6:
  if ( v154 )
    sub_B91220((__int64)&v154, v154);
}
