// Function: sub_27EEBD0
// Address: 0x27eebd0
//
__int64 __fastcall sub_27EEBD0(unsigned __int8 *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v5; // r12
  int v6; // eax
  unsigned int v7; // r14d
  __int64 *v9; // rax
  _BYTE *v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rcx
  _BYTE *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // r8
  _QWORD *v18; // r11
  _QWORD *v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // r9
  unsigned __int8 *v22; // rcx
  __int64 v23; // rax
  int v24; // eax
  int v25; // esi
  bool v26; // al
  bool v27; // al
  char v28; // al
  __int64 v29; // rdx
  __int64 v30; // rcx
  char v31; // al
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  unsigned int v36; // eax
  __int64 v37; // rsi
  int v38; // ecx
  __int64 *v39; // rdx
  char v40; // al
  __int64 v41; // rdx
  bool v42; // al
  bool v43; // al
  int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  _QWORD *v52; // rdx
  unsigned __int64 v53; // rax
  int v54; // edx
  __int64 v55; // r13
  __int64 v56; // rax
  _QWORD *v57; // r11
  __int64 v58; // rbx
  __int64 v59; // r14
  __int64 v60; // rdx
  unsigned __int8 *v61; // r13
  unsigned __int8 *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  unsigned __int8 **v66; // r14
  __int64 v67; // rcx
  unsigned __int8 *v68; // r12
  int v69; // edx
  char v70; // r9
  int v71; // eax
  unsigned __int8 *v72; // r15
  int v73; // edx
  int v74; // r10d
  __int64 v75; // r8
  __int64 (__fastcall *v76)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64); // rax
  __int64 v77; // rax
  __int64 v78; // r13
  int v79; // eax
  unsigned __int8 *v80; // r15
  __int64 (__fastcall *v81)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v82; // rax
  int v83; // r10d
  __int64 v84; // r15
  _BYTE *v85; // r15
  unsigned __int64 v86; // rbx
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // r15
  _BYTE *v90; // rcx
  __int64 v91; // r15
  unsigned __int64 v92; // r12
  _BYTE *v93; // rbx
  __int64 v94; // rdx
  unsigned int v95; // esi
  __int64 v96; // rcx
  __int64 v97; // r8
  __int64 v98; // r9
  __int64 v99; // rax
  unsigned __int8 **v100; // rbx
  unsigned __int8 **v101; // r12
  _QWORD *v102; // r14
  unsigned __int8 *v103; // rdi
  _QWORD *v104; // [rsp+8h] [rbp-288h]
  __int64 v105; // [rsp+30h] [rbp-260h]
  char v106; // [rsp+30h] [rbp-260h]
  int v107; // [rsp+30h] [rbp-260h]
  __int64 v108; // [rsp+30h] [rbp-260h]
  unsigned __int8 *v109; // [rsp+30h] [rbp-260h]
  char v110; // [rsp+30h] [rbp-260h]
  unsigned __int8 v111; // [rsp+48h] [rbp-248h]
  _BYTE *v112; // [rsp+48h] [rbp-248h]
  char v113; // [rsp+50h] [rbp-240h]
  _QWORD *v114; // [rsp+50h] [rbp-240h]
  __int64 v115; // [rsp+58h] [rbp-238h]
  _QWORD *v116; // [rsp+58h] [rbp-238h]
  __int64 v117; // [rsp+58h] [rbp-238h]
  _QWORD *v118; // [rsp+58h] [rbp-238h]
  unsigned __int8 *v119; // [rsp+58h] [rbp-238h]
  _QWORD *v120; // [rsp+60h] [rbp-230h]
  unsigned __int8 v121; // [rsp+60h] [rbp-230h]
  _QWORD *v122; // [rsp+60h] [rbp-230h]
  unsigned __int8 v123; // [rsp+60h] [rbp-230h]
  _QWORD *v124; // [rsp+68h] [rbp-228h]
  unsigned __int8 *v125; // [rsp+68h] [rbp-228h]
  __int64 *v126; // [rsp+68h] [rbp-228h]
  _QWORD *v127; // [rsp+68h] [rbp-228h]
  __int64 v128; // [rsp+68h] [rbp-228h]
  unsigned __int8 *v129; // [rsp+68h] [rbp-228h]
  _QWORD *v131; // [rsp+70h] [rbp-220h]
  unsigned __int64 v132; // [rsp+70h] [rbp-220h]
  _QWORD *v133; // [rsp+70h] [rbp-220h]
  __int64 v134; // [rsp+70h] [rbp-220h]
  unsigned __int8 v135; // [rsp+7Ah] [rbp-216h]
  unsigned __int8 v136; // [rsp+7Ch] [rbp-214h]
  unsigned __int8 v137; // [rsp+7Fh] [rbp-211h]
  __int64 v139; // [rsp+98h] [rbp-1F8h]
  __int64 v140; // [rsp+A0h] [rbp-1F0h]
  _QWORD v141[4]; // [rsp+B0h] [rbp-1E0h] BYREF
  char v142; // [rsp+D0h] [rbp-1C0h]
  char v143; // [rsp+D1h] [rbp-1BFh]
  _QWORD v144[4]; // [rsp+E0h] [rbp-1B0h] BYREF
  __int16 v145; // [rsp+100h] [rbp-190h]
  _BYTE *v146; // [rsp+110h] [rbp-180h] BYREF
  __int64 v147; // [rsp+118h] [rbp-178h]
  _BYTE v148[48]; // [rsp+120h] [rbp-170h] BYREF
  unsigned __int8 **v149; // [rsp+150h] [rbp-140h] BYREF
  __int64 v150; // [rsp+158h] [rbp-138h]
  _BYTE v151[48]; // [rsp+160h] [rbp-130h] BYREF
  _QWORD *v152; // [rsp+190h] [rbp-100h] BYREF
  __int64 v153; // [rsp+198h] [rbp-F8h]
  _QWORD v154[6]; // [rsp+1A0h] [rbp-F0h] BYREF
  _BYTE *v155; // [rsp+1D0h] [rbp-C0h] BYREF
  __int64 v156; // [rsp+1D8h] [rbp-B8h]
  _BYTE v157[32]; // [rsp+1E0h] [rbp-B0h] BYREF
  __int64 v158; // [rsp+200h] [rbp-90h]
  __int64 v159; // [rsp+208h] [rbp-88h]
  __int64 v160; // [rsp+210h] [rbp-80h]
  __int64 v161; // [rsp+218h] [rbp-78h]
  void **v162; // [rsp+220h] [rbp-70h]
  void **v163; // [rsp+228h] [rbp-68h]
  __int64 v164; // [rsp+230h] [rbp-60h]
  unsigned int v165; // [rsp+238h] [rbp-58h]
  __int16 v166; // [rsp+23Ch] [rbp-54h]
  char v167; // [rsp+23Eh] [rbp-52h]
  __int64 v168; // [rsp+240h] [rbp-50h]
  __int64 v169; // [rsp+248h] [rbp-48h]
  void *v170; // [rsp+250h] [rbp-40h] BYREF
  void *v171; // [rsp+258h] [rbp-38h] BYREF

  v5 = a1;
  v6 = *a1;
  v139 = a3;
  if ( v6 != 46 && (v6 != 47 || !sub_B451B0((__int64)a1) || !sub_B451E0((__int64)a1)) )
    return 0;
  if ( (a1[7] & 0x40) != 0 )
  {
    v9 = (__int64 *)*((_QWORD *)a1 - 1);
  }
  else
  {
    a3 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
    v9 = (__int64 *)&a1[-a3];
  }
  v10 = (_BYTE *)*v9;
  v11 = v9[4];
  if ( (unsigned __int8)sub_D48480(a2, *v9, a3, (__int64)a4) )
  {
    v14 = v10;
    v10 = (_BYTE *)v11;
    v11 = (__int64)v14;
  }
  v7 = sub_D48480(a2, (__int64)v10, v12, v13);
  if ( (_BYTE)v7 )
    return 0;
  v137 = sub_D48480(a2, v11, v15, v16);
  if ( !v137 )
    return 0;
  v146 = v148;
  v18 = v154;
  v147 = 0x600000000LL;
  v149 = (unsigned __int8 **)v151;
  v150 = 0x600000000LL;
  v152 = v154;
  v153 = 0x600000000LL;
  if ( (unsigned __int8)(*v10 - 42) > 0x11u )
    goto LABEL_19;
  v19 = v154;
  LODWORD(v153) = 1;
  v20 = 1;
  v21 = v7;
  v154[0] = v10;
  while ( 1 )
  {
    v22 = (unsigned __int8 *)v19[v20 - 1];
    LODWORD(v153) = v20 - 1;
    v23 = *((_QWORD *)v22 + 2);
    if ( !v23 || *(_QWORD *)(v23 + 8) )
    {
LABEL_16:
      v7 = v21;
      goto LABEL_17;
    }
    v24 = *v22;
    v25 = v24 - 29;
    if ( v24 != 42 )
    {
      if ( v24 != 43 )
        goto LABEL_27;
      v122 = v18;
      v127 = v19;
      v111 = v21;
      v117 = (__int64)v22;
      v42 = sub_B451B0((__int64)v22);
      v19 = v127;
      v18 = v122;
      if ( !v42 )
        goto LABEL_31;
      v43 = sub_B451E0(v117);
      v22 = (unsigned __int8 *)v117;
      v19 = v127;
      v18 = v122;
      v25 = 14;
      v21 = v111;
      if ( !v43 )
        goto LABEL_31;
    }
    v44 = **((unsigned __int8 **)v22 - 8);
    if ( (unsigned __int8)v44 <= 0x1Cu )
      goto LABEL_16;
    if ( (unsigned int)(v44 - 42) <= 0x11 && (unsigned __int8)(**((_BYTE **)v22 - 4) - 42) <= 0x11u )
    {
      v118 = v18;
      v123 = v21;
      v128 = (__int64)v22;
      sub_27EEB80((__int64)&v152, *((_QWORD *)v22 - 8), (__int64)v19, (__int64)v22, (__int64)v17, v21);
      sub_27EEB80((__int64)&v152, *(_QWORD *)(v128 - 32), v45, v128, v46, v47);
      sub_27EEB80((__int64)&v149, v128, v48, v128, v49, v50);
      v21 = v123;
      v18 = v118;
      goto LABEL_44;
    }
LABEL_27:
    if ( v25 != 17 )
    {
      if ( v25 != 18 )
        goto LABEL_16;
      v120 = v18;
      v124 = v19;
      v113 = v21;
      v115 = (__int64)v22;
      v26 = sub_B451B0((__int64)v22);
      v19 = v124;
      v18 = v120;
      if ( !v26 )
        goto LABEL_31;
      v27 = sub_B451E0(v115);
      v22 = (unsigned __int8 *)v115;
      v19 = v124;
      v18 = v120;
      LOBYTE(v21) = v113;
      if ( !v27 )
        goto LABEL_31;
    }
    v116 = v18;
    v121 = v21;
    v125 = v22;
    v28 = sub_D48480(a2, (__int64)v22, (__int64)v19, (__int64)v22);
    v30 = (__int64)v125;
    v18 = v116;
    if ( v28 )
    {
      v19 = v152;
      v7 = v121;
      goto LABEL_17;
    }
    if ( (v125[7] & 0x40) != 0 )
    {
      v126 = (__int64 *)*((_QWORD *)v125 - 1);
    }
    else
    {
      v30 = (__int64)&v125[-32 * (*((_DWORD *)v125 + 1) & 0x7FFFFFF)];
      v126 = (__int64 *)v30;
    }
    v31 = sub_D48480(a2, *v126, v29, v30);
    v21 = v121;
    v18 = v116;
    v17 = &qword_4FFE2A0;
    if ( v31 )
    {
      v34 = (unsigned int)v147;
      v35 = (unsigned int)v147 + 1LL;
      if ( v35 > HIDWORD(v147) )
      {
        sub_C8D5F0((__int64)&v146, v148, v35, 8u, (__int64)&qword_4FFE2A0, v121);
        v34 = (unsigned int)v147;
        v18 = v116;
        v17 = &qword_4FFE2A0;
        v21 = v121;
      }
      *(_QWORD *)&v146[8 * v34] = v126;
      v36 = v147 + 1;
      LODWORD(v147) = v147 + 1;
    }
    else
    {
      v40 = sub_D48480(a2, v126[4], v32, v33);
      v21 = v121;
      v18 = v116;
      v17 = &qword_4FFE2A0;
      if ( !v40 )
        break;
      v41 = (unsigned int)v147;
      if ( (unsigned __int64)(unsigned int)v147 + 1 > HIDWORD(v147) )
      {
        sub_C8D5F0((__int64)&v146, v148, (unsigned int)v147 + 1LL, 8u, (__int64)&qword_4FFE2A0, v121);
        v41 = (unsigned int)v147;
        v18 = v116;
        v17 = &qword_4FFE2A0;
        v21 = v121;
      }
      *(_QWORD *)&v146[8 * v41] = v126 + 4;
      v36 = v147 + 1;
      LODWORD(v147) = v147 + 1;
    }
    v37 = *((_QWORD *)a1 + 1);
    v38 = *(unsigned __int8 *)(v37 + 8);
    if ( (unsigned int)(v38 - 17) <= 1 )
      LOBYTE(v38) = *(_BYTE *)(**(_QWORD **)(v37 + 16) + 8LL);
    v39 = &qword_4FFE2A0;
    if ( (_BYTE)v38 == 12 )
      v39 = &qword_4FFDF20;
    if ( *((_DWORD *)v39 + 34) < v36 )
      break;
LABEL_44:
    v20 = v153;
    if ( !(_DWORD)v153 )
    {
      v7 = v21;
      if ( !(_DWORD)v147 )
      {
        v19 = v152;
        goto LABEL_17;
      }
      v51 = *((_QWORD *)a1 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v51 + 8) - 17 <= 1 )
        v51 = **(_QWORD **)(v51 + 16);
      if ( *(_BYTE *)(v51 + 8) == 12 && v149 != &v149[(unsigned int)v150] )
      {
        v134 = v11;
        v100 = &v149[(unsigned int)v150];
        v101 = v149;
        v102 = v18;
        do
        {
          v103 = *v101++;
          sub_B44F30(v103);
        }
        while ( v100 != v101 );
        v11 = v134;
        v5 = a1;
        v18 = v102;
      }
      v131 = v18;
      v52 = (_QWORD *)(sub_D4B130(a2) + 48);
      v53 = *v52 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v53 == v52 )
      {
        v55 = 0;
      }
      else
      {
        if ( !v53 )
          BUG();
        v54 = *(unsigned __int8 *)(v53 - 24);
        v55 = 0;
        v56 = v53 - 24;
        if ( (unsigned int)(v54 - 30) < 0xB )
          v55 = v56;
      }
      v161 = sub_BD5C60(v55);
      v162 = &v170;
      v163 = &v171;
      v155 = v157;
      v170 = &unk_49DA100;
      v156 = 0x200000000LL;
      v171 = &unk_49DA0B0;
      v164 = 0;
      v165 = 0;
      v166 = 512;
      v167 = 7;
      v168 = 0;
      v169 = 0;
      v158 = 0;
      v159 = 0;
      LOWORD(v160) = 0;
      sub_D5F1F0((__int64)&v155, v55);
      v57 = v131;
      v112 = &v146[8 * (unsigned int)v147];
      if ( v146 == v112 )
      {
LABEL_111:
        v133 = v57;
        sub_BD84D0((__int64)v5, (__int64)v10);
        sub_27EC480(v5, v139, a4, v96, v97, v98);
        nullsub_61();
        v170 = &unk_49DA100;
        nullsub_63();
        v18 = v133;
        if ( v155 != v157 )
        {
          _libc_free((unsigned __int64)v155);
          v18 = v133;
        }
        v19 = v152;
        v7 = v137;
        goto LABEL_17;
      }
      v132 = (unsigned __int64)v146;
      v119 = (unsigned __int8 *)v11;
      v58 = v105;
      v129 = v10;
      v114 = v5;
      v104 = v57;
      while ( 1 )
      {
        v66 = *(unsigned __int8 ***)v132;
        v67 = v114[1];
        v68 = *(unsigned __int8 **)(*(_QWORD *)v132 + 24LL);
        v69 = *(unsigned __int8 *)(v67 + 8);
        if ( (unsigned int)(v69 - 17) <= 1 )
          LOBYTE(v69) = *(_BYTE *)(**(_QWORD **)(v67 + 16) + 8LL);
        v143 = 1;
        if ( (_BYTE)v69 == 12 )
        {
          v142 = 3;
          v141[0] = "factor.op.mul";
          v80 = *v66;
          v81 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v162 + 4);
          if ( v81 == sub_9201A0 )
          {
            if ( *v80 > 0x15u || *v119 > 0x15u )
              goto LABEL_106;
            v78 = (unsigned __int8)sub_AC47B0(17)
                ? sub_AD5570(17, (__int64)v80, v119, 0, 0)
                : sub_AABE40(0x11u, v80, v119);
LABEL_93:
            if ( !v78 )
            {
LABEL_106:
              v145 = 257;
              v78 = sub_B504D0(17, (__int64)v80, (__int64)v119, (__int64)v144, 0, 0);
              (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v163 + 2))(
                v163,
                v78,
                v141,
                v159,
                v160);
              v89 = 16LL * (unsigned int)v156;
              v90 = &v155[v89];
              if ( v155 != &v155[v89] )
              {
                v109 = v68;
                v91 = v58;
                v92 = (unsigned __int64)v155;
                v93 = v90;
                do
                {
                  v94 = *(_QWORD *)(v92 + 8);
                  v95 = *(_DWORD *)v92;
                  v92 += 16LL;
                  sub_B99FD0(v78, v95, v94);
                }
                while ( v93 != (_BYTE *)v92 );
                v68 = v109;
                v58 = v91;
              }
            }
            sub_B44F30(v68);
            goto LABEL_86;
          }
          v78 = v81((__int64)v162, 17u, *v66, v119, 0, 0);
          goto LABEL_93;
        }
        v142 = 3;
        v70 = 0;
        v141[0] = "factor.op.fmul";
        if ( v68 )
        {
          v71 = sub_B45210((__int64)v68);
          v70 = 1;
          LODWORD(v140) = v71;
        }
        v72 = *v66;
        BYTE4(v140) = v70;
        if ( (_BYTE)v166 )
        {
          v78 = sub_B35400((__int64)&v155, 0x6Cu, (__int64)v72, (__int64)v119, v140, (__int64)v141, 0, v135, v136);
          goto LABEL_86;
        }
        v73 = v165;
        v74 = v140;
        v75 = v165;
        if ( v70 )
          v75 = (unsigned int)v140;
        v76 = (__int64 (__fastcall *)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64))*((_QWORD *)*v162 + 5);
        if ( (char *)v76 != (char *)sub_928A40 )
          break;
        if ( *v72 <= 0x15u && *v119 <= 0x15u )
        {
          v106 = v70;
          if ( (unsigned __int8)sub_AC47B0(18) )
            v77 = sub_AD5570(18, (__int64)v72, v119, 0, 0);
          else
            v77 = sub_AABE40(0x12u, v72, v119);
          v70 = v106;
          v74 = v140;
          v78 = v77;
          goto LABEL_85;
        }
LABEL_97:
        if ( !v70 )
          v74 = v73;
        v145 = 257;
        v107 = v74;
        v82 = sub_B504D0(18, (__int64)v72, (__int64)v119, (__int64)v144, 0, 0);
        v83 = v107;
        v78 = v82;
        if ( v164 )
        {
          sub_B99FD0(v82, 3u, v164);
          v83 = v107;
        }
        sub_B45150(v78, v83);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v163 + 2))(v163, v78, v141, v159, v160);
        v84 = 16LL * (unsigned int)v156;
        if ( v155 != &v155[v84] )
        {
          v108 = v58;
          v85 = &v155[v84];
          v86 = (unsigned __int64)v155;
          do
          {
            v87 = *(_QWORD *)(v86 + 8);
            v88 = *(_DWORD *)v86;
            v86 += 16LL;
            sub_B99FD0(v78, v88, v87);
          }
          while ( v85 != (_BYTE *)v86 );
          v58 = v108;
        }
LABEL_86:
        v79 = sub_BD2910((__int64)v66);
        if ( !v79 )
          goto LABEL_87;
        v59 = *((_QWORD *)v68 - 8);
        if ( v79 != 1 )
        {
          v78 = *((_QWORD *)v68 - 8);
LABEL_87:
          v59 = v78;
          v78 = *((_QWORD *)v68 - 4);
        }
        LOWORD(v58) = 0;
        v144[0] = sub_BD5D20((__int64)v68);
        v145 = 773;
        v144[1] = v60;
        v144[2] = ".reass";
        v61 = (unsigned __int8 *)sub_B504D0((unsigned int)*v68 - 29, v59, v78, (__int64)v144, (__int64)(v68 + 24), v58);
        sub_B45260(v61, (__int64)v68, 1);
        v62 = v129;
        if ( v129 == v68 )
          v62 = v61;
        v129 = v62;
        sub_BD84D0((__int64)v68, (__int64)v61);
        sub_27EC480(v68, v139, a4, v63, v64, v65);
        v132 += 8LL;
        if ( v112 == (_BYTE *)v132 )
        {
          v10 = v129;
          v5 = v114;
          v57 = v104;
          goto LABEL_111;
        }
      }
      v110 = v70;
      v99 = v76(v162, 18, v72, v119, v75);
      v74 = v140;
      v70 = v110;
      v78 = v99;
LABEL_85:
      if ( v78 )
        goto LABEL_86;
      v73 = v165;
      goto LABEL_97;
    }
    v19 = v152;
  }
  v19 = v152;
LABEL_31:
  v7 = 0;
LABEL_17:
  if ( v19 != v18 )
    _libc_free((unsigned __int64)v19);
LABEL_19:
  if ( v149 != (unsigned __int8 **)v151 )
    _libc_free((unsigned __int64)v149);
  if ( v146 != v148 )
    _libc_free((unsigned __int64)v146);
  return v7;
}
