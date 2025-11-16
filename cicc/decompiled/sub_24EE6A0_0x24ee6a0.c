// Function: sub_24EE6A0
// Address: 0x24ee6a0
//
void __fastcall sub_24EE6A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  unsigned int v9; // r8d
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // rax
  char v13; // al
  char v14; // bl
  __int64 v15; // r9
  _QWORD *v16; // r14
  unsigned int *v17; // r12
  unsigned int *v18; // rbx
  __int64 v19; // rdx
  unsigned int v20; // esi
  int v21; // r12d
  unsigned int *v22; // r12
  unsigned int *v23; // rbx
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  _QWORD *v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // r12
  const char *v34; // rax
  __int64 v35; // rdx
  char v36; // cl
  __int64 *v37; // rsi
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // r15
  unsigned __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // r14
  _QWORD *v48; // r11
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // eax
  int v53; // eax
  unsigned int v54; // ecx
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rcx
  int v58; // eax
  int v59; // eax
  unsigned int v60; // edx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // rbx
  __int64 v65; // r14
  _QWORD *v66; // r12
  __int64 v67; // rcx
  __int64 v68; // r8
  unsigned int v69; // r8d
  __int64 v70; // rdx
  __int64 v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // r12
  unsigned int *v74; // r14
  unsigned int *v75; // rbx
  __int64 v76; // rdx
  unsigned int v77; // esi
  __int64 v78; // r12
  __int64 v79; // rbx
  __int64 v80; // r14
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rax
  __int64 v91; // r15
  __int64 v92; // rdx
  __int64 v93; // rsi
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rdx
  __int64 v98; // r15
  __int64 v99; // rdx
  __int64 v100; // rsi
  __int64 v101; // rax
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // rax
  __int64 v105; // rax
  unsigned __int64 v106; // rcx
  __int64 v107; // rax
  __int64 *v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // [rsp-8h] [rbp-2B8h]
  __int64 v111; // [rsp+10h] [rbp-2A0h]
  __int64 v113; // [rsp+40h] [rbp-270h]
  __int64 v115; // [rsp+58h] [rbp-258h]
  unsigned __int8 v116; // [rsp+66h] [rbp-24Ah]
  __int64 *v117; // [rsp+70h] [rbp-240h]
  __int64 v118; // [rsp+78h] [rbp-238h]
  __int64 v119; // [rsp+80h] [rbp-230h]
  __int64 *v120; // [rsp+88h] [rbp-228h]
  __int64 v121; // [rsp+90h] [rbp-220h]
  __int64 v122; // [rsp+98h] [rbp-218h]
  _QWORD *v123; // [rsp+98h] [rbp-218h]
  __int64 v124; // [rsp+98h] [rbp-218h]
  __int64 v125; // [rsp+A8h] [rbp-208h]
  __int64 v126; // [rsp+B0h] [rbp-200h]
  __int64 v127; // [rsp+B8h] [rbp-1F8h]
  __int64 v128; // [rsp+C0h] [rbp-1F0h]
  __int64 *v129; // [rsp+D0h] [rbp-1E0h]
  __int64 v130; // [rsp+E8h] [rbp-1C8h] BYREF
  _QWORD v131[4]; // [rsp+F0h] [rbp-1C0h] BYREF
  char v132; // [rsp+110h] [rbp-1A0h]
  char v133; // [rsp+111h] [rbp-19Fh]
  char *v134; // [rsp+120h] [rbp-190h] BYREF
  __int64 v135; // [rsp+128h] [rbp-188h]
  __int64 *v136; // [rsp+130h] [rbp-180h]
  __int64 v137; // [rsp+138h] [rbp-178h]
  __int16 v138; // [rsp+140h] [rbp-170h]
  char *v139; // [rsp+150h] [rbp-160h] BYREF
  __int64 v140; // [rsp+158h] [rbp-158h]
  _BYTE v141[16]; // [rsp+160h] [rbp-150h] BYREF
  char v142; // [rsp+170h] [rbp-140h]
  char v143; // [rsp+171h] [rbp-13Fh]
  __int64 v144; // [rsp+180h] [rbp-130h]
  __int64 v145; // [rsp+188h] [rbp-128h]
  __int64 v146; // [rsp+190h] [rbp-120h]
  _QWORD *v147; // [rsp+198h] [rbp-118h]
  void **v148; // [rsp+1A0h] [rbp-110h]
  void **v149; // [rsp+1A8h] [rbp-108h]
  __int64 v150; // [rsp+1B0h] [rbp-100h]
  int v151; // [rsp+1B8h] [rbp-F8h]
  __int16 v152; // [rsp+1BCh] [rbp-F4h]
  char v153; // [rsp+1BEh] [rbp-F2h]
  __int64 v154; // [rsp+1C0h] [rbp-F0h]
  __int64 v155; // [rsp+1C8h] [rbp-E8h]
  void *v156; // [rsp+1D0h] [rbp-E0h] BYREF
  void *v157; // [rsp+1D8h] [rbp-D8h] BYREF
  char v158[8]; // [rsp+1E0h] [rbp-D0h] BYREF
  unsigned __int64 v159; // [rsp+1E8h] [rbp-C8h]
  char v160; // [rsp+1FCh] [rbp-B4h]

  v4 = a2;
  sub_24E3E60((__int64)v158, a1);
  v143 = 1;
  v120 = (__int64 *)sub_B2BE50(a1);
  v139 = "resume.entry";
  v142 = 3;
  v113 = sub_22077B0(0x50u);
  if ( v113 )
    sub_AA4D50(v113, (__int64)v120, (__int64)&v139, a1, 0);
  v143 = 1;
  v139 = "unreachable";
  v142 = 3;
  v5 = sub_22077B0(0x50u);
  v111 = v5;
  if ( v5 )
    sub_AA4D50(v5, (__int64)v120, (__int64)&v139, a1, 0);
  v6 = (_QWORD *)sub_AA48A0(v113);
  v7 = *(_QWORD *)(a2 + 288);
  v147 = v6;
  v8 = *(_QWORD *)(v4 + 312);
  v9 = *(_DWORD *)(v4 + 352);
  v148 = &v156;
  v149 = &v157;
  v139 = v141;
  v156 = &unk_49DA100;
  v140 = 0x200000000LL;
  LOWORD(v146) = 0;
  v157 = &unk_49DA0B0;
  v145 = v113 + 48;
  v134 = "index.addr";
  v152 = 512;
  v115 = v7;
  v144 = v113;
  v118 = v8;
  v150 = 0;
  v151 = 0;
  v153 = 7;
  v154 = 0;
  v155 = 0;
  v138 = 259;
  v10 = sub_9213A0((unsigned int **)&v139, v7, v8, 0, v9, (__int64)&v134, 7u);
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 288) + 16LL) + 8LL * *(unsigned int *)(v4 + 352));
  v133 = 1;
  v131[0] = "index";
  v132 = 3;
  v12 = sub_AA4E30(v144);
  v13 = sub_AE5020(v12, v11);
  v138 = 257;
  v14 = v13;
  v16 = sub_BD2C40(80, unk_3F10A14);
  if ( v16 )
  {
    sub_B4D190((__int64)v16, v11, v10, (__int64)&v134, 0, v14, 0, 0);
    v15 = v110;
  }
  (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, __int64, __int64, __int64))*v149 + 2))(
    v149,
    v16,
    v131,
    v145,
    v146,
    v15);
  v17 = (unsigned int *)v139;
  v18 = (unsigned int *)&v139[16 * (unsigned int)v140];
  if ( v139 != (char *)v18 )
  {
    do
    {
      v19 = *((_QWORD *)v17 + 1);
      v20 = *v17;
      v17 += 4;
      sub_B99FD0((__int64)v16, v20, v19);
    }
    while ( v18 != v17 );
  }
  v21 = *(_DWORD *)(v4 + 128);
  v138 = 257;
  v121 = sub_BD2DA0(80);
  if ( v121 )
    sub_B53A60(v121, (__int64)v16, v111, v21, 0, 0);
  (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v149 + 2))(v149, v121, &v134, v145, v146);
  v22 = (unsigned int *)v139;
  v23 = (unsigned int *)&v139[16 * (unsigned int)v140];
  if ( v139 != (char *)v23 )
  {
    do
    {
      v24 = *((_QWORD *)v22 + 1);
      v25 = *v22;
      v22 += 4;
      sub_B99FD0(v121, v25, v24);
    }
    while ( v23 != v22 );
  }
  v26 = *(unsigned int *)(v4 + 128);
  v130 = 0;
  *(_QWORD *)(v4 + 328) = v121;
  v27 = *(__int64 **)(v4 + 120);
  v117 = &v27[v26];
  if ( v27 != v117 )
  {
    v129 = *(__int64 **)(v4 + 120);
    v28 = 0;
    v128 = v4;
    do
    {
      v64 = *v129;
      v65 = sub_ACD640(
              *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v128 + 288) + 16LL) + 8LL * *(unsigned int *)(v128 + 352)),
              v28,
              0);
      v66 = *(_QWORD **)(v64 - 32LL * (*(_DWORD *)(v64 + 4) & 0x7FFFFFF));
      if ( *(_BYTE *)v66 == 85
        && (v71 = *(v66 - 4)) != 0
        && !*(_BYTE *)v71
        && *(_QWORD *)(v71 + 24) == v66[10]
        && (*(_BYTE *)(v71 + 33) & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v71 + 36) != 57 )
          v66 = 0;
      }
      else
      {
        v66 = 0;
      }
      sub_D5F1F0((__int64)&v139, (__int64)v66);
      if ( sub_AD7A80(
             *(_BYTE **)(v64 + 32 * (1LL - (*(_DWORD *)(v64 + 4) & 0x7FFFFFF))),
             (__int64)v66,
             *(_DWORD *)(v64 + 4) & 0x7FFFFFF,
             v67,
             v68) )
      {
        sub_24E58A0((__int64)&v139, v128, v118);
      }
      else
      {
        v134 = "index.addr";
        v69 = *(_DWORD *)(v128 + 352);
        v138 = 259;
        v70 = sub_9213A0((unsigned int **)&v139, v115, v118, 0, v69, (__int64)&v134, 7u);
        sub_2463EC0((__int64 *)&v139, v65, v70, v116, 0);
      }
      v29 = sub_AC3540(v120);
      sub_BD84D0((__int64)v66, v29);
      sub_B43D60(v66);
      v30 = *(_QWORD **)(v64 + 40);
      v134 = "resume.";
      v136 = &v130;
      v138 = 2819;
      v31 = v125;
      LOWORD(v31) = 0;
      v125 = v31;
      v32 = sub_AA8550(v30, (__int64 *)(v64 + 24), v31, (__int64)&v134, 0);
      v133 = 1;
      v33 = (_QWORD *)v32;
      v132 = 3;
      v131[0] = ".landing";
      v34 = sub_BD5D20(v32);
      v36 = v132;
      if ( v132 )
      {
        if ( v132 == 1 )
        {
          v134 = (char *)v34;
          v135 = v35;
          v138 = 261;
        }
        else
        {
          if ( v133 == 1 )
          {
            v37 = (__int64 *)v131[0];
            v119 = v131[1];
          }
          else
          {
            v37 = v131;
            v36 = 2;
          }
          v134 = (char *)v34;
          v135 = v35;
          v136 = v37;
          v137 = v119;
          LOBYTE(v138) = 5;
          HIBYTE(v138) = v36;
        }
      }
      else
      {
        v138 = 256;
      }
      v38 = *(_QWORD *)(v64 + 32);
      if ( v38 == *(_QWORD *)(v64 + 40) + 48LL || !v38 )
        v39 = 0;
      else
        v39 = v38 - 24;
      v40 = v126;
      LOWORD(v40) = 0;
      v126 = v40;
      v41 = sub_AA8550(v33, (__int64 *)(v39 + 24), v40, (__int64)&v134, 0);
      sub_B53E30(v121, v65, (__int64)v33);
      v42 = v30[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v42 == v30 + 6 )
        goto LABEL_104;
      if ( !v42 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v42 - 24) - 30 > 0xA )
LABEL_104:
        BUG();
      if ( *(_QWORD *)(v42 - 56) )
      {
        v43 = *(_QWORD *)(v42 - 48);
        **(_QWORD **)(v42 - 40) = v43;
        if ( v43 )
          *(_QWORD *)(v43 + 16) = *(_QWORD *)(v42 - 40);
      }
      *(_QWORD *)(v42 - 56) = v41;
      if ( v41 )
      {
        v44 = *(_QWORD *)(v41 + 16);
        *(_QWORD *)(v42 - 48) = v44;
        if ( v44 )
          *(_QWORD *)(v44 + 16) = v42 - 48;
        *(_QWORD *)(v42 - 40) = v41 + 16;
        *(_QWORD *)(v41 + 16) = v42 - 56;
      }
      v138 = 257;
      v122 = sub_BCB2B0(v147);
      v45 = sub_BD2DA0(80);
      v46 = v122;
      v47 = v45;
      if ( v45 )
      {
        v123 = (_QWORD *)v45;
        sub_B44260(v45, v46, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v47 + 72) = 2;
        sub_BD6B50((unsigned __int8 *)v47, (const char **)&v134);
        sub_BD2A10(v47, *(_DWORD *)(v47 + 72), 1);
        v48 = v123;
      }
      else
      {
        v48 = 0;
      }
      v49 = v127;
      LOWORD(v49) = 1;
      v127 = v49;
      sub_B44220(v48, *(_QWORD *)(v41 + 56), v49);
      sub_BD84D0(v64, v47);
      v50 = sub_BCB2B0(v147);
      v51 = sub_ACD640(v50, 255, 0);
      v52 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
      if ( v52 == *(_DWORD *)(v47 + 72) )
      {
        v124 = v51;
        sub_B48D90(v47);
        v51 = v124;
        v52 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
      }
      v53 = (v52 + 1) & 0x7FFFFFF;
      v54 = v53 | *(_DWORD *)(v47 + 4) & 0xF8000000;
      v55 = *(_QWORD *)(v47 - 8) + 32LL * (unsigned int)(v53 - 1);
      *(_DWORD *)(v47 + 4) = v54;
      if ( *(_QWORD *)v55 )
      {
        v56 = *(_QWORD *)(v55 + 8);
        **(_QWORD **)(v55 + 16) = v56;
        if ( v56 )
          *(_QWORD *)(v56 + 16) = *(_QWORD *)(v55 + 16);
      }
      *(_QWORD *)v55 = v51;
      if ( v51 )
      {
        v57 = *(_QWORD *)(v51 + 16);
        *(_QWORD *)(v55 + 8) = v57;
        if ( v57 )
          *(_QWORD *)(v57 + 16) = v55 + 8;
        *(_QWORD *)(v55 + 16) = v51 + 16;
        *(_QWORD *)(v51 + 16) = v55;
      }
      *(_QWORD *)(*(_QWORD *)(v47 - 8)
                + 32LL * *(unsigned int *)(v47 + 72)
                + 8LL * ((*(_DWORD *)(v47 + 4) & 0x7FFFFFFu) - 1)) = v30;
      v58 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
      if ( v58 == *(_DWORD *)(v47 + 72) )
      {
        sub_B48D90(v47);
        v58 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
      }
      v59 = (v58 + 1) & 0x7FFFFFF;
      v60 = v59 | *(_DWORD *)(v47 + 4) & 0xF8000000;
      v61 = *(_QWORD *)(v47 - 8) + 32LL * (unsigned int)(v59 - 1);
      *(_DWORD *)(v47 + 4) = v60;
      if ( *(_QWORD *)v61 )
      {
        v62 = *(_QWORD *)(v61 + 8);
        **(_QWORD **)(v61 + 16) = v62;
        if ( v62 )
          *(_QWORD *)(v62 + 16) = *(_QWORD *)(v61 + 16);
      }
      *(_QWORD *)v61 = v64;
      v63 = *(_QWORD *)(v64 + 16);
      *(_QWORD *)(v61 + 8) = v63;
      if ( v63 )
        *(_QWORD *)(v63 + 16) = v61 + 8;
      *(_QWORD *)(v61 + 16) = v64 + 16;
      *(_QWORD *)(v64 + 16) = v61;
      ++v129;
      *(_QWORD *)(*(_QWORD *)(v47 - 8)
                + 32LL * *(unsigned int *)(v47 + 72)
                + 8LL * ((*(_DWORD *)(v47 + 4) & 0x7FFFFFFu) - 1)) = v33;
      v28 = ++v130;
    }
    while ( v117 != v129 );
    v4 = v128;
  }
  v138 = 257;
  v144 = v111;
  v145 = v111 + 48;
  LOWORD(v146) = 0;
  v72 = sub_BD2C40(72, unk_3F148B8);
  v73 = (__int64)v72;
  if ( v72 )
    sub_B4C8A0((__int64)v72, (__int64)v147, 0, 0);
  (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v149 + 2))(v149, v73, &v134, v145, v146);
  v74 = (unsigned int *)v139;
  v75 = (unsigned int *)&v139[16 * (unsigned int)v140];
  if ( v139 != (char *)v75 )
  {
    do
    {
      v76 = *((_QWORD *)v74 + 1);
      v77 = *v74;
      v74 += 4;
      sub_B99FD0(v73, v77, v76);
    }
    while ( v75 != v74 );
  }
  *(_QWORD *)(v4 + 344) = v113;
  nullsub_61();
  v156 = &unk_49DA100;
  nullsub_63();
  if ( v139 != v141 )
    _libc_free((unsigned __int64)v139);
  v143 = 1;
  v139 = ".resume";
  v142 = 3;
  v78 = sub_24EE210(a1, (__int64)&v139, v4, 0, a4, (__int64)v158);
  v143 = 1;
  v139 = ".destroy";
  v142 = 3;
  v79 = sub_24EE210(a1, (__int64)&v139, v4, 1, a4, (__int64)v158);
  v143 = 1;
  v139 = ".cleanup";
  v142 = 3;
  v80 = sub_24EE210(a1, (__int64)&v139, v4, 2, a4, (__int64)v158);
  sub_F62E00(v78, 0, 0, v81, v82, v83);
  sub_F62E00(v79, 0, 0, v84, v85, v86);
  sub_F62E00(v80, 0, 0, v87, v88, v89);
  v90 = *(_QWORD *)(v4 + 312);
  if ( *(_BYTE *)v90 <= 0x1Cu )
  {
    v90 = *(_QWORD *)(*(_QWORD *)(v90 + 24) + 80LL);
    if ( !v90 )
      BUG();
  }
  v91 = *(_QWORD *)(v90 + 32);
  if ( v91 )
    v91 -= 24;
  v147 = (_QWORD *)sub_BD5C60(v91);
  v152 = 512;
  v148 = &v156;
  v139 = v141;
  v149 = &v157;
  LOWORD(v146) = 0;
  v140 = 0x200000000LL;
  v156 = &unk_49DA100;
  v150 = 0;
  v151 = 0;
  v157 = &unk_49DA0B0;
  v153 = 7;
  v154 = 0;
  v155 = 0;
  v144 = 0;
  v145 = 0;
  sub_D5F1F0((__int64)&v139, v91);
  v92 = *(_QWORD *)(v4 + 312);
  v93 = *(_QWORD *)(v4 + 288);
  v134 = "resume.addr";
  v138 = 259;
  v94 = sub_9213A0((unsigned int **)&v139, v93, v92, 0, 0, (__int64)&v134, 7u);
  sub_2463EC0((__int64 *)&v139, v78, v94, 0, 0);
  v95 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v4 - 32LL * (*(_DWORD *)(*(_QWORD *)v4 + 4LL) & 0x7FFFFFF)) + 16LL);
  if ( v95 )
  {
    while ( 1 )
    {
      v96 = *(_QWORD *)(v95 + 24);
      if ( *(_BYTE *)v96 == 85 )
      {
        v97 = *(_QWORD *)(v96 - 32);
        if ( v97 )
        {
          if ( !*(_BYTE *)v97
            && *(_QWORD *)(v97 + 24) == *(_QWORD *)(v96 + 80)
            && (*(_BYTE *)(v97 + 33) & 0x20) != 0
            && *(_DWORD *)(v97 + 36) == 28 )
          {
            break;
          }
        }
      }
      v95 = *(_QWORD *)(v95 + 8);
      if ( !v95 )
        goto LABEL_86;
    }
    v138 = 257;
    v98 = sub_B36550((unsigned int **)&v139, v96, v79, v80, (__int64)&v134, 0);
  }
  else
  {
LABEL_86:
    v98 = v79;
  }
  v99 = *(_QWORD *)(v4 + 312);
  v100 = *(_QWORD *)(v4 + 288);
  v134 = "destroy.addr";
  v138 = 259;
  v101 = sub_9213A0((unsigned int **)&v139, v100, v99, 0, 1u, (__int64)&v134, 7u);
  sub_2463EC0((__int64 *)&v139, v98, v101, 0, 0);
  nullsub_61();
  v156 = &unk_49DA100;
  nullsub_63();
  if ( v139 != v141 )
    _libc_free((unsigned __int64)v139);
  v104 = *(unsigned int *)(a3 + 8);
  if ( v104 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v104 + 1, 8u, v102, v103);
    v104 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v104) = v78;
  v105 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v105;
  if ( v105 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v105 + 1, 8u, v102, v103);
    v105 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v105) = v79;
  v106 = *(unsigned int *)(a3 + 12);
  v107 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v107;
  if ( v107 + 1 > v106 )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v107 + 1, 8u, v102, v103);
    v107 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v107) = v80;
  v108 = *(__int64 **)a3;
  v109 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v109;
  sub_24E5240(a1, v4, v108, v109, v102, v103);
  if ( !v160 )
    _libc_free(v159);
}
