// Function: sub_2A9EC20
// Address: 0x2a9ec20
//
__int64 __fastcall sub_2A9EC20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 *a7)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // r12
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r12
  int v28; // eax
  int v29; // eax
  unsigned int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 (__fastcall *v35)(__int64, unsigned int, unsigned __int8 *, _BYTE *, unsigned __int8, char); // rax
  unsigned __int8 *v36; // r14
  __int64 v37; // rax
  __int64 *v38; // rdi
  __int64 v39; // r15
  bool v40; // al
  __int64 v41; // rax
  _QWORD *v42; // rdi
  __int64 v43; // r14
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r15
  bool v48; // al
  char *v49; // rax
  size_t v50; // rdx
  size_t v51; // r14
  __int64 *v52; // rax
  _BYTE *v53; // r14
  __int64 *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  _QWORD *v57; // rdi
  __int64 v58; // r14
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // r14
  _QWORD *v62; // rax
  __int64 v63; // r15
  __int64 v64; // rdi
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rcx
  __int64 v69; // r14
  __int64 v70; // rbx
  __int64 v71; // rdx
  unsigned int v72; // esi
  unsigned __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 (__fastcall *v77)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v78; // r14
  __int64 v79; // rdx
  __int64 v80; // r14
  int v81; // eax
  int v82; // eax
  unsigned int v83; // ecx
  __int64 v84; // rax
  __int64 v85; // rcx
  __int64 v86; // rcx
  __int64 v87; // r14
  _QWORD *v88; // rax
  __int64 v89; // r15
  __int64 v90; // rdi
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 v95; // rcx
  __int64 v96; // r14
  __int64 v97; // rbx
  __int64 v98; // rdx
  unsigned int v99; // esi
  unsigned __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // r15
  unsigned __int64 v105; // r14
  int v106; // eax
  int v107; // eax
  unsigned int v108; // edx
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rdx
  __int64 v112; // r15
  __int64 v113; // rcx
  int v114; // eax
  int v115; // eax
  unsigned int v116; // edx
  __int64 v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rdx
  __int64 v120; // rdi
  __int64 (__fastcall *v121)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v122; // r12
  __int64 v123; // r14
  __int64 v124; // rdi
  __int64 (__fastcall *v125)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v126; // r12
  __int64 v128; // r15
  __int64 v129; // rbx
  __int64 v130; // rdx
  unsigned int v131; // esi
  _QWORD *v132; // rax
  __int64 v133; // rax
  __int64 v134; // rbx
  __int64 v135; // r15
  __int64 v136; // rdx
  unsigned int v137; // esi
  __int64 v138; // r13
  __int64 v139; // rbx
  __int64 v140; // r13
  __int64 v141; // rdx
  unsigned int v142; // esi
  _QWORD *v143; // rax
  __int64 v144; // rax
  __int64 v145; // rbx
  __int64 v146; // r14
  __int64 v147; // rdx
  unsigned int v148; // esi
  __int64 v151; // [rsp+30h] [rbp-120h]
  __int64 v152; // [rsp+30h] [rbp-120h]
  __int64 v154; // [rsp+40h] [rbp-110h]
  __int64 v155; // [rsp+40h] [rbp-110h]
  __int64 v156; // [rsp+40h] [rbp-110h]
  __int64 v157; // [rsp+40h] [rbp-110h]
  __int64 *v158; // [rsp+50h] [rbp-100h]
  __int64 v159; // [rsp+50h] [rbp-100h]
  __int64 v160; // [rsp+50h] [rbp-100h]
  __int64 v161; // [rsp+50h] [rbp-100h]
  __int64 **v162; // [rsp+58h] [rbp-F8h]
  __int64 v163; // [rsp+60h] [rbp-F0h]
  unsigned __int64 v164; // [rsp+60h] [rbp-F0h]
  __int64 v165; // [rsp+60h] [rbp-F0h]
  __int64 v166; // [rsp+60h] [rbp-F0h]
  __int64 v167; // [rsp+60h] [rbp-F0h]
  __int64 v168; // [rsp+60h] [rbp-F0h]
  __int64 v169; // [rsp+70h] [rbp-E0h]
  char *v170; // [rsp+70h] [rbp-E0h]
  __int64 v171; // [rsp+70h] [rbp-E0h]
  __int64 **v172; // [rsp+78h] [rbp-D8h]
  __int64 v175; // [rsp+88h] [rbp-C8h]
  __int64 v176; // [rsp+88h] [rbp-C8h]
  _BYTE *v177; // [rsp+98h] [rbp-B8h] BYREF
  __int64 v178; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 **v179; // [rsp+A8h] [rbp-A8h] BYREF
  __int64 v180; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v181; // [rsp+B8h] [rbp-98h]
  char *v182; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v183; // [rsp+C8h] [rbp-88h]
  __int64 v184; // [rsp+D0h] [rbp-80h]
  unsigned __int64 v185; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v186; // [rsp+E0h] [rbp-70h]
  char *v187; // [rsp+F0h] [rbp-60h] BYREF
  unsigned __int64 v188; // [rsp+F8h] [rbp-58h]
  char *v189; // [rsp+100h] [rbp-50h]
  unsigned __int64 v190; // [rsp+108h] [rbp-48h]
  __int16 v191; // [rsp+110h] [rbp-40h]

  v9 = a2;
  v172 = (__int64 **)sub_BCB2E0(*(_QWORD **)(a2 + 72));
  v162 = (__int64 **)sub_BCB2D0(*(_QWORD **)(a2 + 72));
  v10 = sub_BCB2B0(*(_QWORD **)(a2 + 72));
  v11 = *(_QWORD *)(a1 + 64);
  v158 = (__int64 *)v10;
  v154 = *(_QWORD *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF));
  v151 = *(_QWORD *)(a5 - 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF));
  v12 = sub_BD2C40(72, 1u);
  v13 = (__int64)v12;
  if ( v12 )
    sub_B4C8F0((__int64)v12, v11, 1u, 0, 0);
  v14 = *(_QWORD *)(a2 + 88);
  v15 = *(_QWORD *)(a2 + 56);
  v16 = *(_QWORD *)(a2 + 64);
  v191 = 257;
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v14 + 16LL))(
    v14,
    v13,
    &v187,
    v15,
    v16);
  v20 = *(_QWORD *)a2;
  v21 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v21 )
  {
    do
    {
      v22 = *(_QWORD *)(v20 + 8);
      v23 = *(_DWORD *)v20;
      v20 += 16;
      sub_B99FD0(v13, v23, v22);
    }
    while ( v21 != v20 );
  }
  v163 = *(_QWORD *)(a1 + 64);
  v187 = *(char **)(a1 + 56);
  v188 = v163 & 0xFFFFFFFFFFFFFFFBLL;
  sub_FFB3D0(a3, (unsigned __int64 *)&v187, 1, v17, v18, v19);
  v24 = *(_QWORD *)(a1 + 64);
  *(_WORD *)(v9 + 64) = 0;
  *(_QWORD *)(v9 + 48) = v24;
  *(_QWORD *)(v9 + 56) = v24 + 48;
  v187 = "mismatch_vector_index";
  v191 = 259;
  v25 = sub_D5C860((__int64 *)v9, (__int64)v172, 2, (__int64)&v187);
  v26 = *(_QWORD *)(a1 + 56);
  v27 = v25;
  v28 = *(_DWORD *)(v25 + 4) & 0x7FFFFFF;
  if ( v28 == *(_DWORD *)(v27 + 72) )
  {
    sub_B48D90(v27);
    v28 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
  }
  v29 = (v28 + 1) & 0x7FFFFFF;
  v30 = v29 | *(_DWORD *)(v27 + 4) & 0xF8000000;
  v31 = *(_QWORD *)(v27 - 8) + 32LL * (unsigned int)(v29 - 1);
  *(_DWORD *)(v27 + 4) = v30;
  if ( *(_QWORD *)v31 )
  {
    v32 = *(_QWORD *)(v31 + 8);
    **(_QWORD **)(v31 + 16) = v32;
    if ( v32 )
      *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 16);
  }
  *(_QWORD *)v31 = a6;
  if ( a6 )
  {
    v33 = *(_QWORD *)(a6 + 16);
    *(_QWORD *)(v31 + 8) = v33;
    if ( v33 )
      *(_QWORD *)(v33 + 16) = v31 + 8;
    *(_QWORD *)(v31 + 16) = a6 + 16;
    *(_QWORD *)(a6 + 16) = v31;
  }
  *(_QWORD *)(*(_QWORD *)(v27 - 8) + 32LL * *(unsigned int *)(v27 + 72)
                                   + 8LL * ((*(_DWORD *)(v27 + 4) & 0x7FFFFFFu) - 1)) = v26;
  v34 = *(_QWORD *)(v9 + 80);
  v182 = "avl";
  LOWORD(v186) = 259;
  v35 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v34 + 32LL);
  if ( v35 != sub_9201A0 )
  {
    v36 = (unsigned __int8 *)v35(v34, 15u, a7, (_BYTE *)v27, 1u, 1);
    goto LABEL_19;
  }
  if ( *a7 <= 0x15u && *(_BYTE *)v27 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(15) )
      v36 = (unsigned __int8 *)sub_AD5570(15, (__int64)a7, (unsigned __int8 *)v27, 3, 0);
    else
      v36 = (unsigned __int8 *)sub_AABE40(0xFu, a7, (unsigned __int8 *)v27);
LABEL_19:
    if ( v36 )
      goto LABEL_20;
  }
  v191 = 257;
  v36 = (unsigned __int8 *)sub_B504D0(15, (__int64)a7, v27, (__int64)&v187, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, char **, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
    *(_QWORD *)(v9 + 88),
    v36,
    &v182,
    *(_QWORD *)(v9 + 56),
    *(_QWORD *)(v9 + 64));
  if ( *(_QWORD *)v9 != *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8) )
  {
    v166 = v9;
    v128 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
    v129 = *(_QWORD *)v9;
    do
    {
      v130 = *(_QWORD *)(v129 + 8);
      v131 = *(_DWORD *)v129;
      v129 += 16;
      sub_B99FD0((__int64)v36, v131, v130);
    }
    while ( v128 != v129 );
    v9 = v166;
  }
  sub_B447F0(v36, 1);
  sub_B44850(v36, 1);
LABEL_20:
  v169 = sub_BCDE10(v158, *(_DWORD *)(a1 + 4));
  v37 = sub_AD64C0((__int64)v162, *(unsigned int *)(a1 + 4), 0);
  v38 = *(__int64 **)(v9 + 72);
  v182 = (char *)v36;
  v191 = 257;
  HIDWORD(v180) = 0;
  v183 = v37;
  v184 = sub_ACD6D0(v38);
  v179 = v172;
  v177 = (_BYTE *)v27;
  v39 = sub_B33D10(v9, 0x98u, (__int64)&v179, 1, (int)&v182, 3, (unsigned int)v180, (__int64)&v187);
  v40 = sub_B4DE30(a4);
  v191 = 257;
  v41 = sub_921130((unsigned int **)v9, (__int64)v158, v154, &v177, 1, (__int64)&v187, v40 ? 3 : 0);
  v42 = *(_QWORD **)(v9 + 72);
  v43 = v41;
  LODWORD(v41) = *(_DWORD *)(v169 + 32);
  BYTE4(v179) = *(_BYTE *)(v169 + 8) == 18;
  LODWORD(v179) = v41;
  v44 = (__int64 *)sub_BCB2A0(v42);
  v45 = sub_BCE1B0(v44, (__int64)v179);
  v46 = sub_AD62B0(v45);
  v184 = v39;
  v187 = "lhs.load";
  BYTE4(v178) = 0;
  v182 = (char *)v43;
  v180 = v169;
  v191 = 259;
  v155 = v46;
  v183 = v46;
  v164 = v39;
  v181 = *(_QWORD *)(v43 + 8);
  v47 = sub_B33D10(v9, 0x1B6u, (__int64)&v180, 2, (int)&v182, 3, v178, (__int64)&v187);
  v48 = sub_B4DE30(a5);
  v191 = 257;
  v182 = (char *)sub_921130((unsigned int **)v9, (__int64)v158, v151, &v177, 1, (__int64)&v187, v48 ? 3 : 0);
  v187 = "rhs.load";
  BYTE4(v178) = 0;
  v183 = v155;
  v180 = v169;
  v191 = 259;
  v184 = v164;
  v181 = *(_QWORD *)(v43 + 8);
  v159 = sub_B33D10(v9, 0x1B6u, (__int64)&v180, 2, (int)&v182, 3, v178, (__int64)&v187);
  v49 = sub_B52C80(33);
  v51 = v50;
  v170 = v49;
  v52 = (__int64 *)sub_BD5C60(v47);
  v53 = (_BYTE *)sub_B9B140(v52, v170, v51);
  v54 = (__int64 *)sub_BD5C60(v47);
  v55 = sub_B9F6F0(v54, v53);
  v187 = "mismatch.cmp";
  v184 = v55;
  BYTE4(v180) = 0;
  v182 = (char *)v47;
  v183 = v159;
  v185 = v155;
  v191 = 259;
  v186 = v164;
  v178 = *(_QWORD *)(v47 + 8);
  v56 = sub_B33D10(v9, 0x1B2u, (__int64)&v178, 1, (int)&v182, 5, v180, (__int64)&v187);
  v57 = *(_QWORD **)(v9 + 72);
  HIDWORD(v178) = 0;
  v191 = 257;
  v58 = v56;
  v182 = (char *)v56;
  v59 = sub_BCB2A0(v57);
  v183 = sub_ACD640(v59, 0, 0);
  v184 = v155;
  v185 = v164;
  v180 = (__int64)v162;
  v181 = *(_QWORD *)(v58 + 8);
  v60 = sub_B33D10(v9, 0x19Fu, (__int64)&v180, 2, (int)&v182, 4, v178, (__int64)&v187);
  v191 = 257;
  v171 = v60;
  v152 = sub_92B530((unsigned int **)v9, 0x21u, v60, (_BYTE *)v164, (__int64)&v187);
  v61 = *(_QWORD *)(a1 + 72);
  v160 = *(_QWORD *)(a1 + 80);
  v62 = sub_BD2C40(72, 3u);
  v63 = (__int64)v62;
  if ( v62 )
    sub_B4C9A0((__int64)v62, v61, v160, v152, 3u, 0, 0, 0);
  v64 = *(_QWORD *)(v9 + 88);
  v65 = *(_QWORD *)(v9 + 56);
  v191 = 257;
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, _QWORD))(*(_QWORD *)v64 + 16LL))(
    v64,
    v63,
    &v187,
    v65,
    *(_QWORD *)(v9 + 64));
  v68 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
  if ( *(_QWORD *)v9 != v68 )
  {
    v156 = v9;
    v69 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
    v70 = *(_QWORD *)v9;
    do
    {
      v71 = *(_QWORD *)(v70 + 8);
      v72 = *(_DWORD *)v70;
      v70 += 16;
      sub_B99FD0(v63, v72, v71);
    }
    while ( v69 != v70 );
    v9 = v156;
  }
  v73 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFBLL;
  v187 = *(char **)(a1 + 64);
  v188 = v73;
  v74 = *(_QWORD *)(a1 + 80);
  v189 = v187;
  v190 = v74 & 0xFFFFFFFFFFFFFFFBLL;
  sub_FFB3D0(a3, (unsigned __int64 *)&v187, 2, v68, v66, v67);
  v75 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(v9 + 48) = v75;
  *(_QWORD *)(v9 + 56) = v75 + 48;
  *(_WORD *)(v9 + 64) = 0;
  LOWORD(v186) = 257;
  if ( v172 == *(__int64 ***)(v164 + 8) )
  {
    v78 = (_BYTE *)v164;
    goto LABEL_32;
  }
  v76 = *(_QWORD *)(v9 + 80);
  v77 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v76 + 120LL);
  if ( v77 != sub_920130 )
  {
    v78 = (_BYTE *)v77(v76, 39u, (_BYTE *)v164, (__int64)v172);
    goto LABEL_31;
  }
  if ( *(_BYTE *)v164 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v78 = (_BYTE *)sub_ADAB70(39, v164, v172, 0);
    else
      v78 = (_BYTE *)sub_AA93C0(0x27u, v164, (__int64)v172);
LABEL_31:
    if ( v78 )
      goto LABEL_32;
  }
  v191 = 257;
  v132 = sub_BD2C40(72, 1u);
  v78 = v132;
  if ( v132 )
    sub_B515B0((__int64)v132, v164, (__int64)v172, (__int64)&v187, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, char **, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
    *(_QWORD *)(v9 + 88),
    v78,
    &v182,
    *(_QWORD *)(v9 + 56),
    *(_QWORD *)(v9 + 64));
  v133 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
  if ( *(_QWORD *)v9 != v133 )
  {
    v167 = v9;
    v134 = *(_QWORD *)v9;
    v135 = v133;
    do
    {
      v136 = *(_QWORD *)(v134 + 8);
      v137 = *(_DWORD *)v134;
      v134 += 16;
      sub_B99FD0((__int64)v78, v137, v136);
    }
    while ( v135 != v134 );
    v9 = v167;
  }
LABEL_32:
  v191 = 257;
  v79 = sub_929C50((unsigned int **)v9, (_BYTE *)v27, v78, (__int64)&v187, 1u, 1);
  v80 = *(_QWORD *)(a1 + 80);
  v81 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
  if ( v81 == *(_DWORD *)(v27 + 72) )
  {
    v168 = v79;
    sub_B48D90(v27);
    v79 = v168;
    v81 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
  }
  v82 = (v81 + 1) & 0x7FFFFFF;
  v83 = v82 | *(_DWORD *)(v27 + 4) & 0xF8000000;
  v84 = *(_QWORD *)(v27 - 8) + 32LL * (unsigned int)(v82 - 1);
  *(_DWORD *)(v27 + 4) = v83;
  if ( *(_QWORD *)v84 )
  {
    v85 = *(_QWORD *)(v84 + 8);
    **(_QWORD **)(v84 + 16) = v85;
    if ( v85 )
      *(_QWORD *)(v85 + 16) = *(_QWORD *)(v84 + 16);
  }
  *(_QWORD *)v84 = v79;
  if ( v79 )
  {
    v86 = *(_QWORD *)(v79 + 16);
    *(_QWORD *)(v84 + 8) = v86;
    if ( v86 )
      *(_QWORD *)(v86 + 16) = v84 + 8;
    *(_QWORD *)(v84 + 16) = v79 + 16;
    *(_QWORD *)(v79 + 16) = v84;
  }
  *(_QWORD *)(*(_QWORD *)(v27 - 8) + 32LL * *(unsigned int *)(v27 + 72)
                                   + 8LL * ((*(_DWORD *)(v27 + 4) & 0x7FFFFFFu) - 1)) = v80;
  v191 = 257;
  v161 = sub_92B530((unsigned int **)v9, 0x21u, v79, a7, (__int64)&v187);
  v87 = *(_QWORD *)(a1 + 64);
  v157 = *(_QWORD *)(a1 + 48);
  v88 = sub_BD2C40(72, 3u);
  v89 = (__int64)v88;
  if ( v88 )
    sub_B4C9A0((__int64)v88, v87, v157, v161, 3u, 0, 0, 0);
  v90 = *(_QWORD *)(v9 + 88);
  v91 = *(_QWORD *)(v9 + 56);
  v92 = *(_QWORD *)(v9 + 64);
  v191 = 257;
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v90 + 16LL))(
    v90,
    v89,
    &v187,
    v91,
    v92);
  v95 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
  if ( *(_QWORD *)v9 != v95 )
  {
    v165 = v9;
    v96 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
    v97 = *(_QWORD *)v9;
    do
    {
      v98 = *(_QWORD *)(v97 + 8);
      v99 = *(_DWORD *)v97;
      v97 += 16;
      sub_B99FD0(v89, v99, v98);
    }
    while ( v96 != v97 );
    v9 = v165;
  }
  v100 = *(_QWORD *)(a1 + 64) & 0xFFFFFFFFFFFFFFFBLL;
  v187 = *(char **)(a1 + 80);
  v188 = v100;
  v101 = *(_QWORD *)(a1 + 48);
  v189 = v187;
  v190 = v101 & 0xFFFFFFFFFFFFFFFBLL;
  sub_FFB3D0(a3, (unsigned __int64 *)&v187, 2, v95, v93, v94);
  v102 = *(_QWORD *)(a1 + 72);
  HIBYTE(v191) = 1;
  *(_WORD *)(v9 + 64) = 0;
  *(_QWORD *)(v9 + 48) = v102;
  *(_QWORD *)(v9 + 56) = v102 + 48;
  v187 = "ctz";
  LOBYTE(v191) = 3;
  v103 = sub_D5C860((__int64 *)v9, *(_QWORD *)(v171 + 8), 1, (__int64)&v187);
  v104 = *(_QWORD *)(a1 + 64);
  v105 = v103;
  v106 = *(_DWORD *)(v103 + 4) & 0x7FFFFFF;
  if ( v106 == *(_DWORD *)(v105 + 72) )
  {
    sub_B48D90(v105);
    v106 = *(_DWORD *)(v105 + 4) & 0x7FFFFFF;
  }
  v107 = (v106 + 1) & 0x7FFFFFF;
  v108 = v107 | *(_DWORD *)(v105 + 4) & 0xF8000000;
  v109 = *(_QWORD *)(v105 - 8) + 32LL * (unsigned int)(v107 - 1);
  *(_DWORD *)(v105 + 4) = v108;
  if ( *(_QWORD *)v109 )
  {
    v110 = *(_QWORD *)(v109 + 8);
    **(_QWORD **)(v109 + 16) = v110;
    if ( v110 )
      *(_QWORD *)(v110 + 16) = *(_QWORD *)(v109 + 16);
  }
  *(_QWORD *)v109 = v171;
  v111 = *(_QWORD *)(v171 + 16);
  *(_QWORD *)(v109 + 8) = v111;
  if ( v111 )
    *(_QWORD *)(v111 + 16) = v109 + 8;
  *(_QWORD *)(v109 + 16) = v171 + 16;
  *(_QWORD *)(v171 + 16) = v109;
  *(_QWORD *)(*(_QWORD *)(v105 - 8)
            + 32LL * *(unsigned int *)(v105 + 72)
            + 8LL * ((*(_DWORD *)(v105 + 4) & 0x7FFFFFFu) - 1)) = v104;
  v187 = "mismatch_vector_index";
  v191 = 259;
  v112 = sub_D5C860((__int64 *)v9, *(_QWORD *)(v27 + 8), 1, (__int64)&v187);
  v113 = *(_QWORD *)(a1 + 64);
  v114 = *(_DWORD *)(v112 + 4) & 0x7FFFFFF;
  if ( v114 == *(_DWORD *)(v112 + 72) )
  {
    v176 = *(_QWORD *)(a1 + 64);
    sub_B48D90(v112);
    v113 = v176;
    v114 = *(_DWORD *)(v112 + 4) & 0x7FFFFFF;
  }
  v115 = (v114 + 1) & 0x7FFFFFF;
  v116 = v115 | *(_DWORD *)(v112 + 4) & 0xF8000000;
  v117 = *(_QWORD *)(v112 - 8) + 32LL * (unsigned int)(v115 - 1);
  *(_DWORD *)(v112 + 4) = v116;
  if ( *(_QWORD *)v117 )
  {
    v118 = *(_QWORD *)(v117 + 8);
    **(_QWORD **)(v117 + 16) = v118;
    if ( v118 )
      *(_QWORD *)(v118 + 16) = *(_QWORD *)(v117 + 16);
  }
  *(_QWORD *)v117 = v27;
  v119 = *(_QWORD *)(v27 + 16);
  *(_QWORD *)(v117 + 8) = v119;
  if ( v119 )
    *(_QWORD *)(v119 + 16) = v117 + 8;
  *(_QWORD *)(v117 + 16) = v27 + 16;
  *(_QWORD *)(v27 + 16) = v117;
  *(_QWORD *)(*(_QWORD *)(v112 - 8)
            + 32LL * *(unsigned int *)(v112 + 72)
            + 8LL * ((*(_DWORD *)(v112 + 4) & 0x7FFFFFFu) - 1)) = v113;
  LOWORD(v186) = 257;
  if ( v172 == *(__int64 ***)(v105 + 8) )
  {
    v122 = (_BYTE *)v105;
    goto LABEL_67;
  }
  v120 = *(_QWORD *)(v9 + 80);
  v121 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v120 + 120LL);
  if ( v121 != sub_920130 )
  {
    v122 = (_BYTE *)v121(v120, 39u, (_BYTE *)v105, (__int64)v172);
    goto LABEL_66;
  }
  if ( *(_BYTE *)v105 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v122 = (_BYTE *)sub_ADAB70(39, v105, v172, 0);
    else
      v122 = (_BYTE *)sub_AA93C0(0x27u, v105, (__int64)v172);
LABEL_66:
    if ( v122 )
      goto LABEL_67;
  }
  v191 = 257;
  v143 = sub_BD2C40(72, 1u);
  v122 = v143;
  if ( v143 )
    sub_B515B0((__int64)v143, v105, (__int64)v172, (__int64)&v187, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, char **, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
    *(_QWORD *)(v9 + 88),
    v122,
    &v182,
    *(_QWORD *)(v9 + 56),
    *(_QWORD *)(v9 + 64));
  v144 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
  if ( *(_QWORD *)v9 != v144 )
  {
    v175 = v9;
    v145 = *(_QWORD *)v9;
    v146 = v144;
    do
    {
      v147 = *(_QWORD *)(v145 + 8);
      v148 = *(_DWORD *)v145;
      v145 += 16;
      sub_B99FD0((__int64)v122, v148, v147);
    }
    while ( v146 != v145 );
    v9 = v175;
  }
LABEL_67:
  v191 = 257;
  v123 = sub_929C50((unsigned int **)v9, (_BYTE *)v112, v122, (__int64)&v187, 1u, 1);
  LOWORD(v186) = 257;
  if ( v162 == *(__int64 ***)(v123 + 8) )
    return v123;
  v124 = *(_QWORD *)(v9 + 80);
  v125 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v124 + 120LL);
  if ( v125 != sub_920130 )
  {
    v126 = v125(v124, 38u, (_BYTE *)v123, (__int64)v162);
    goto LABEL_72;
  }
  if ( *(_BYTE *)v123 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v126 = sub_ADAB70(38, v123, v162, 0);
    else
      v126 = sub_AA93C0(0x26u, v123, (__int64)v162);
LABEL_72:
    if ( v126 )
      return v126;
  }
  v191 = 257;
  v126 = sub_B51D30(38, v123, (__int64)v162, (__int64)&v187, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
    *(_QWORD *)(v9 + 88),
    v126,
    &v182,
    *(_QWORD *)(v9 + 56),
    *(_QWORD *)(v9 + 64));
  v138 = 16LL * *(unsigned int *)(v9 + 8);
  v139 = *(_QWORD *)v9;
  v140 = v139 + v138;
  while ( v140 != v139 )
  {
    v141 = *(_QWORD *)(v139 + 8);
    v142 = *(_DWORD *)v139;
    v139 += 16;
    sub_B99FD0(v126, v142, v141);
  }
  return v126;
}
