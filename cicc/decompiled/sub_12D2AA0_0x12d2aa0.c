// Function: sub_12D2AA0
// Address: 0x12d2aa0
//
__int64 __fastcall sub_12D2AA0(
        int a1,
        const char **a2,
        int a3,
        int *a4,
        __int64 *a5,
        int *a6,
        __int64 *a7,
        int *a8,
        __int64 *a9,
        int *a10,
        __int64 *a11,
        int *a12,
        __int64 *a13,
        _DWORD *a14)
{
  _DWORD *v14; // rbx
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r13
  int v21; // eax
  int v22; // edx
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rdi
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rbx
  __int64 v31; // r12
  __int64 v32; // r13
  __int64 *v33; // r15
  size_t v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rcx
  _BYTE *v37; // rdi
  _BYTE *v38; // rax
  __int64 v39; // rax
  unsigned int v40; // [rsp+1Ch] [rbp-6FCh]
  __int64 v41; // [rsp+20h] [rbp-6F8h]
  __int64 v42; // [rsp+28h] [rbp-6F0h]
  __int64 v43; // [rsp+30h] [rbp-6E8h]
  __int64 v44; // [rsp+38h] [rbp-6E0h]
  __int64 v45; // [rsp+38h] [rbp-6E0h]
  unsigned __int64 v46; // [rsp+40h] [rbp-6D8h]
  __int64 v47; // [rsp+40h] [rbp-6D8h]
  __int64 v48; // [rsp+48h] [rbp-6D0h] BYREF
  __int64 v49; // [rsp+50h] [rbp-6C8h]
  __int64 v50; // [rsp+58h] [rbp-6C0h]
  int v51; // [rsp+68h] [rbp-6B0h] BYREF
  int v52; // [rsp+70h] [rbp-6A8h]
  _QWORD v53[2]; // [rsp+78h] [rbp-6A0h] BYREF
  char v54; // [rsp+88h] [rbp-690h] BYREF
  _QWORD v55[2]; // [rsp+98h] [rbp-680h] BYREF
  char v56; // [rsp+A8h] [rbp-670h] BYREF
  _QWORD v57[2]; // [rsp+B8h] [rbp-660h] BYREF
  char v58; // [rsp+C8h] [rbp-650h] BYREF
  _QWORD v59[2]; // [rsp+D8h] [rbp-640h] BYREF
  char v60; // [rsp+E8h] [rbp-630h] BYREF
  _QWORD v61[2]; // [rsp+F8h] [rbp-620h] BYREF
  char v62; // [rsp+108h] [rbp-610h] BYREF
  _QWORD v63[2]; // [rsp+118h] [rbp-600h] BYREF
  char v64; // [rsp+128h] [rbp-5F0h] BYREF
  __int64 v65; // [rsp+138h] [rbp-5E0h]
  __int64 v66; // [rsp+140h] [rbp-5D8h]
  __int64 v67; // [rsp+148h] [rbp-5D0h]
  char v68; // [rsp+158h] [rbp-5C0h]
  int v69; // [rsp+160h] [rbp-5B8h]
  int v70; // [rsp+170h] [rbp-5A8h] BYREF
  __int64 v71; // [rsp+178h] [rbp-5A0h]
  int *v72; // [rsp+180h] [rbp-598h]
  int *v73; // [rsp+188h] [rbp-590h]
  __int64 v74; // [rsp+190h] [rbp-588h]
  char *v75; // [rsp+198h] [rbp-580h]
  __int64 v76; // [rsp+1A0h] [rbp-578h]
  char v77; // [rsp+1A8h] [rbp-570h] BYREF
  char *v78; // [rsp+1B8h] [rbp-560h]
  __int64 v79; // [rsp+1C0h] [rbp-558h]
  char v80; // [rsp+1C8h] [rbp-550h] BYREF
  char *v81; // [rsp+1D8h] [rbp-540h]
  __int64 v82; // [rsp+1E0h] [rbp-538h]
  char v83; // [rsp+1E8h] [rbp-530h] BYREF
  char *v84; // [rsp+1F8h] [rbp-520h]
  __int64 v85; // [rsp+200h] [rbp-518h]
  char v86; // [rsp+208h] [rbp-510h] BYREF
  char *v87; // [rsp+218h] [rbp-500h]
  __int64 v88; // [rsp+220h] [rbp-4F8h]
  char v89; // [rsp+228h] [rbp-4F0h] BYREF
  char *v90; // [rsp+238h] [rbp-4E0h]
  __int64 v91; // [rsp+240h] [rbp-4D8h]
  char v92; // [rsp+248h] [rbp-4D0h] BYREF
  char *v93; // [rsp+258h] [rbp-4C0h]
  __int64 v94; // [rsp+260h] [rbp-4B8h]
  char v95; // [rsp+268h] [rbp-4B0h] BYREF
  char *v96; // [rsp+278h] [rbp-4A0h]
  __int64 v97; // [rsp+280h] [rbp-498h]
  char v98; // [rsp+288h] [rbp-490h] BYREF
  char *v99; // [rsp+298h] [rbp-480h]
  __int64 v100; // [rsp+2A0h] [rbp-478h]
  char v101; // [rsp+2A8h] [rbp-470h] BYREF
  char *v102; // [rsp+2B8h] [rbp-460h]
  __int64 v103; // [rsp+2C0h] [rbp-458h]
  char v104; // [rsp+2C8h] [rbp-450h] BYREF
  char *v105; // [rsp+2D8h] [rbp-440h]
  __int64 v106; // [rsp+2E0h] [rbp-438h]
  char v107; // [rsp+2E8h] [rbp-430h] BYREF
  char *v108; // [rsp+2F8h] [rbp-420h]
  __int64 v109; // [rsp+300h] [rbp-418h]
  char v110; // [rsp+308h] [rbp-410h] BYREF
  char *v111; // [rsp+318h] [rbp-400h]
  __int64 v112; // [rsp+320h] [rbp-3F8h]
  char v113; // [rsp+328h] [rbp-3F0h] BYREF
  char *v114; // [rsp+338h] [rbp-3E0h]
  __int64 v115; // [rsp+340h] [rbp-3D8h]
  char v116; // [rsp+348h] [rbp-3D0h] BYREF
  char *v117; // [rsp+358h] [rbp-3C0h]
  __int64 v118; // [rsp+360h] [rbp-3B8h]
  char v119; // [rsp+368h] [rbp-3B0h] BYREF
  char *v120; // [rsp+378h] [rbp-3A0h]
  __int64 v121; // [rsp+380h] [rbp-398h]
  char v122; // [rsp+388h] [rbp-390h] BYREF
  char *v123; // [rsp+398h] [rbp-380h]
  __int64 v124; // [rsp+3A0h] [rbp-378h]
  char v125; // [rsp+3A8h] [rbp-370h] BYREF
  char *v126; // [rsp+3B8h] [rbp-360h]
  __int64 v127; // [rsp+3C0h] [rbp-358h]
  char v128; // [rsp+3C8h] [rbp-350h] BYREF
  char *v129; // [rsp+3D8h] [rbp-340h]
  __int64 v130; // [rsp+3E0h] [rbp-338h]
  char v131; // [rsp+3E8h] [rbp-330h] BYREF
  char *v132; // [rsp+3F8h] [rbp-320h]
  __int64 v133; // [rsp+400h] [rbp-318h]
  char v134; // [rsp+408h] [rbp-310h] BYREF
  char *v135; // [rsp+418h] [rbp-300h]
  __int64 v136; // [rsp+420h] [rbp-2F8h]
  char v137; // [rsp+428h] [rbp-2F0h] BYREF
  char *v138; // [rsp+438h] [rbp-2E0h]
  __int64 v139; // [rsp+440h] [rbp-2D8h]
  char v140; // [rsp+448h] [rbp-2D0h] BYREF
  char *v141; // [rsp+458h] [rbp-2C0h]
  __int64 v142; // [rsp+460h] [rbp-2B8h]
  char v143; // [rsp+468h] [rbp-2B0h] BYREF
  char *v144; // [rsp+478h] [rbp-2A0h]
  __int64 v145; // [rsp+480h] [rbp-298h]
  char v146; // [rsp+488h] [rbp-290h] BYREF
  char *v147; // [rsp+498h] [rbp-280h]
  __int64 v148; // [rsp+4A0h] [rbp-278h]
  char v149; // [rsp+4A8h] [rbp-270h] BYREF
  char *v150; // [rsp+4B8h] [rbp-260h]
  __int64 v151; // [rsp+4C0h] [rbp-258h]
  char v152; // [rsp+4C8h] [rbp-250h] BYREF
  char *v153; // [rsp+4D8h] [rbp-240h]
  __int64 v154; // [rsp+4E0h] [rbp-238h]
  char v155; // [rsp+4E8h] [rbp-230h] BYREF
  char *v156; // [rsp+4F8h] [rbp-220h]
  __int64 v157; // [rsp+500h] [rbp-218h]
  char v158; // [rsp+508h] [rbp-210h] BYREF
  char *v159; // [rsp+518h] [rbp-200h]
  __int64 v160; // [rsp+520h] [rbp-1F8h]
  char v161; // [rsp+528h] [rbp-1F0h] BYREF
  char *v162; // [rsp+538h] [rbp-1E0h]
  __int64 v163; // [rsp+540h] [rbp-1D8h]
  char v164; // [rsp+548h] [rbp-1D0h] BYREF
  char *v165; // [rsp+558h] [rbp-1C0h]
  __int64 v166; // [rsp+560h] [rbp-1B8h]
  char v167; // [rsp+568h] [rbp-1B0h] BYREF
  char *v168; // [rsp+578h] [rbp-1A0h]
  __int64 v169; // [rsp+580h] [rbp-198h]
  char v170; // [rsp+588h] [rbp-190h] BYREF
  char *v171; // [rsp+598h] [rbp-180h]
  __int64 v172; // [rsp+5A0h] [rbp-178h]
  char v173; // [rsp+5A8h] [rbp-170h] BYREF
  char *v174; // [rsp+5B8h] [rbp-160h]
  __int64 v175; // [rsp+5C0h] [rbp-158h]
  char v176; // [rsp+5C8h] [rbp-150h] BYREF
  char *v177; // [rsp+5D8h] [rbp-140h]
  __int64 v178; // [rsp+5E0h] [rbp-138h]
  char v179; // [rsp+5E8h] [rbp-130h] BYREF
  char *v180; // [rsp+5F8h] [rbp-120h]
  __int64 v181; // [rsp+600h] [rbp-118h]
  char v182; // [rsp+608h] [rbp-110h] BYREF
  char *v183; // [rsp+618h] [rbp-100h]
  __int64 v184; // [rsp+620h] [rbp-F8h]
  char v185; // [rsp+628h] [rbp-F0h] BYREF
  char *v186; // [rsp+638h] [rbp-E0h]
  __int64 v187; // [rsp+640h] [rbp-D8h]
  char v188; // [rsp+648h] [rbp-D0h] BYREF
  char *v189; // [rsp+658h] [rbp-C0h]
  __int64 v190; // [rsp+660h] [rbp-B8h]
  char v191; // [rsp+668h] [rbp-B0h] BYREF
  char *v192; // [rsp+678h] [rbp-A0h]
  __int64 v193; // [rsp+680h] [rbp-98h]
  char v194; // [rsp+688h] [rbp-90h] BYREF
  char *v195; // [rsp+698h] [rbp-80h]
  __int64 v196; // [rsp+6A0h] [rbp-78h]
  char v197; // [rsp+6A8h] [rbp-70h] BYREF
  char *v198; // [rsp+6B8h] [rbp-60h]
  __int64 v199; // [rsp+6C0h] [rbp-58h]
  char v200; // [rsp+6C8h] [rbp-50h] BYREF

  v53[0] = &v54;
  v14 = a14;
  v55[0] = &v56;
  v57[0] = &v58;
  v59[0] = &v60;
  v61[0] = &v62;
  v63[0] = &v64;
  v52 = 0;
  v53[1] = 0;
  v54 = 0;
  v55[1] = 0;
  v56 = 0;
  v57[1] = 0;
  v58 = 0;
  v59[1] = 0;
  v60 = 0;
  v61[1] = 0;
  v62 = 0;
  v63[1] = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0x1000000000LL;
  v69 = 0;
  v70 = 0;
  v72 = &v70;
  v73 = &v70;
  v75 = &v77;
  v78 = &v80;
  v81 = &v83;
  v84 = &v86;
  v87 = &v89;
  v90 = &v92;
  v93 = &v95;
  v96 = &v98;
  v99 = &v101;
  v102 = &v104;
  v71 = 0;
  v74 = 0;
  v76 = 0;
  v77 = 0;
  v79 = 0;
  v80 = 0;
  v82 = 0;
  v83 = 0;
  v85 = 0;
  v86 = 0;
  v88 = 0;
  v89 = 0;
  v91 = 0;
  v92 = 0;
  v94 = 0;
  v95 = 0;
  v97 = 0;
  v98 = 0;
  v100 = 0;
  v101 = 0;
  v103 = 0;
  v105 = &v107;
  v108 = &v110;
  v111 = &v113;
  v114 = &v116;
  v117 = &v119;
  v120 = &v122;
  v123 = &v125;
  v126 = &v128;
  v129 = &v131;
  v132 = &v134;
  v135 = &v137;
  v104 = 0;
  v106 = 0;
  v107 = 0;
  v109 = 0;
  v110 = 0;
  v112 = 0;
  v113 = 0;
  v115 = 0;
  v116 = 0;
  v118 = 0;
  v119 = 0;
  v121 = 0;
  v122 = 0;
  v124 = 0;
  v125 = 0;
  v127 = 0;
  v128 = 0;
  v130 = 0;
  v131 = 0;
  v133 = 0;
  v134 = 0;
  v136 = 0;
  v138 = &v140;
  v141 = &v143;
  v144 = &v146;
  v147 = &v149;
  v150 = &v152;
  v153 = &v155;
  v156 = &v158;
  v159 = &v161;
  v162 = &v164;
  v165 = &v167;
  v168 = &v170;
  v137 = 0;
  v139 = 0;
  v140 = 0;
  v142 = 0;
  v143 = 0;
  v145 = 0;
  v146 = 0;
  v148 = 0;
  v149 = 0;
  v151 = 0;
  v152 = 0;
  v154 = 0;
  v155 = 0;
  v157 = 0;
  v158 = 0;
  v160 = 0;
  v161 = 0;
  v163 = 0;
  v164 = 0;
  v166 = 0;
  v167 = 0;
  v169 = 0;
  v171 = &v173;
  v174 = &v176;
  v177 = &v179;
  v180 = &v182;
  v183 = &v185;
  v186 = &v188;
  v189 = &v191;
  v192 = &v194;
  v195 = &v197;
  v198 = &v200;
  v170 = 0;
  v172 = 0;
  v173 = 0;
  v175 = 0;
  v176 = 0;
  v178 = 0;
  v179 = 0;
  v181 = 0;
  v182 = 0;
  v184 = 0;
  v185 = 0;
  v187 = 0;
  v188 = 0;
  v190 = 0;
  v191 = 0;
  v193 = 0;
  v194 = 0;
  v196 = 0;
  v197 = 0;
  v199 = 0;
  v200 = 0;
  v40 = sub_12D1E20((__int64)&v51, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, 1);
  *a14 = v51;
  a14[2] = v52;
  sub_2240AE0(a14 + 4, v53);
  sub_2240AE0(a14 + 12, v55);
  sub_2240AE0(a14 + 20, v57);
  sub_2240AE0(a14 + 28, v59);
  sub_2240AE0(a14 + 36, v61);
  sub_2240AE0(a14 + 44, v63);
  v15 = HIDWORD(v66);
  v48 = 0;
  v49 = 0;
  v50 = 0x1000000000LL;
  if ( HIDWORD(v66) )
  {
    sub_16D1890(&v48, (unsigned int)v66);
    v17 = HIDWORD(v66);
    v16 = (unsigned int)v67;
    v42 = v48;
    v15 = v49;
    v41 = v65;
    HIDWORD(v49) = HIDWORD(v66);
    LODWORD(v50) = v67;
    if ( !(_DWORD)v49 )
    {
      v18 = v48;
      goto LABEL_3;
    }
    v29 = v65;
    v30 = 8LL * (unsigned int)v49 + 8;
    v31 = 0;
    v43 = 8LL * (unsigned int)(v49 - 1);
    v18 = v48;
    while ( 1 )
    {
      v32 = *(_QWORD *)(v29 + v31);
      v33 = (__int64 *)(v18 + v31);
      if ( v32 != -8 )
      {
        if ( v32 )
          break;
      }
      *v33 = v32;
LABEL_14:
      v18 = v48;
      v30 += 4;
      if ( v43 == v31 )
      {
        v14 = a14;
        v15 = v49;
        v17 = HIDWORD(v49);
        LODWORD(v16) = v50;
        goto LABEL_3;
      }
      v29 = v65;
      v31 += 8;
    }
    v34 = *(_QWORD *)v32;
    v44 = *(_QWORD *)v32 + 17LL;
    v46 = *(_QWORD *)v32 + 1LL;
    v27 = malloc(v44, v44, v29, v16, v27, v28);
    if ( !v27 )
    {
      if ( !v44 )
      {
        v39 = malloc(1, 0, v35, v36, 0, v28);
        v27 = 0;
        if ( v39 )
        {
          v37 = (_BYTE *)(v39 + 16);
          v27 = v39;
          goto LABEL_20;
        }
      }
      v45 = v27;
      sub_16BD1C0("Allocation failed");
      v27 = v45;
    }
    v37 = (_BYTE *)(v27 + 16);
    if ( v46 <= 1 )
    {
LABEL_13:
      v37[v34] = 0;
      v16 = v42;
      *(_QWORD *)v27 = v34;
      *(_DWORD *)(v27 + 8) = *(_DWORD *)(v32 + 8);
      *v33 = v27;
      *(_DWORD *)(v42 + v30) = *(_DWORD *)(v41 + v30);
      goto LABEL_14;
    }
LABEL_20:
    v47 = v27;
    v38 = memcpy(v37, (const void *)(v32 + 16), v34);
    v27 = v47;
    v37 = v38;
    goto LABEL_13;
  }
  LODWORD(v16) = 0;
  v17 = 0;
  v18 = 0;
LABEL_3:
  v19 = *((_QWORD *)v14 + 26);
  v20 = (unsigned int)v14[54];
  *((_QWORD *)v14 + 26) = v18;
  v14[54] = v15;
  v21 = v14[55];
  v22 = v14[56];
  v48 = v19;
  LODWORD(v49) = v20;
  v14[55] = v17;
  HIDWORD(v49) = v21;
  v14[56] = v16;
  LODWORD(v50) = v22;
  if ( v21 && (_DWORD)v20 )
  {
    v23 = 8 * v20;
    v24 = 0;
    do
    {
      v25 = *(_QWORD *)(v19 + v24);
      if ( v25 != -8 && v25 )
      {
        _libc_free(v25, v17);
        v19 = v48;
      }
      v24 += 8;
    }
    while ( v23 != v24 );
  }
  _libc_free(v19, v17);
  *((_BYTE *)v14 + 240) = v68;
  sub_12C73A0((__int64)&v51, v17);
  return v40;
}
