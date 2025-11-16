// Function: sub_2469530
// Address: 0x2469530
//
_QWORD **__fastcall sub_2469530(__int64 a1)
{
  _QWORD **result; // rax
  int *v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 (__fastcall *v6)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int8 *v10; // r12
  __int64 v11; // rcx
  __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 (__fastcall *v20)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v21; // r14
  __int64 v22; // r12
  __int64 v23; // rcx
  __int64 v24; // r12
  __int64 v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r12
  unsigned __int64 v31; // r12
  __int64 v32; // rax
  __int64 **v33; // r10
  _BYTE *v34; // r14
  __int64 v35; // rax
  _BYTE *v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  int v40; // edx
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __int64 **v43; // r10
  _BYTE *v44; // r14
  _BYTE *v45; // rax
  unsigned __int64 v46; // rax
  __int64 v47; // rsi
  int v48; // eax
  _BYTE *v49; // r14
  __int64 v50; // rax
  __int64 **v51; // r11
  __int64 v52; // rax
  _BYTE *v53; // rax
  unsigned __int64 v54; // rax
  int v55; // eax
  __int64 v56; // rsi
  int v57; // eax
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  _BYTE *v60; // rax
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  __int64 **v63; // r10
  _BYTE *v64; // r14
  _BYTE *v65; // rax
  unsigned __int64 v66; // rax
  __int64 v67; // rax
  int v68; // edx
  _BYTE *v69; // r14
  __int64 v70; // rax
  __int64 **v71; // rdi
  __int64 v72; // rax
  _BYTE *v73; // rax
  unsigned __int64 v74; // rax
  __int64 v75; // rax
  int v76; // edx
  unsigned __int64 v77; // rdx
  __int64 v78; // rax
  _BYTE *v79; // rax
  unsigned __int64 v80; // rax
  unsigned __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // r14
  unsigned __int8 *v84; // r12
  __int64 v85; // rax
  __int64 v86; // r12
  unsigned int *v87; // r12
  unsigned int *v88; // rbx
  __int64 v89; // rdx
  unsigned int v90; // esi
  __int64 v91; // r12
  unsigned int *v92; // r12
  unsigned int *v93; // rbx
  __int64 v94; // rdx
  unsigned int v95; // esi
  __int64 v96; // rax
  __int64 v97; // r12
  __int64 v98; // rax
  __int64 v99; // rax
  _BYTE *v100; // r12
  __int64 v101; // rax
  _BYTE *v102; // rax
  __int64 v103; // r12
  __int64 v104; // rax
  __int64 *v105; // rax
  __int64 v106; // rax
  __int64 v107; // rsi
  unsigned int v108; // r14d
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // r9
  unsigned int v114; // ecx
  unsigned int v115; // eax
  _QWORD **v116; // [rsp+40h] [rbp-200h]
  _BYTE *v117; // [rsp+48h] [rbp-1F8h]
  int *v118; // [rsp+48h] [rbp-1F8h]
  unsigned __int16 v119; // [rsp+56h] [rbp-1EAh]
  unsigned __int16 v120; // [rsp+58h] [rbp-1E8h]
  unsigned __int16 v121; // [rsp+5Ah] [rbp-1E6h]
  unsigned __int16 v122; // [rsp+5Ch] [rbp-1E4h]
  unsigned __int16 v123; // [rsp+5Eh] [rbp-1E2h]
  _BYTE *v124; // [rsp+60h] [rbp-1E0h]
  _BYTE *v125; // [rsp+60h] [rbp-1E0h]
  unsigned __int64 v126; // [rsp+68h] [rbp-1D8h]
  _BYTE *v127; // [rsp+70h] [rbp-1D0h]
  __int64 **v128; // [rsp+70h] [rbp-1D0h]
  unsigned __int64 v129; // [rsp+70h] [rbp-1D0h]
  int *v130; // [rsp+70h] [rbp-1D0h]
  __int64 v131; // [rsp+78h] [rbp-1C8h]
  __int64 **v132; // [rsp+78h] [rbp-1C8h]
  __int64 **v133; // [rsp+78h] [rbp-1C8h]
  _BYTE *v134; // [rsp+78h] [rbp-1C8h]
  __int64 **v135; // [rsp+78h] [rbp-1C8h]
  __int64 v136; // [rsp+78h] [rbp-1C8h]
  unsigned __int8 *v137; // [rsp+80h] [rbp-1C0h]
  unsigned __int8 *v138; // [rsp+88h] [rbp-1B8h]
  __int64 v139; // [rsp+90h] [rbp-1B0h]
  __int64 v140; // [rsp+90h] [rbp-1B0h]
  __int64 **v141; // [rsp+90h] [rbp-1B0h]
  _QWORD **v142; // [rsp+98h] [rbp-1A8h]
  int v143; // [rsp+B8h] [rbp-188h]
  int v144[8]; // [rsp+C0h] [rbp-180h] BYREF
  __int16 v145; // [rsp+E0h] [rbp-160h]
  int v146[8]; // [rsp+F0h] [rbp-150h] BYREF
  __int16 v147; // [rsp+110h] [rbp-130h]
  unsigned int v148[8]; // [rsp+120h] [rbp-120h] BYREF
  __int16 v149; // [rsp+140h] [rbp-100h]
  int v150[8]; // [rsp+150h] [rbp-F0h] BYREF
  __int16 v151; // [rsp+170h] [rbp-D0h]
  unsigned int *v152; // [rsp+180h] [rbp-C0h] BYREF
  unsigned int v153; // [rsp+188h] [rbp-B8h]
  char v154; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v155; // [rsp+1B8h] [rbp-88h]
  __int64 v156; // [rsp+1C0h] [rbp-80h]
  __int64 *v157; // [rsp+1C8h] [rbp-78h]
  __int64 v158; // [rsp+1D0h] [rbp-70h]
  __int64 v159; // [rsp+1D8h] [rbp-68h]
  void *v160; // [rsp+200h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 40) )
  {
    sub_23D0AB0((__int64)&v152, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL), 0, 0, 0);
    v96 = *(_QWORD *)(a1 + 16);
    v151 = 257;
    v97 = *(_QWORD *)(v96 + 152);
    v98 = sub_BCB2E0(v157);
    v99 = sub_A82CA0(&v152, v98, v97, 0, 0, (__int64)v150);
    *(_QWORD *)(a1 + 192) = v99;
    v100 = (_BYTE *)v99;
    v101 = *(_QWORD *)(a1 + 16);
    v151 = 257;
    v102 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v101 + 80), 192, 0);
    v103 = sub_929C50(&v152, v102, v100, (__int64)v150, 0, 0);
    v104 = *(_QWORD *)(a1 + 16);
    v151 = 257;
    v105 = (__int64 *)sub_BCB2B0(*(_QWORD **)(v104 + 72));
    v106 = sub_23DEB90((__int64 *)&v152, v105, v103, (__int64)v150);
    v107 = (unsigned __int8)byte_4FE8EA8;
    *(_QWORD *)(a1 + 184) = v106;
    *(_WORD *)(v106 + 2) = v107 | *(_WORD *)(v106 + 2) & 0xFFC0;
    LODWORD(v106) = (unsigned __int8)v107;
    BYTE1(v106) = 1;
    v108 = v106;
    v109 = sub_BCB2B0(v157);
    v110 = sub_AD6530(v109, v107);
    sub_B34240((__int64)&v152, *(_QWORD *)(a1 + 184), v110, v103, v108, 0, 0, 0, 0);
    v111 = *(_QWORD *)(a1 + 16);
    v148[1] = 0;
    v151 = 257;
    v112 = sub_AD64C0(*(_QWORD *)(v111 + 80), 800, 0);
    v113 = sub_B33C40((__int64)&v152, 0x16Eu, v103, v112, v148[0], (__int64)v150);
    v114 = (unsigned __int8)byte_4FE8EA8;
    v115 = (unsigned __int8)byte_4FE8EA8;
    BYTE1(v114) = 1;
    BYTE1(v115) = 1;
    sub_B343C0(
      (__int64)&v152,
      0xEEu,
      *(_QWORD *)(a1 + 184),
      v114,
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 136LL),
      v115,
      v113,
      0,
      0,
      0,
      0,
      0);
    sub_F94A20(&v152, 238);
  }
  v138 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL), 64, 0);
  v137 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL), 128, 0);
  result = *(_QWORD ***)(a1 + 32);
  v116 = &result[*(unsigned int *)(a1 + 40)];
  if ( v116 != result )
  {
    v142 = *(_QWORD ***)(a1 + 32);
    v3 = v150;
    while ( 1 )
    {
      v30 = (__int64)*v142;
      sub_2468350((__int64)&v152, *v142);
      v31 = *(_QWORD *)(v30 - 32LL * (*(_DWORD *)(v30 + 4) & 0x7FFFFFF));
      v141 = (__int64 **)sub_BCE3C0(v157, 0);
      v32 = *(_QWORD *)(a1 + 16);
      v145 = 257;
      v151 = 257;
      v33 = *(__int64 ***)(v32 + 96);
      v149 = 257;
      v132 = v33;
      v34 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v32 + 80), 0, 0);
      v35 = *(_QWORD *)(a1 + 16);
      v147 = 257;
      v36 = (_BYTE *)sub_24633A0((__int64 *)&v152, 0x2Fu, v31, *(__int64 ***)(v35 + 80), (__int64)v146, 0, v143, 0);
      v37 = sub_929C50(&v152, v36, v34, (__int64)v148, 0, 0);
      LODWORD(v34) = sub_24633A0((__int64 *)&v152, 0x30u, v37, v132, (__int64)v3, 0, v143, 0);
      v38 = *(_QWORD *)(a1 + 16);
      v151 = 257;
      v39 = sub_BCB2E0(*(_QWORD **)(v38 + 72));
      v40 = v123;
      BYTE1(v40) = 0;
      v123 = (unsigned __int8)v123;
      v41 = sub_A82CA0(&v152, v39, (int)v34, v40, 0, (__int64)v3);
      v126 = sub_24633A0((__int64 *)&v152, 0x30u, v41, v141, (__int64)v144, 0, v150[0], 0);
      v42 = *(_QWORD *)(a1 + 16);
      v151 = 257;
      v43 = *(__int64 ***)(v42 + 96);
      v149 = 257;
      v133 = v43;
      v44 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v42 + 80), 8, 0);
      v147 = 257;
      v45 = (_BYTE *)sub_24633A0(
                       (__int64 *)&v152,
                       0x2Fu,
                       v31,
                       *(__int64 ***)(*(_QWORD *)(a1 + 16) + 80LL),
                       (__int64)v146,
                       0,
                       v144[0],
                       0);
      v46 = sub_929C50(&v152, v45, v44, (__int64)v148, 0, 0);
      LODWORD(v44) = sub_24633A0((__int64 *)&v152, 0x30u, v46, v133, (__int64)v3, 0, v144[0], 0);
      v151 = 257;
      v47 = sub_BCB2E0(*(_QWORD **)(*(_QWORD *)(a1 + 16) + 72LL));
      v48 = v122;
      BYTE1(v48) = 0;
      v122 = (unsigned __int8)v122;
      v49 = (_BYTE *)sub_A82CA0(&v152, v47, (int)v44, v48, 0, (__int64)v3);
      v151 = 257;
      v50 = *(_QWORD *)(a1 + 16);
      v51 = *(__int64 ***)(v50 + 96);
      v149 = 257;
      v128 = v51;
      v134 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v50 + 80), 24, 0);
      v52 = *(_QWORD *)(a1 + 16);
      v147 = 257;
      v53 = (_BYTE *)sub_24633A0((__int64 *)&v152, 0x2Fu, v31, *(__int64 ***)(v52 + 80), (__int64)v146, 0, v144[0], 0);
      v54 = sub_929C50(&v152, v53, v134, (__int64)v148, 0, 0);
      v55 = sub_24633A0((__int64 *)&v152, 0x30u, v54, v128, (__int64)v3, 0, v144[0], 0);
      v151 = 257;
      LODWORD(v134) = v55;
      v56 = sub_BCB2D0(v157);
      v57 = v121;
      BYTE1(v57) = 0;
      v121 = (unsigned __int8)v121;
      v58 = sub_A82CA0(&v152, v56, (int)v134, v57, 0, (__int64)v3);
      v59 = *(_QWORD *)(a1 + 16);
      v151 = 257;
      v60 = (_BYTE *)sub_24633A0((__int64 *)&v152, 0x28u, v58, *(__int64 ***)(v59 + 80), (__int64)v3, 0, v148[0], 0);
      v151 = 257;
      v149 = 257;
      v117 = v60;
      v61 = sub_929C50(&v152, v49, v60, (__int64)v148, 0, 0);
      v129 = sub_24633A0((__int64 *)&v152, 0x30u, v61, v141, (__int64)v3, 0, v146[0], 0);
      v62 = *(_QWORD *)(a1 + 16);
      v151 = 257;
      v63 = *(__int64 ***)(v62 + 96);
      v149 = 257;
      v135 = v63;
      v64 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v62 + 80), 16, 0);
      v147 = 257;
      v65 = (_BYTE *)sub_24633A0(
                       (__int64 *)&v152,
                       0x2Fu,
                       v31,
                       *(__int64 ***)(*(_QWORD *)(a1 + 16) + 80LL),
                       (__int64)v146,
                       0,
                       v144[0],
                       0);
      v66 = sub_929C50(&v152, v65, v64, (__int64)v148, 0, 0);
      LODWORD(v64) = sub_24633A0((__int64 *)&v152, 0x30u, v66, v135, (__int64)v3, 0, v144[0], 0);
      v151 = 257;
      v67 = sub_BCB2E0(*(_QWORD **)(*(_QWORD *)(a1 + 16) + 72LL));
      v68 = v120;
      BYTE1(v68) = 0;
      v120 = (unsigned __int8)v120;
      v69 = (_BYTE *)sub_A82CA0(&v152, v67, (int)v64, v68, 0, (__int64)v3);
      v151 = 257;
      v70 = *(_QWORD *)(a1 + 16);
      v71 = *(__int64 ***)(v70 + 96);
      v149 = 257;
      v124 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v70 + 80), 28, 0);
      v72 = *(_QWORD *)(a1 + 16);
      v147 = 257;
      v73 = (_BYTE *)sub_24633A0((__int64 *)&v152, 0x2Fu, v31, *(__int64 ***)(v72 + 80), (__int64)v146, 0, v144[0], 0);
      v74 = sub_929C50(&v152, v73, v124, (__int64)v148, 0, 0);
      LODWORD(v31) = sub_24633A0((__int64 *)&v152, 0x30u, v74, v71, (__int64)v3, 0, v144[0], 0);
      v151 = 257;
      v75 = sub_BCB2D0(v157);
      v76 = v119;
      BYTE1(v76) = 0;
      v119 = (unsigned __int8)v119;
      v77 = sub_A82CA0(&v152, v75, v31, v76, 0, (__int64)v3);
      v78 = *(_QWORD *)(a1 + 16);
      v151 = 257;
      v79 = (_BYTE *)sub_24633A0((__int64 *)&v152, 0x28u, v77, *(__int64 ***)(v78 + 80), (__int64)v3, 0, v148[0], 0);
      v151 = 257;
      v149 = 257;
      v125 = v79;
      v80 = sub_929C50(&v152, v69, v79, (__int64)v148, 0, 0);
      v81 = sub_24633A0((__int64 *)&v152, 0x30u, v80, v141, (__int64)v3, 0, v146[0], 0);
      v151 = 257;
      v136 = v81;
      v82 = sub_929C50(&v152, v138, v117, (__int64)v3, 0, 0);
      v83 = *(_QWORD *)(a1 + 24);
      v84 = (unsigned __int8 *)v82;
      v85 = sub_BCB2B0(v157);
      v127 = **(_BYTE **)(v83 + 8)
           ? sub_2465B30((__int64 *)v83, v129, (__int64)&v152, v85, 1)
           : (_BYTE *)sub_2463FC0(v83, v129, &v152, 0x103u);
      *(_QWORD *)v148 = v84;
      v151 = 257;
      v4 = *(_QWORD *)(a1 + 184);
      v5 = sub_BCB2B0(v157);
      v139 = sub_921130(&v152, v5, v4, (_BYTE **)v148, 1, (__int64)v3, 3u);
      v149 = 257;
      v6 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v158 + 32LL);
      if ( v6 != sub_9201A0 )
        break;
      if ( *v138 <= 0x15u && *v84 <= 0x15u )
      {
        if ( (unsigned __int8)sub_AC47B0(15) )
          v7 = sub_AD5570(15, (__int64)v138, v84, 0, 0);
        else
          v7 = sub_AABE40(0xFu, v138, v84);
LABEL_11:
        if ( v7 )
          goto LABEL_12;
      }
      v151 = 257;
      v7 = sub_B504D0(15, (__int64)v138, (__int64)v84, (__int64)v3, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, unsigned int *, __int64, __int64))(*(_QWORD *)v159 + 16LL))(
        v159,
        v7,
        v148,
        v155,
        v156);
      v86 = 4LL * v153;
      if ( v152 != &v152[v86] )
      {
        v118 = v3;
        v87 = &v152[v86];
        v88 = v152;
        do
        {
          v89 = *((_QWORD *)v88 + 1);
          v90 = *v88;
          v88 += 4;
          sub_B99FD0(v7, v90, v89);
        }
        while ( v87 != v88 );
        v3 = v118;
      }
LABEL_12:
      sub_B343C0((__int64)&v152, 0xEEu, (__int64)v127, 0x103u, v139, 0x103u, v7, 0, 0, 0, 0, 0);
      v151 = 257;
      v8 = sub_929C50(&v152, v137, v125, (__int64)v3, 0, 0);
      v9 = *(_QWORD *)(a1 + 24);
      v10 = (unsigned __int8 *)v8;
      v11 = sub_BCB2B0(v157);
      if ( **(_BYTE **)(v9 + 8) )
        v131 = (__int64)sub_2465B30((__int64 *)v9, v136, (__int64)&v152, v11, 1);
      else
        v131 = sub_2463FC0(v9, v136, &v152, 0x103u);
      v151 = 257;
      v149 = 257;
      v12 = sub_BCB2D0(v157);
      v13 = (_BYTE *)sub_ACD640(v12, 64, 0);
      v14 = *(_QWORD *)(a1 + 184);
      *(_QWORD *)v146 = v13;
      v15 = sub_BCB2B0(v157);
      v16 = sub_921130(&v152, v15, v14, (_BYTE **)v146, 1, (__int64)v148, 3u);
      *(_QWORD *)v146 = v10;
      v17 = v16;
      v18 = sub_BCB2B0(v157);
      v19 = sub_921130(&v152, v18, v17, (_BYTE **)v146, 1, (__int64)v3, 3u);
      v149 = 257;
      v140 = v19;
      v20 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v158 + 32LL);
      if ( v20 == sub_9201A0 )
      {
        if ( *v137 > 0x15u || *v10 > 0x15u )
        {
LABEL_33:
          v151 = 257;
          v21 = sub_B504D0(15, (__int64)v137, (__int64)v10, (__int64)v3, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, unsigned int *, __int64, __int64))(*(_QWORD *)v159 + 16LL))(
            v159,
            v21,
            v148,
            v155,
            v156);
          v91 = 4LL * v153;
          if ( v152 != &v152[v91] )
          {
            v130 = v3;
            v92 = &v152[v91];
            v93 = v152;
            do
            {
              v94 = *((_QWORD *)v93 + 1);
              v95 = *v93;
              v93 += 4;
              sub_B99FD0(v21, v95, v94);
            }
            while ( v92 != v93 );
            v3 = v130;
          }
          goto LABEL_20;
        }
        if ( (unsigned __int8)sub_AC47B0(15) )
          v21 = sub_AD5570(15, (__int64)v137, v10, 0, 0);
        else
          v21 = sub_AABE40(0xFu, v137, v10);
      }
      else
      {
        v21 = v20(v158, 15u, v137, v10, 0, 0);
      }
      if ( !v21 )
        goto LABEL_33;
LABEL_20:
      sub_B343C0((__int64)&v152, 0xEEu, v131, 0x103u, v140, 0x103u, v21, 0, 0, 0, 0, 0);
      v22 = *(_QWORD *)(a1 + 24);
      v23 = sub_BCB2B0(v157);
      if ( **(_BYTE **)(v22 + 8) )
        v24 = (__int64)sub_2465B30((__int64 *)v22, v126, (__int64)&v152, v23, 1);
      else
        v24 = sub_2463FC0(v22, v126, &v152, 0x104u);
      v151 = 257;
      v25 = sub_BCB2D0(v157);
      v26 = (_BYTE *)sub_ACD640(v25, 192, 0);
      v27 = *(_QWORD *)(a1 + 184);
      *(_QWORD *)v148 = v26;
      v28 = sub_BCB2B0(v157);
      v29 = sub_921130(&v152, v28, v27, (_BYTE **)v148, 1, (__int64)v3, 3u);
      sub_B343C0((__int64)&v152, 0xEEu, v24, 0x104u, v29, 0x104u, *(_QWORD *)(a1 + 192), 0, 0, 0, 0, 0);
      nullsub_61();
      v160 = &unk_49DA100;
      nullsub_63();
      if ( v152 != (unsigned int *)&v154 )
        _libc_free((unsigned __int64)v152);
      result = ++v142;
      if ( v116 == v142 )
        return result;
    }
    v7 = v6(v158, 15u, v138, v84, 0, 0);
    goto LABEL_11;
  }
  return result;
}
