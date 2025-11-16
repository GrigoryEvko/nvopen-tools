// Function: sub_38014E0
// Address: 0x38014e0
//
__int64 __fastcall sub_38014E0(
        _QWORD *a1,
        unsigned __int64 *a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r14
  unsigned __int16 *v14; // rax
  __int64 v15; // r15
  __int64 (__fastcall *v16)(__int64, __int64, __int64, __int64, __int64); // r13
  __int64 v17; // rax
  unsigned __int16 v18; // ax
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // r13
  __int128 v22; // kr00_16
  int v23; // edx
  __int64 v24; // rdx
  __int64 v25; // r9
  unsigned int v26; // edx
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r14
  __int64 v30; // r12
  unsigned __int16 *v31; // rax
  __int64 (__fastcall *v32)(__int64, __int64, __int64, __int64, __int64); // r15
  __int64 v33; // rax
  unsigned __int16 v34; // ax
  __int64 v35; // rdx
  __int128 v36; // kr10_16
  __int64 v37; // rdx
  __int64 v38; // r9
  unsigned int v39; // edx
  __int64 v40; // rbx
  __int64 v41; // r9
  unsigned __int16 *v42; // rax
  __int64 v43; // r9
  __int64 v44; // r12
  unsigned int v45; // edx
  __int64 v46; // r14
  unsigned __int16 *v47; // rax
  __int64 v48; // r15
  __int64 (__fastcall *v49)(__int64, __int64, __int64, __int64, __int64); // r13
  __int64 v50; // rax
  unsigned __int16 v51; // ax
  __int64 v52; // rdx
  __int128 v53; // kr20_16
  __int128 v54; // kr30_16
  __int64 v55; // rdx
  unsigned int v56; // edx
  __int64 v57; // r13
  __int64 v58; // rbx
  __int64 v59; // r14
  __int64 v60; // r12
  unsigned __int16 *v61; // rax
  __int64 (__fastcall *v62)(__int64, __int64, __int64, __int64, __int64); // r15
  __int64 v63; // rax
  unsigned __int16 v64; // ax
  __int128 v65; // kr40_16
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // r9
  unsigned int v69; // edx
  __int64 v70; // rbx
  __int64 v71; // r9
  int v72; // r12d
  unsigned __int16 *v73; // rax
  unsigned int v74; // edx
  unsigned __int64 v75; // rcx
  unsigned __int16 *v76; // rdx
  __int64 v77; // r9
  int v78; // edx
  __int128 v80; // rax
  __int64 v81; // r9
  __int128 v82; // rax
  __int128 v83; // rax
  __int64 v84; // r9
  __int128 v85; // rax
  __int64 v86; // r9
  __int128 v87; // [rsp-30h] [rbp-1D0h]
  __int128 v88; // [rsp-20h] [rbp-1C0h]
  __int128 v89; // [rsp-20h] [rbp-1C0h]
  __int128 v90; // [rsp-20h] [rbp-1C0h]
  __int128 v91; // [rsp-10h] [rbp-1B0h]
  __int128 v92; // [rsp-10h] [rbp-1B0h]
  __int128 v93; // [rsp-10h] [rbp-1B0h]
  __int128 v94; // [rsp-10h] [rbp-1B0h]
  __int128 v95; // [rsp-10h] [rbp-1B0h]
  _BOOL4 v96; // [rsp+Ch] [rbp-194h]
  __int64 v97; // [rsp+10h] [rbp-190h]
  _BOOL4 v98; // [rsp+10h] [rbp-190h]
  unsigned int v99; // [rsp+10h] [rbp-190h]
  __int64 v100; // [rsp+18h] [rbp-188h]
  __int64 v101; // [rsp+18h] [rbp-188h]
  __int64 v102; // [rsp+18h] [rbp-188h]
  __int64 v103; // [rsp+18h] [rbp-188h]
  __int64 v104; // [rsp+18h] [rbp-188h]
  __int64 v105; // [rsp+20h] [rbp-180h]
  __int64 v106; // [rsp+20h] [rbp-180h]
  int v107; // [rsp+20h] [rbp-180h]
  __int64 v108; // [rsp+20h] [rbp-180h]
  __int64 v109; // [rsp+20h] [rbp-180h]
  unsigned int v110; // [rsp+20h] [rbp-180h]
  __int128 v111; // [rsp+20h] [rbp-180h]
  unsigned int v114; // [rsp+40h] [rbp-160h]
  __int64 v115; // [rsp+40h] [rbp-160h]
  __int64 v116; // [rsp+40h] [rbp-160h]
  __int64 v117; // [rsp+40h] [rbp-160h]
  __int64 v118; // [rsp+40h] [rbp-160h]
  __int64 v119; // [rsp+40h] [rbp-160h]
  __int64 v120; // [rsp+48h] [rbp-158h]
  unsigned int v121; // [rsp+48h] [rbp-158h]
  _QWORD *v122; // [rsp+48h] [rbp-158h]
  unsigned int v123; // [rsp+48h] [rbp-158h]
  unsigned int v126; // [rsp+58h] [rbp-148h]
  __int128 v127; // [rsp+60h] [rbp-140h]
  __int128 v128; // [rsp+70h] [rbp-130h]
  __int64 v130; // [rsp+90h] [rbp-110h]
  __int128 v131; // [rsp+90h] [rbp-110h]
  __int64 v132; // [rsp+98h] [rbp-108h]
  __int64 v133; // [rsp+A0h] [rbp-100h]
  __int64 v134; // [rsp+A0h] [rbp-100h]
  unsigned __int8 *v135; // [rsp+A0h] [rbp-100h]
  __int64 v136; // [rsp+A8h] [rbp-F8h]
  unsigned __int64 v137; // [rsp+A8h] [rbp-F8h]
  __int128 v138; // [rsp+D0h] [rbp-D0h] BYREF
  __int128 v139; // [rsp+E0h] [rbp-C0h] BYREF
  __int128 v140; // [rsp+F0h] [rbp-B0h] BYREF
  __int128 v141; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v142; // [rsp+110h] [rbp-90h] BYREF
  __int64 v143; // [rsp+118h] [rbp-88h]
  __int16 v144; // [rsp+120h] [rbp-80h]
  __int64 v145; // [rsp+128h] [rbp-78h]
  __int64 v146; // [rsp+130h] [rbp-70h] BYREF
  int v147; // [rsp+138h] [rbp-68h]
  __int128 v148; // [rsp+140h] [rbp-60h]
  __int128 v149; // [rsp+150h] [rbp-50h]
  __int64 v150; // [rsp+160h] [rbp-40h]
  __int64 v151; // [rsp+168h] [rbp-38h]

  v9 = *a2;
  v10 = a2[1];
  *(_QWORD *)&v138 = 0;
  DWORD2(v138) = 0;
  *(_QWORD *)&v139 = 0;
  DWORD2(v139) = 0;
  *(_QWORD *)&v140 = 0;
  DWORD2(v140) = 0;
  *(_QWORD *)&v141 = 0;
  DWORD2(v141) = 0;
  sub_375E6F0((__int64)a1, v9, v10, (__int64)&v138, (__int64)&v139);
  sub_375E6F0((__int64)a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), (__int64)&v140, (__int64)&v141);
  v11 = *a1;
  v12 = a1[1];
  v13 = *(_QWORD *)(v12 + 64);
  v14 = (unsigned __int16 *)(*(_QWORD *)(v139 + 48) + 16LL * DWORD2(v139));
  v15 = *((_QWORD *)v14 + 1);
  v133 = *v14;
  v16 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v17 = sub_2E79000(*(__int64 **)(v12 + 40));
  v18 = v16(v11, v17, v13, v133, v15);
  v19 = v139;
  v134 = v20;
  v21 = *((_QWORD *)&v139 + 1);
  v120 = v18;
  v22 = v141;
  if ( *(_QWORD *)a6 )
  {
    v23 = *(_DWORD *)(a6 + 8);
    v146 = *(_QWORD *)a6;
    v147 = v23;
    v148 = v139;
    v149 = v141;
    v150 = sub_33ED040((_QWORD *)v12, 1u);
    v151 = v24;
    *((_QWORD *)&v91 + 1) = 4;
    *(_QWORD *)&v91 = &v146;
    v142 = v120;
    v143 = v134;
    v144 = 1;
    v145 = 0;
    v27 = (__int64)sub_3411BE0(
                     (_QWORD *)v12,
                     147 - ((unsigned int)(a7 == 0) - 1),
                     a5,
                     (unsigned __int16 *)&v142,
                     2,
                     v25,
                     v91);
  }
  else
  {
    *(_QWORD *)&v85 = sub_33ED040((_QWORD *)v12, 1u);
    *((_QWORD *)&v87 + 1) = v21;
    *(_QWORD *)&v87 = v19;
    v27 = sub_340F900((_QWORD *)v12, 0xD0u, a5, v120, v134, v86, v87, v22, v85);
  }
  v28 = 0;
  v114 = v26;
  v136 = v26;
  if ( *(_DWORD *)(v27 + 68) >= 2u )
    v28 = v27;
  v29 = *a1;
  v30 = a1[1];
  v96 = *(_DWORD *)(v27 + 68) >= 2u;
  v121 = *a4;
  v31 = (unsigned __int16 *)(*(_QWORD *)(v138 + 48) + 16LL * DWORD2(v138));
  v105 = *(_QWORD *)(v30 + 64);
  v97 = *((_QWORD *)v31 + 1);
  v32 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v100 = *v31;
  v33 = sub_2E79000(*(__int64 **)(v30 + 40));
  v34 = v32(v29, v33, v105, v100, v97);
  v36 = v138;
  if ( v28 )
  {
    v101 = v34;
    v146 = v28;
    v147 = v96;
    v148 = v138;
    v149 = v140;
    v106 = v35;
    v150 = sub_33ED040((_QWORD *)v30, v121);
    v151 = v37;
    *((_QWORD *)&v92 + 1) = 4;
    *(_QWORD *)&v92 = &v146;
    v142 = v101;
    v143 = v106;
    v144 = 1;
    v145 = 0;
    v40 = (__int64)sub_3411BE0(
                     (_QWORD *)v30,
                     147 - ((unsigned int)(a7 == 0) - 1),
                     a5,
                     (unsigned __int16 *)&v142,
                     2,
                     v38,
                     v92);
  }
  else
  {
    v99 = v34;
    v111 = v140;
    v104 = v35;
    *(_QWORD *)&v83 = sub_33ED040((_QWORD *)v30, v121);
    v40 = sub_340F900((_QWORD *)v30, 0xD0u, a5, v99, v104, v84, v36, v111, v83);
  }
  v130 = v40;
  v132 = v39;
  if ( *(_DWORD *)(v40 + 68) <= 1u )
  {
    v107 = 0;
    v40 = 0;
  }
  else
  {
    v107 = 1;
  }
  v42 = (unsigned __int16 *)(*(_QWORD *)(v27 + 48) + 16LL * v114);
  *((_QWORD *)&v93 + 1) = v39;
  *(_QWORD *)&v93 = v130;
  *((_QWORD *)&v88 + 1) = v136;
  *(_QWORD *)&v88 = v27;
  *(_QWORD *)&v128 = sub_3406EB0((_QWORD *)a1[1], 0xBAu, a5, *v42, *((_QWORD *)v42 + 1), v41, v88, v93);
  v43 = a1[1];
  v44 = *a1;
  *((_QWORD *)&v128 + 1) = v45;
  v46 = *(_QWORD *)(v43 + 64);
  v122 = (_QWORD *)v43;
  v47 = (unsigned __int16 *)(*(_QWORD *)(v139 + 48) + 16LL * DWORD2(v139));
  v48 = *((_QWORD *)v47 + 1);
  v115 = *v47;
  v49 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v50 = sub_2E79000(*(__int64 **)(v43 + 40));
  v51 = v49(v44, v50, v46, v115, v48);
  v53 = v139;
  v54 = v141;
  if ( v40 )
  {
    v102 = v51;
    v116 = v52;
    v146 = v40;
    v147 = v107;
    v148 = v139;
    v149 = v141;
    v150 = sub_33ED040(v122, 0xEu);
    v151 = v55;
    *((_QWORD *)&v94 + 1) = 4;
    *(_QWORD *)&v94 = &v146;
    v143 = v116;
    v142 = v102;
    v144 = 1;
    v145 = 0;
    v57 = (__int64)sub_3411BE0(
                     v122,
                     147 - ((unsigned int)(a7 == 0) - 1),
                     a5,
                     (unsigned __int16 *)&v142,
                     2,
                     (__int64)v122,
                     v94);
  }
  else
  {
    v110 = v51;
    v119 = v52;
    *(_QWORD *)&v82 = sub_33ED040(v122, 0xEu);
    v57 = sub_340F900(v122, 0xD0u, a5, v110, v119, (__int64)v122, v53, v54, v82);
  }
  v123 = v56;
  v58 = 0;
  v137 = v56 | v136 & 0xFFFFFFFF00000000LL;
  if ( *(_DWORD *)(v57 + 68) >= 2u )
    v58 = v57;
  v59 = *a1;
  v60 = a1[1];
  v98 = *(_DWORD *)(v57 + 68) >= 2u;
  v126 = *a4;
  v61 = (unsigned __int16 *)(*(_QWORD *)(v139 + 48) + 16LL * DWORD2(v139));
  v117 = *(_QWORD *)(v60 + 64);
  v103 = *((_QWORD *)v61 + 1);
  v62 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v108 = *v61;
  v63 = sub_2E79000(*(__int64 **)(v60 + 40));
  v64 = v62(v59, v63, v117, v108, v103);
  v65 = v139;
  v118 = v66;
  v109 = v64;
  if ( v58 )
  {
    v146 = v58;
    v149 = v141;
    v147 = v98;
    v148 = v139;
    v150 = sub_33ED040((_QWORD *)v60, v126);
    v151 = v67;
    *((_QWORD *)&v95 + 1) = 4;
    *(_QWORD *)&v95 = &v146;
    v142 = v109;
    v143 = v118;
    v144 = 1;
    v145 = 0;
    v70 = (__int64)sub_3411BE0(
                     (_QWORD *)v60,
                     147 - ((unsigned int)(a7 == 0) - 1),
                     a5,
                     (unsigned __int16 *)&v142,
                     2,
                     v68,
                     v95);
  }
  else
  {
    v127 = v141;
    *(_QWORD *)&v80 = sub_33ED040((_QWORD *)v60, v126);
    v70 = sub_340F900((_QWORD *)v60, 0xD0u, a5, v109, v118, v81, v65, v127, v80);
  }
  *(_QWORD *)&v131 = v70;
  v72 = 1;
  *((_QWORD *)&v131 + 1) = v69 | v132 & 0xFFFFFFFF00000000LL;
  if ( *(_DWORD *)(v70 + 68) <= 1u )
  {
    v72 = 0;
    v70 = 0;
  }
  v73 = (unsigned __int16 *)(*(_QWORD *)(v57 + 48) + 16LL * v123);
  *((_QWORD *)&v89 + 1) = v137;
  *(_QWORD *)&v89 = v57;
  v135 = sub_3406EB0((_QWORD *)a1[1], 0xBAu, a5, *v73, *((_QWORD *)v73 + 1), v71, v89, v131);
  v75 = v74 | v137 & 0xFFFFFFFF00000000LL;
  v76 = (unsigned __int16 *)(*((_QWORD *)v135 + 6) + 16LL * v74);
  *((_QWORD *)&v90 + 1) = v75;
  *(_QWORD *)&v90 = v135;
  *a2 = (unsigned __int64)sub_3406EB0((_QWORD *)a1[1], 0xBBu, a5, *v76, *((_QWORD *)v76 + 1), v77, v90, v128);
  *((_DWORD *)a2 + 2) = v78;
  *(_QWORD *)a3 = 0;
  *(_DWORD *)(a3 + 8) = 0;
  *(_QWORD *)a6 = v70;
  *(_DWORD *)(a6 + 8) = v72;
  return a6;
}
