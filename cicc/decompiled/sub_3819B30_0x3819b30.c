// Function: sub_3819B30
// Address: 0x3819b30
//
__int64 __fastcall sub_3819B30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r9
  __int128 v7; // xmm0
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v11; // rax
  unsigned __int16 v12; // si
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rax
  unsigned int v22; // r15d
  __int128 v23; // rax
  __int64 v24; // r9
  __int128 v25; // rax
  __int64 v26; // r9
  __int128 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int16 v30; // ax
  __int64 v31; // rdx
  __int128 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int16 v38; // ax
  __int64 v39; // rdx
  __int128 v40; // rax
  __int64 v41; // r9
  __int64 v42; // r9
  __int64 v43; // rdx
  int v44; // eax
  _QWORD *v45; // r13
  unsigned int v46; // edx
  __int64 v47; // r9
  unsigned __int8 *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r15
  unsigned __int8 *v51; // r14
  __int64 v52; // r9
  __int128 v53; // rax
  __int64 v54; // r9
  unsigned int v55; // edx
  unsigned int v56; // edx
  __int64 v57; // r9
  __int64 v58; // r13
  unsigned int v59; // edx
  __int128 v60; // rax
  __int64 v61; // rax
  int v62; // edx
  int v63; // edx
  int v64; // edx
  _QWORD *v66; // r13
  unsigned int v67; // edx
  __int64 v68; // r9
  unsigned __int8 *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r15
  unsigned __int8 *v72; // r14
  __int64 v73; // r9
  __int128 v74; // rax
  __int64 v75; // r9
  unsigned int v76; // edx
  unsigned int v77; // edx
  __int64 v78; // r9
  unsigned int v79; // edx
  __int64 v80; // rax
  int v81; // edx
  int v82; // edx
  __int64 v83; // r13
  __int128 v84; // rax
  int v85; // edx
  _QWORD *v86; // r13
  unsigned int v87; // edx
  __int64 v88; // r9
  __int128 v89; // rax
  __int64 v90; // r9
  __int128 v91; // rax
  __int64 v92; // r9
  _QWORD *v93; // r13
  unsigned int v94; // edx
  __int128 v95; // rax
  __int64 v96; // r9
  unsigned int v97; // edx
  __int64 v98; // r9
  __int64 v99; // r13
  unsigned int v100; // edx
  __int128 v101; // rax
  __int64 v102; // rax
  int v103; // ecx
  int v104; // edx
  int v105; // edx
  int v106; // edx
  __int64 v107; // rdx
  __int128 v108; // [rsp-20h] [rbp-280h]
  __int128 v109; // [rsp-20h] [rbp-280h]
  __int128 v110; // [rsp-10h] [rbp-270h]
  __int128 v111; // [rsp-10h] [rbp-270h]
  __int128 v113; // [rsp+10h] [rbp-250h]
  __int128 v114; // [rsp+20h] [rbp-240h]
  __int64 v115; // [rsp+30h] [rbp-230h]
  __int64 v116; // [rsp+38h] [rbp-228h]
  __int64 v117; // [rsp+40h] [rbp-220h]
  _QWORD *v118; // [rsp+40h] [rbp-220h]
  __int128 v119; // [rsp+40h] [rbp-220h]
  __int64 (__fastcall *v120)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+50h] [rbp-210h]
  __int64 v121; // [rsp+50h] [rbp-210h]
  __int64 v122; // [rsp+50h] [rbp-210h]
  __int64 v123; // [rsp+50h] [rbp-210h]
  __int64 v124; // [rsp+58h] [rbp-208h]
  unsigned int v126; // [rsp+68h] [rbp-1F8h]
  __int64 v127; // [rsp+70h] [rbp-1F0h]
  __int64 v128; // [rsp+70h] [rbp-1F0h]
  __int64 v129; // [rsp+70h] [rbp-1F0h]
  __int128 v130; // [rsp+70h] [rbp-1F0h]
  __int128 v131; // [rsp+70h] [rbp-1F0h]
  __int128 v132; // [rsp+70h] [rbp-1F0h]
  _QWORD *v133; // [rsp+80h] [rbp-1E0h]
  __int64 (__fastcall *v134)(__int64, __int64, __int64, __int64, __int64); // [rsp+80h] [rbp-1E0h]
  unsigned __int8 *v135; // [rsp+80h] [rbp-1E0h]
  __int128 v136; // [rsp+80h] [rbp-1E0h]
  unsigned __int8 *v137; // [rsp+80h] [rbp-1E0h]
  __int128 v138; // [rsp+90h] [rbp-1D0h]
  __int128 v139; // [rsp+90h] [rbp-1D0h]
  __int128 v140; // [rsp+90h] [rbp-1D0h]
  __int128 v141; // [rsp+90h] [rbp-1D0h]
  unsigned __int16 v142; // [rsp+A0h] [rbp-1C0h]
  __int64 v143; // [rsp+A0h] [rbp-1C0h]
  __int128 v144; // [rsp+A0h] [rbp-1C0h]
  __int128 v145; // [rsp+A0h] [rbp-1C0h]
  __int128 v146; // [rsp+A0h] [rbp-1C0h]
  int v147; // [rsp+D8h] [rbp-188h]
  int v148; // [rsp+138h] [rbp-128h]
  int v149; // [rsp+198h] [rbp-C8h]
  unsigned int v150; // [rsp+1E0h] [rbp-80h] BYREF
  __int64 v151; // [rsp+1E8h] [rbp-78h]
  __int64 v152; // [rsp+1F0h] [rbp-70h] BYREF
  int v153; // [rsp+1F8h] [rbp-68h]
  __int128 v154; // [rsp+200h] [rbp-60h] BYREF
  __int128 v155; // [rsp+210h] [rbp-50h] BYREF
  __int64 v156; // [rsp+220h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *a1;
  v7 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 40));
  v8 = *(_QWORD *)(v5 + 40);
  v9 = *(unsigned int *)(v5 + 48);
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v11 = *(__int16 **)(a2 + 48);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v14 = a1[1];
  if ( v10 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v155, v6, *(_QWORD *)(v14 + 64), v12, v13);
    LOWORD(v150) = WORD4(v155);
    v151 = v156;
  }
  else
  {
    v150 = v10(v6, *(_QWORD *)(v14 + 64), v12, v13);
    v151 = v107;
  }
  v15 = *(_QWORD *)(v8 + 48) + 16 * v9;
  v16 = *(_QWORD *)(v15 + 8);
  v142 = *(_WORD *)v15;
  *(_QWORD *)&v155 = sub_2D5B750((unsigned __int16 *)&v150);
  *((_QWORD *)&v155 + 1) = v17;
  v18 = sub_CA1930(&v155);
  v19 = *(_QWORD *)(a2 + 80);
  v126 = v18;
  v152 = v19;
  if ( v19 )
    sub_B96E90((__int64)&v152, v19, 1);
  v20 = *(_DWORD *)(a2 + 72);
  DWORD2(v154) = 0;
  *(_QWORD *)&v155 = 0;
  v153 = v20;
  v21 = *(_QWORD *)(a2 + 40);
  DWORD2(v155) = 0;
  *(_QWORD *)&v154 = 0;
  sub_375E510((__int64)a1, *(_QWORD *)v21, *(_QWORD *)(v21 + 8), (__int64)&v154, (__int64)&v155);
  v22 = v142;
  *(_QWORD *)&v23 = sub_3400BD0(a1[1], v126, (__int64)&v152, v142, v16, 0, (__m128i)v7, 0);
  v138 = v23;
  *(_QWORD *)&v25 = sub_3406EB0((_QWORD *)a1[1], 0x39u, (__int64)&v152, v142, v16, v24, v7, v23);
  v114 = v25;
  *(_QWORD *)&v27 = sub_3406EB0((_QWORD *)a1[1], 0x39u, (__int64)&v152, v142, v16, v26, v138, v7);
  v28 = a1[1];
  v113 = v27;
  v133 = (_QWORD *)v28;
  v117 = *a1;
  v120 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 528LL);
  v127 = *(_QWORD *)(v28 + 64);
  v29 = sub_2E79000(*(__int64 **)(v28 + 40));
  v30 = v120(v117, v29, v127, v142, v16);
  v121 = v31;
  LODWORD(v127) = v30;
  *(_QWORD *)&v32 = sub_33ED040(v133, 0xCu);
  v33 = sub_340F900(v133, 0xD0u, (__int64)&v152, v127, v121, (__int64)v133, v7, v138, v32);
  v116 = v34;
  v118 = (_QWORD *)a1[1];
  v115 = v33;
  *(_QWORD *)&v138 = sub_3400BD0((__int64)v118, 0, (__int64)&v152, v142, v16, 0, (__m128i)v7, 0);
  v35 = a1[1];
  *((_QWORD *)&v138 + 1) = v36;
  v122 = v142;
  v128 = *a1;
  v134 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v143 = *(_QWORD *)(v35 + 64);
  v37 = sub_2E79000(*(__int64 **)(v35 + 40));
  v38 = v134(v128, v37, v143, v122, v16);
  v129 = v39;
  LODWORD(v134) = v38;
  *(_QWORD *)&v40 = sub_33ED040(v118, 0x11u);
  v123 = sub_340F900(v118, 0xD0u, (__int64)&v152, (unsigned int)v134, v129, v41, v7, v138, v40);
  v124 = v43;
  v44 = *(_DWORD *)(a2 + 24);
  switch ( v44 )
  {
    case 191:
      *(_QWORD *)&v141 = sub_3406EB0((_QWORD *)a1[1], 0xBFu, (__int64)&v152, v150, v151, v42, v155, v7);
      v86 = (_QWORD *)a1[1];
      *((_QWORD *)&v141 + 1) = v87;
      *(_QWORD *)&v89 = sub_3406EB0(v86, 0xBEu, (__int64)&v152, v150, v151, v88, v155, v113);
      v119 = v89;
      *(_QWORD *)&v91 = sub_3406EB0((_QWORD *)a1[1], 0xC0u, (__int64)&v152, v150, v151, v90, v154, v7);
      *(_QWORD *)&v146 = sub_3406EB0(v86, 0xBBu, (__int64)&v152, v150, v151, v92, v91, v119);
      v93 = (_QWORD *)a1[1];
      *((_QWORD *)&v146 + 1) = v94;
      *(_QWORD *)&v95 = sub_3400BD0((__int64)v93, v126 - 1, (__int64)&v152, v22, v16, 0, (__m128i)v7, 0);
      *(_QWORD *)&v132 = sub_3406EB0(v93, 0xBFu, (__int64)&v152, v150, v151, v96, v155, v95);
      *((_QWORD *)&v132 + 1) = v97;
      v137 = sub_3406EB0((_QWORD *)a1[1], 0xBFu, (__int64)&v152, v150, v151, v98, v155, v114);
      v99 = a1[1];
      *((_QWORD *)&v109 + 1) = v100;
      *(_QWORD *)&v109 = v137;
      *(_QWORD *)&v101 = sub_3288B20(v99, (int)&v152, v150, v151, v115, v116, v146, v109, 0);
      v102 = sub_3288B20(v99, (int)&v152, v150, v151, v123, v124, v154, v101, 0);
      v103 = v151;
      v147 = v104;
      v105 = v150;
      *(_QWORD *)a3 = v102;
      *(_DWORD *)(a3 + 8) = v147;
      *(_QWORD *)a4 = sub_3288B20(a1[1], (int)&v152, v105, v103, v115, v116, v141, v132, 0);
      *(_DWORD *)(a4 + 8) = v106;
      break;
    case 192:
      *(_QWORD *)&v139 = sub_3406EB0((_QWORD *)a1[1], 0xC0u, (__int64)&v152, v150, v151, v42, v155, v7);
      v45 = (_QWORD *)a1[1];
      *((_QWORD *)&v139 + 1) = v46;
      v48 = sub_3406EB0(v45, 0xBEu, (__int64)&v152, v150, v151, v47, v155, v113);
      v50 = v49;
      v51 = v48;
      *(_QWORD *)&v53 = sub_3406EB0((_QWORD *)a1[1], 0xC0u, (__int64)&v152, v150, v151, v52, v154, v7);
      *((_QWORD *)&v110 + 1) = v50;
      *(_QWORD *)&v110 = v51;
      *(_QWORD *)&v144 = sub_3406EB0(v45, 0xBBu, (__int64)&v152, v150, v151, v54, v53, v110);
      *((_QWORD *)&v144 + 1) = v55;
      *(_QWORD *)&v130 = sub_3400BD0(a1[1], 0, (__int64)&v152, v150, v151, 0, (__m128i)v7, 0);
      *((_QWORD *)&v130 + 1) = v56;
      v135 = sub_3406EB0((_QWORD *)a1[1], 0xC0u, (__int64)&v152, v150, v151, v57, v155, v114);
      v58 = a1[1];
      *((_QWORD *)&v108 + 1) = v59;
      *(_QWORD *)&v108 = v135;
      *(_QWORD *)&v60 = sub_3288B20(v58, (int)&v152, v150, v151, v115, v116, v144, v108, 0);
      v61 = sub_3288B20(v58, (int)&v152, v150, v151, v123, v124, v154, v60, 0);
      v148 = v62;
      v63 = v150;
      *(_QWORD *)a3 = v61;
      *(_DWORD *)(a3 + 8) = v148;
      *(_QWORD *)a4 = sub_3288B20(a1[1], (int)&v152, v63, v151, v115, v116, v139, v130, 0);
      *(_DWORD *)(a4 + 8) = v64;
      break;
    case 190:
      *(_QWORD *)&v145 = sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v152, v150, v151, v42, v154, v7);
      v66 = (_QWORD *)a1[1];
      *((_QWORD *)&v145 + 1) = v67;
      v69 = sub_3406EB0(v66, 0xC0u, (__int64)&v152, v150, v151, v68, v154, v113);
      v71 = v70;
      v72 = v69;
      *(_QWORD *)&v74 = sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v152, v150, v151, v73, v155, v7);
      *((_QWORD *)&v111 + 1) = v71;
      *(_QWORD *)&v111 = v72;
      *(_QWORD *)&v140 = sub_3406EB0(v66, 0xBBu, (__int64)&v152, v150, v151, v75, v74, v111);
      *((_QWORD *)&v140 + 1) = v76;
      *(_QWORD *)&v136 = sub_3400BD0(a1[1], 0, (__int64)&v152, v150, v151, 0, (__m128i)v7, 0);
      *((_QWORD *)&v136 + 1) = v77;
      *(_QWORD *)&v131 = sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v152, v150, v151, v78, v154, v114);
      *((_QWORD *)&v131 + 1) = v79;
      v80 = sub_3288B20(a1[1], (int)&v152, v150, v151, v115, v116, v145, v136, 0);
      v149 = v81;
      v82 = v150;
      *(_QWORD *)a3 = v80;
      *(_DWORD *)(a3 + 8) = v149;
      v83 = a1[1];
      *(_QWORD *)&v84 = sub_3288B20(v83, (int)&v152, v82, v151, v115, v116, v140, v131, 0);
      *(_QWORD *)a4 = sub_3288B20(v83, (int)&v152, v150, v151, v123, v124, v155, v84, 0);
      *(_DWORD *)(a4 + 8) = v85;
      break;
    default:
      BUG();
  }
  if ( v152 )
    sub_B91220((__int64)&v152, v152);
  return 1;
}
