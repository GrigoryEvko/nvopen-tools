// Function: sub_3329E60
// Address: 0x3329e60
//
__int64 __fastcall sub_3329E60(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int16 *v5; // rax
  __int16 v6; // dx
  __int64 v7; // rax
  __int128 v8; // xmm0
  unsigned __int16 *v9; // rax
  __int64 v10; // r14
  unsigned int v11; // r15d
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  int v19; // eax
  int v20; // edx
  unsigned int *v21; // r13
  __int128 v22; // rax
  __int128 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r13
  __int128 v26; // rax
  int v27; // r9d
  __int128 v28; // rax
  __int128 v29; // rax
  __int128 v30; // rax
  int v31; // r9d
  __int128 v32; // rax
  __int128 v33; // rax
  __int64 v34; // r13
  __int128 v35; // rax
  int v36; // r9d
  __int128 v37; // rax
  __int128 v38; // rax
  int v39; // r9d
  __int128 v40; // rax
  int v41; // r9d
  __int128 v42; // rax
  int v43; // r9d
  __int128 v44; // rax
  int v45; // r9d
  __int128 v46; // rax
  __int64 v47; // r13
  __int128 v48; // rax
  int v49; // r9d
  __int128 v50; // rax
  __int128 v51; // rax
  __int128 v52; // rax
  __int128 v53; // rax
  __int128 v54; // rax
  int v55; // r9d
  __int128 v56; // rax
  __int128 v57; // rax
  __int128 v58; // rax
  int v59; // r9d
  __int128 v60; // rax
  int v61; // r9d
  __int128 v62; // rax
  __int64 v63; // r13
  __int64 v64; // rdx
  __int128 v65; // rax
  __int128 v66; // rax
  int v67; // r9d
  __int128 v68; // rax
  int v69; // r9d
  __int128 v70; // rax
  __int64 v71; // r13
  int v72; // r9d
  __int128 v73; // rax
  int v74; // r9d
  __int128 v75; // rax
  __int64 v76; // r13
  int v77; // r9d
  __int128 v78; // rax
  int v79; // r9d
  __int128 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r13
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r15
  __int64 v87; // r14
  __int128 v88; // rax
  __int128 v89; // rax
  int v90; // r9d
  __int128 v91; // rax
  int v92; // r9d
  void **v93; // rbx
  void **v94; // r14
  void **v95; // r14
  __int64 v96; // rdx
  void **v97; // rdi
  void **v98; // r12
  void **v99; // r13
  void **v100; // r12
  __int128 v101; // [rsp+0h] [rbp-200h]
  __int64 v102; // [rsp+0h] [rbp-200h]
  __int128 v103; // [rsp+0h] [rbp-200h]
  __int128 v104; // [rsp+0h] [rbp-200h]
  __int128 v105; // [rsp+10h] [rbp-1F0h]
  __int128 v106; // [rsp+20h] [rbp-1E0h]
  __int128 v107; // [rsp+20h] [rbp-1E0h]
  __int128 v108; // [rsp+30h] [rbp-1D0h]
  __int128 v109; // [rsp+30h] [rbp-1D0h]
  __int128 v110; // [rsp+40h] [rbp-1C0h]
  __int128 v111; // [rsp+40h] [rbp-1C0h]
  __int128 v112; // [rsp+50h] [rbp-1B0h]
  __int128 v113; // [rsp+50h] [rbp-1B0h]
  __int128 v114; // [rsp+50h] [rbp-1B0h]
  __int128 v115; // [rsp+70h] [rbp-190h]
  __int128 v116; // [rsp+70h] [rbp-190h]
  __int128 v117; // [rsp+70h] [rbp-190h]
  __int128 v118; // [rsp+70h] [rbp-190h]
  __int128 v119; // [rsp+70h] [rbp-190h]
  __int128 v120; // [rsp+70h] [rbp-190h]
  __int128 v121; // [rsp+80h] [rbp-180h]
  __int128 v122; // [rsp+A0h] [rbp-160h]
  __int64 v123; // [rsp+B0h] [rbp-150h]
  int v124; // [rsp+BCh] [rbp-144h]
  unsigned int v125; // [rsp+C0h] [rbp-140h]
  unsigned int v126; // [rsp+C8h] [rbp-138h]
  int v127; // [rsp+D0h] [rbp-130h]
  __int128 v128; // [rsp+D0h] [rbp-130h]
  int v129; // [rsp+E0h] [rbp-120h]
  __int128 v130; // [rsp+E0h] [rbp-120h]
  __int128 v131; // [rsp+E0h] [rbp-120h]
  __int64 (__fastcall *v132)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+F0h] [rbp-110h]
  unsigned int v133; // [rsp+F0h] [rbp-110h]
  __int128 v134; // [rsp+F0h] [rbp-110h]
  __int128 v135; // [rsp+F0h] [rbp-110h]
  __int128 v136; // [rsp+100h] [rbp-100h]
  int v137; // [rsp+110h] [rbp-F0h]
  __int64 v138; // [rsp+118h] [rbp-E8h]
  void *v139; // [rsp+118h] [rbp-E8h]
  __int128 v140; // [rsp+120h] [rbp-E0h]
  __int64 v141; // [rsp+120h] [rbp-E0h]
  __int64 v142; // [rsp+130h] [rbp-D0h] BYREF
  int v143; // [rsp+138h] [rbp-C8h]
  unsigned int v144; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v145; // [rsp+148h] [rbp-B8h]
  void *v146; // [rsp+150h] [rbp-B0h] BYREF
  void **v147; // [rsp+158h] [rbp-A8h]
  void *v148; // [rsp+170h] [rbp-90h] BYREF
  void **v149; // [rsp+178h] [rbp-88h]
  void *v150; // [rsp+190h] [rbp-70h] BYREF
  void **v151; // [rsp+198h] [rbp-68h]
  void *v152; // [rsp+1B0h] [rbp-50h] BYREF
  void **v153; // [rsp+1B8h] [rbp-48h]

  v4 = *(_QWORD *)(a2 + 80);
  v142 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v142, v4, 1);
  v143 = *(_DWORD *)(a2 + 72);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v145 = *((_QWORD *)v5 + 1);
  v7 = *(_QWORD *)(a2 + 40);
  LOWORD(v144) = v6;
  v8 = (__int128)_mm_loadu_si128((const __m128i *)v7);
  v140 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v7 + 40) + 48LL) + 16LL * *(unsigned int *)(v7 + 48));
  v10 = *((_QWORD *)v9 + 1);
  v11 = *v9;
  v12 = sub_327FF20((unsigned __int16 *)&v144, v4);
  v123 = v13;
  v126 = v12;
  if ( ((_WORD)v12 || v13) && *(_DWORD *)(a2 + 24) != 110 )
  {
    v16 = *(_QWORD *)(a1 + 16);
    v132 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 528LL);
    v138 = *(_QWORD *)(v16 + 64);
    v17 = *(_QWORD *)(a1 + 8);
    v18 = sub_2E79000(*(__int64 **)(v16 + 40));
    v19 = v132(v17, v18, v138, v11, v10);
    v127 = v20;
    v137 = v19;
    v21 = (unsigned int *)sub_300AC80((unsigned __int16 *)&v144, v18);
    v133 = sub_C336B0(v21);
    v129 = sub_C336C0((__int64)v21);
    v124 = sub_C336A0((__int64)v21);
    *(_QWORD *)&v22 = sub_3401400(*(_QWORD *)(a1 + 16), v133, (unsigned int)&v142, v11, v10, 0, 0);
    v122 = v22;
    *(_QWORD *)&v23 = sub_3401400(*(_QWORD *)(a1 + 16), v129, (unsigned int)&v142, v11, v10, 0, 0);
    v112 = v23;
    *(_QWORD *)&v115 = sub_3401400(*(_QWORD *)(a1 + 16), 2 * v133, (unsigned int)&v142, v11, v10, 0, 0);
    *((_QWORD *)&v115 + 1) = v24;
    sub_C43310(&v146, v21, (unsigned __int64)"1.0", 3u);
    v139 = sub_C33340();
    if ( v146 == v139 )
      sub_C3C790(&v152, (_QWORD **)&v146);
    else
      sub_C33EB0(&v152, (__int64 *)&v146);
    sub_3329C90(&v148, (__int64 *)&v152, v133, 1);
    if ( v152 == v139 )
    {
      if ( v153 )
      {
        v96 = (__int64)*(v153 - 1);
        v97 = &v153[3 * v96];
        if ( v153 != v97 )
        {
          v98 = &v153[3 * v96];
          do
          {
            v98 -= 3;
            if ( v139 == *v98 )
              sub_969EE0((__int64)v98);
            else
              sub_C338F0((__int64)v98);
          }
          while ( v153 != v98 );
          v97 = v98;
        }
        j_j_j___libc_free_0_0((unsigned __int64)(v97 - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v152);
    }
    v125 = v124 + v129;
    if ( v146 == v139 )
      sub_C3C790(&v152, (_QWORD **)&v146);
    else
      sub_C33EB0(&v152, (__int64 *)&v146);
    sub_3329C90(&v150, (__int64 *)&v152, v125, 1);
    if ( v152 == v139 )
    {
      if ( v153 )
      {
        v99 = &v153[3 * (_QWORD)*(v153 - 1)];
        if ( v153 != v99 )
        {
          v100 = &v153[3 * (_QWORD)*(v153 - 1)];
          do
          {
            v100 -= 3;
            if ( v139 == *v100 )
              sub_969EE0((__int64)v100);
            else
              sub_C338F0((__int64)v100);
          }
          while ( v153 != v100 );
          v99 = v100;
        }
        j_j_j___libc_free_0_0((unsigned __int64)(v99 - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v152);
    }
    v25 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)&v26 = sub_33ED040(v25, 18);
    *(_QWORD *)&v28 = sub_340F900(v25, 208, (unsigned int)&v142, v137, v127, v27, v140, v122, v26);
    v121 = v28;
    *(_QWORD *)&v29 = sub_3405C90(*(_QWORD *)(a1 + 16), 57, (unsigned int)&v142, v11, v10, 2, v140, v122);
    v110 = v29;
    *(_QWORD *)&v30 = sub_3400BD0(*(_QWORD *)(a1 + 16), 3 * v133, (unsigned int)&v142, v11, v10, 0, 0);
    *(_QWORD *)&v32 = sub_3406EB0(*(_QWORD *)(a1 + 16), 180, (unsigned int)&v142, v11, v10, v31, v140, v30);
    *(_QWORD *)&v33 = sub_3405C90(*(_QWORD *)(a1 + 16), 57, (unsigned int)&v142, v11, v10, 2, v32, v115);
    v34 = *(_QWORD *)(a1 + 16);
    v108 = v33;
    *(_QWORD *)&v35 = sub_33ED040(v34, 10);
    *(_QWORD *)&v37 = sub_340F900(v34, 208, (unsigned int)&v142, v137, v127, v36, v140, v115, v35);
    v134 = v37;
    *(_QWORD *)&v38 = sub_33FE6E0(*(_QWORD *)(a1 + 16), &v148, &v142, v144, v145, 0);
    v116 = v38;
    *(_QWORD *)&v40 = sub_3406EB0(*(_QWORD *)(a1 + 16), 98, (unsigned int)&v142, v144, v145, v39, v8, v38);
    v101 = v116;
    v117 = v40;
    *(_QWORD *)&v42 = sub_3406EB0(*(_QWORD *)(a1 + 16), 98, (unsigned int)&v142, v144, v145, v41, v40, v101);
    v106 = v42;
    *(_QWORD *)&v44 = sub_340F900(*(_QWORD *)(a1 + 16), 205, (unsigned int)&v142, v11, v10, v43, v134, v108, v110);
    v111 = v44;
    *(_QWORD *)&v46 = sub_340F900(*(_QWORD *)(a1 + 16), 205, (unsigned int)&v142, v144, v145, v45, v134, v106, v117);
    v47 = *(_QWORD *)(a1 + 16);
    v109 = v46;
    *(_QWORD *)&v48 = sub_33ED040(v47, 20);
    *(_QWORD *)&v50 = sub_340F900(v47, 208, (unsigned int)&v142, v137, v127, v49, v140, v112, v48);
    v135 = v50;
    *(_QWORD *)&v51 = sub_3400BD0(*(_QWORD *)(a1 + 16), -v125, (unsigned int)&v142, v11, v10, 0, 0);
    LODWORD(v102) = 0;
    v113 = v51;
    *(_QWORD *)&v52 = sub_3400BD0(*(_QWORD *)(a1 + 16), -2 * v125, (unsigned int)&v142, v11, v10, 0, v102);
    v118 = v52;
    *(_QWORD *)&v53 = sub_3405C90(*(_QWORD *)(a1 + 16), 56, (unsigned int)&v142, v11, v10, 3, v140, v113);
    v114 = v53;
    *(_QWORD *)&v54 = sub_3401400(*(_QWORD *)(a1 + 16), 3 * v129 + 2 * v124, (unsigned int)&v142, v11, v10, 0, 0);
    *(_QWORD *)&v56 = sub_3406EB0(*(_QWORD *)(a1 + 16), 181, (unsigned int)&v142, v11, v10, v55, v140, v54);
    *(_QWORD *)&v57 = sub_3405C90(*(_QWORD *)(a1 + 16), 56, (unsigned int)&v142, v11, v10, 2, v56, v118);
    v107 = v57;
    *(_QWORD *)&v58 = sub_33FE6E0(*(_QWORD *)(a1 + 16), &v150, &v142, v144, v145, 0);
    v119 = v58;
    *(_QWORD *)&v60 = sub_3406EB0(*(_QWORD *)(a1 + 16), 98, (unsigned int)&v142, v144, v145, v59, v8, v58);
    v103 = v119;
    v120 = v60;
    *(_QWORD *)&v62 = sub_3406EB0(*(_QWORD *)(a1 + 16), 98, (unsigned int)&v142, v144, v145, v61, v60, v103);
    v63 = *(_QWORD *)(a1 + 16);
    v105 = v62;
    *(_QWORD *)&v130 = sub_3401400(v63, v129 + v125, (unsigned int)&v142, v11, v10, 0, 0);
    *((_QWORD *)&v130 + 1) = v64;
    *(_QWORD *)&v65 = sub_33ED040(v63, 12);
    *(_QWORD *)&v66 = sub_340F900(v63, 208, (unsigned int)&v142, v137, v127, DWORD2(v130), v140, v130, v65);
    v128 = v66;
    *(_QWORD *)&v68 = sub_340F900(*(_QWORD *)(a1 + 16), 205, (unsigned int)&v142, v11, v10, v67, v66, v107, v114);
    v131 = v68;
    *(_QWORD *)&v70 = sub_340F900(*(_QWORD *)(a1 + 16), 205, (unsigned int)&v142, v144, v145, v69, v128, v105, v120);
    v71 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)&v73 = sub_340F900(v71, 205, (unsigned int)&v142, v144, v145, v72, v135, v70, v8);
    *(_QWORD *)&v75 = sub_340F900(v71, 205, (unsigned int)&v142, v144, v145, v74, v121, v109, v73);
    v76 = *(_QWORD *)(a1 + 16);
    v136 = v75;
    *(_QWORD *)&v78 = sub_340F900(v76, 205, (unsigned int)&v142, v11, v10, v77, v135, v131, v140);
    *(_QWORD *)&v80 = sub_340F900(v76, 205, (unsigned int)&v142, v11, v10, v79, v121, v111, v78);
    v81 = sub_3405C90(*(_QWORD *)(a1 + 16), 56, (unsigned int)&v142, v11, v10, 2, v80, v122);
    v83 = v82;
    v141 = v81;
    v84 = sub_3400E40(*(_QWORD *)(a1 + 16), v124 - 1, v11, v10, &v142);
    v86 = v85;
    v87 = v84;
    *(_QWORD *)&v88 = sub_33FB310(*(_QWORD *)(a1 + 16), v141, v83, &v142, v126, v123);
    *((_QWORD *)&v104 + 1) = v86;
    *(_QWORD *)&v104 = v87;
    *(_QWORD *)&v89 = sub_3405C90(*(_QWORD *)(a1 + 16), 190, (unsigned int)&v142, v126, v123, 3, v88, v104);
    *(_QWORD *)&v91 = sub_33FAF80(*(_QWORD *)(a1 + 16), 234, (unsigned int)&v142, v144, v145, v90, v89);
    v14 = sub_3406EB0(*(_QWORD *)(a1 + 16), 98, (unsigned int)&v142, v144, v145, v92, v136, v91);
    if ( v150 == v139 )
    {
      if ( v151 )
      {
        v95 = &v151[3 * (_QWORD)*(v151 - 1)];
        while ( v151 != v95 )
        {
          v95 -= 3;
          if ( v139 == *v95 )
            sub_969EE0((__int64)v95);
          else
            sub_C338F0((__int64)v95);
        }
        j_j_j___libc_free_0_0((unsigned __int64)(v95 - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v150);
    }
    if ( v148 == v139 )
    {
      if ( v149 )
      {
        v94 = &v149[3 * (_QWORD)*(v149 - 1)];
        while ( v149 != v94 )
        {
          v94 -= 3;
          if ( v139 == *v94 )
            sub_969EE0((__int64)v94);
          else
            sub_C338F0((__int64)v94);
        }
        j_j_j___libc_free_0_0((unsigned __int64)(v94 - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v148);
    }
    if ( v146 == v139 )
    {
      if ( v147 )
      {
        v93 = &v147[3 * (_QWORD)*(v147 - 1)];
        while ( v147 != v93 )
        {
          v93 -= 3;
          if ( v139 == *v93 )
            sub_969EE0((__int64)v93);
          else
            sub_C338F0((__int64)v93);
        }
        j_j_j___libc_free_0_0((unsigned __int64)(v93 - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v146);
    }
  }
  else
  {
    v14 = 0;
  }
  if ( v142 )
    sub_B91220((__int64)&v142, v142);
  return v14;
}
