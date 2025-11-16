// Function: sub_32CEA50
// Address: 0x32cea50
//
__int64 __fastcall sub_32CEA50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // r13
  __int16 *v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 (__fastcall *v14)(__int64, __int64, __int64, __int64, __int64); // r15
  __int64 v15; // rsi
  int v16; // eax
  unsigned __int16 v17; // r13
  int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r9
  __int64 v22; // r10
  char v23; // r13
  __int64 v24; // r13
  __int64 v25; // r12
  __int64 v27; // rdi
  __int64 (*v28)(); // r8
  __int64 v29; // rcx
  __int64 v30; // rdi
  __int64 v31; // r8
  __int64 v32; // rdx
  __int64 v33; // r15
  _QWORD *v34; // r13
  __int64 v35; // r12
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // r13d
  __int64 v45; // rdx
  int v46; // eax
  bool v47; // al
  __int64 v48; // r12
  unsigned __int16 *v49; // rax
  __int64 v50; // r13
  unsigned int v51; // r14d
  __int64 v52; // rax
  unsigned int v53; // eax
  __int64 v54; // rdx
  __int128 v55; // rax
  int v56; // r9d
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r14
  __int64 v60; // r13
  unsigned int v61; // edx
  unsigned __int64 v62; // r14
  int v63; // r9d
  __int128 v64; // rax
  int v65; // eax
  __int128 v66; // rax
  int v67; // r9d
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r10
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r10
  __int128 v79; // rax
  __int64 v80; // r8
  __int64 v81; // r9
  __int128 v82; // rax
  __int128 v83; // rax
  __int64 v84; // r13
  __int128 v85; // rax
  int v86; // r9d
  __int128 v87; // rax
  __int64 v88; // r13
  __int128 v89; // rax
  int v90; // r9d
  __int128 v91; // rax
  int v92; // r9d
  __int64 v93; // rax
  __int64 v94; // r13
  int v95; // ecx
  __int64 v96; // rsi
  __int64 v97; // r10
  __int64 v98; // rdx
  __int64 v99; // r11
  __int64 v100; // rax
  int v101; // r14d
  __int16 v102; // dx
  __int64 v103; // rax
  int v104; // esi
  unsigned int v105; // edx
  __int128 v106; // rax
  int v107; // r9d
  __int128 v108; // rax
  __int64 v109; // r14
  __int128 v110; // rax
  int v111; // r9d
  __int64 v112; // rax
  __int64 v113; // r14
  __int64 v114; // rcx
  __int64 v115; // r10
  unsigned int v116; // edx
  __int64 v117; // r13
  __int64 v118; // rax
  int v119; // r8d
  __int16 v120; // dx
  __int64 v121; // rax
  __int64 v122; // r11
  int v123; // ebx
  int v124; // esi
  __int64 v125; // rdi
  __int64 v126; // rcx
  _QWORD *v127; // r13
  __int64 v128; // r12
  _QWORD *v129; // r13
  __int64 v130; // rbx
  __int64 v131; // r15
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 v134; // rax
  bool v135; // al
  bool v136; // al
  __int64 v137; // rax
  __int64 v138; // rax
  __int128 v139; // [rsp-20h] [rbp-190h]
  __int128 v140; // [rsp-20h] [rbp-190h]
  __int128 v141; // [rsp-10h] [rbp-180h]
  __int128 v142; // [rsp-10h] [rbp-180h]
  __int128 v143; // [rsp-10h] [rbp-180h]
  __int64 v144; // [rsp+8h] [rbp-168h]
  __int64 v145; // [rsp+10h] [rbp-160h]
  __int64 v146; // [rsp+10h] [rbp-160h]
  __int128 v147; // [rsp+10h] [rbp-160h]
  unsigned int v148; // [rsp+20h] [rbp-150h]
  __int128 v149; // [rsp+20h] [rbp-150h]
  int v150; // [rsp+20h] [rbp-150h]
  __int128 v151; // [rsp+20h] [rbp-150h]
  int v152; // [rsp+20h] [rbp-150h]
  unsigned int v153; // [rsp+30h] [rbp-140h]
  __int64 v154; // [rsp+30h] [rbp-140h]
  __int64 v155; // [rsp+30h] [rbp-140h]
  __int128 v156; // [rsp+30h] [rbp-140h]
  __int128 v157; // [rsp+30h] [rbp-140h]
  __int128 v158; // [rsp+30h] [rbp-140h]
  __int64 v159; // [rsp+38h] [rbp-138h]
  __int64 v160; // [rsp+38h] [rbp-138h]
  __int64 v161; // [rsp+38h] [rbp-138h]
  int v162; // [rsp+40h] [rbp-130h]
  __int64 v163; // [rsp+48h] [rbp-128h]
  int v164; // [rsp+48h] [rbp-128h]
  int v165; // [rsp+48h] [rbp-128h]
  __int64 v166; // [rsp+50h] [rbp-120h]
  __int64 v167; // [rsp+50h] [rbp-120h]
  __int64 v168; // [rsp+50h] [rbp-120h]
  __int128 v169; // [rsp+50h] [rbp-120h]
  __int64 v170; // [rsp+50h] [rbp-120h]
  __int64 v171; // [rsp+50h] [rbp-120h]
  __int128 v172; // [rsp+50h] [rbp-120h]
  __int128 v174; // [rsp+60h] [rbp-110h]
  __int128 v175; // [rsp+60h] [rbp-110h]
  __int128 v176; // [rsp+70h] [rbp-100h]
  __int64 v177; // [rsp+70h] [rbp-100h]
  _QWORD *v178; // [rsp+70h] [rbp-100h]
  __int64 v179; // [rsp+70h] [rbp-100h]
  __int64 v180; // [rsp+A0h] [rbp-D0h] BYREF
  int v181; // [rsp+A8h] [rbp-C8h]
  __int64 v182; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v183; // [rsp+B8h] [rbp-B8h]
  __int64 v184; // [rsp+C0h] [rbp-B0h]
  __int64 v185; // [rsp+C8h] [rbp-A8h]
  __int64 v186[2]; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 (__fastcall *v187)(__int64 *, __int64, int); // [rsp+E0h] [rbp-90h]
  bool (__fastcall *v188)(__int64, __int64); // [rsp+E8h] [rbp-88h]
  _QWORD *v189; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v190; // [rsp+F8h] [rbp-78h]
  _QWORD v191[14]; // [rsp+100h] [rbp-70h] BYREF

  *(_QWORD *)&v174 = a2;
  v8 = *(_QWORD *)(a6 + 80);
  *((_QWORD *)&v174 + 1) = a3;
  *(_QWORD *)&v176 = a4;
  *((_QWORD *)&v176 + 1) = a5;
  v148 = a3;
  v180 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v180, v8, 1);
  v9 = a1[1];
  v181 = *(_DWORD *)(a6 + 72);
  v10 = *(__int16 **)(a6 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v182) = v11;
  v13 = v182;
  v183 = v12;
  v14 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 528LL);
  v166 = v12;
  v163 = *(_QWORD *)(*a1 + 64LL);
  v15 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
  v16 = v14(v9, v15, v163, v13, v166);
  v17 = v182;
  v162 = v18;
  v164 = v16;
  if ( (_WORD)v182 )
  {
    if ( (unsigned __int16)(v182 - 17) <= 0xD3u )
    {
      v190 = 0;
      v17 = word_4456580[(unsigned __int16)v182 - 1];
      LOWORD(v189) = v17;
      if ( !v17 )
        goto LABEL_7;
      goto LABEL_31;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v182) )
  {
LABEL_5:
    v19 = v183;
    goto LABEL_6;
  }
  v17 = sub_3009970((__int64)&v182, v15, v39, v40, v41);
LABEL_6:
  LOWORD(v189) = v17;
  v190 = v19;
  if ( !v17 )
  {
LABEL_7:
    v184 = sub_3007260((__int64)&v189);
    v185 = v20;
    LODWORD(v167) = v184;
    goto LABEL_8;
  }
LABEL_31:
  if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
    BUG();
  v167 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
LABEL_8:
  if ( (*(_BYTE *)(a6 + 28) & 4) != 0 )
    goto LABEL_14;
  v188 = sub_3260EB0;
  v187 = sub_325D4D0;
  v191[0] = 0;
  sub_325D4D0(&v189, (__int64)v186, 2);
  v191[1] = v188;
  v191[0] = v187;
  v23 = sub_33CA8D0(v22, v21, &v189);
  if ( v191[0] )
    ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v191[0])(&v189, &v189, 3);
  if ( v187 )
    v187(v186, (__int64)v186, 3);
  if ( !v23 )
  {
LABEL_14:
    v24 = *(_QWORD *)(**(_QWORD **)(*a1 + 40LL) + 120LL);
    if ( (unsigned __int8)sub_326A930(v176, DWORD2(v176), 0) )
    {
      v27 = a1[1];
      v28 = *(__int64 (**)())(*(_QWORD *)v27 + 200LL);
      if ( (v28 == sub_2FE2F30
         || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v28)(
               v27,
               **(unsigned __int16 **)(a6 + 48),
               *(_QWORD *)(*(_QWORD *)(a6 + 48) + 8LL),
               v24))
        && !(unsigned __int8)sub_B2D610(**(_QWORD **)(*a1 + 40LL), 18) )
      {
        v29 = *((unsigned __int8 *)a1 + 33);
        v30 = a1[1];
        v31 = *((unsigned __int8 *)a1 + 34);
        v32 = *a1;
        v189 = v191;
        v190 = 0x800000000LL;
        v177 = sub_344B630(v30, a6, v32, v29, v31, &v189);
        if ( v177 )
        {
          v33 = (__int64)v189;
          v34 = &v189[(unsigned int)v190];
          if ( v189 != v34 )
          {
            do
            {
              v35 = *(_QWORD *)v33;
              if ( *(_DWORD *)(*(_QWORD *)v33 + 24LL) != 328 )
              {
                v186[0] = *(_QWORD *)v33;
                sub_32B3B20((__int64)(a1 + 71), v186);
                if ( *(int *)(v35 + 88) < 0 )
                {
                  *(_DWORD *)(v35 + 88) = *((_DWORD *)a1 + 12);
                  v38 = *((unsigned int *)a1 + 12);
                  if ( v38 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
                  {
                    sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v38 + 1, 8u, v36, v37);
                    v38 = *((unsigned int *)a1 + 12);
                  }
                  *(_QWORD *)(a1[5] + 8 * v38) = v35;
                  ++*((_DWORD *)a1 + 12);
                }
              }
              v33 += 8;
            }
            while ( v34 != (_QWORD *)v33 );
            v34 = v189;
          }
          if ( v34 != v191 )
            _libc_free((unsigned __int64)v34);
          v25 = v177;
          goto LABEL_16;
        }
        if ( v189 != v191 )
          _libc_free((unsigned __int64)v189);
      }
    }
LABEL_15:
    v25 = 0;
    goto LABEL_16;
  }
  v42 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(a6 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a6 + 40) + 48LL), 0, 0);
  if ( v42 )
  {
    v43 = *(_QWORD *)(v42 + 96);
    v44 = *(_DWORD *)(v43 + 32);
    v45 = v43 + 24;
    if ( v44 <= 0x40 )
    {
      v47 = *(_QWORD *)(v43 + 24) == 0;
    }
    else
    {
      v145 = v43 + 24;
      v46 = sub_C444A0(v43 + 24);
      v45 = v145;
      v47 = v44 == v46;
    }
    if ( !v47 )
    {
      v125 = a1[1];
      v126 = *a1;
      v189 = v191;
      v190 = 0x800000000LL;
      v144 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD **))(*(_QWORD *)v125 + 2544LL))(
               v125,
               a6,
               v45,
               v126,
               &v189);
      if ( v144 )
      {
        v127 = &v189[(unsigned int)v190];
        if ( v189 != v127 )
        {
          v178 = &v189[(unsigned int)v190];
          v128 = (__int64)v189;
          v129 = a1;
          v130 = (__int64)(a1 + 71);
          do
          {
            v131 = *(_QWORD *)v128;
            if ( *(_DWORD *)(*(_QWORD *)v128 + 24LL) != 328 )
            {
              v186[0] = *(_QWORD *)v128;
              sub_32B3B20(v130, v186);
              if ( *(int *)(v131 + 88) < 0 )
              {
                *(_DWORD *)(v131 + 88) = *((_DWORD *)v129 + 12);
                v134 = *((unsigned int *)v129 + 12);
                if ( v134 + 1 > (unsigned __int64)*((unsigned int *)v129 + 13) )
                {
                  sub_C8D5F0((__int64)(v129 + 5), v129 + 7, v134 + 1, 8u, v132, v133);
                  v134 = *((unsigned int *)v129 + 12);
                }
                *(_QWORD *)(v129[5] + 8 * v134) = v131;
                ++*((_DWORD *)v129 + 12);
              }
            }
            v128 += 8;
          }
          while ( v178 != (_QWORD *)v128 );
          v127 = v189;
        }
        if ( v127 != v191 )
          _libc_free((unsigned __int64)v127);
        v25 = v144;
        goto LABEL_16;
      }
      if ( v189 != v191 )
        _libc_free((unsigned __int64)v189);
    }
  }
  v48 = a1[1];
  v49 = (unsigned __int16 *)(*(_QWORD *)(v174 + 48) + 16LL * v148);
  v50 = *((_QWORD *)v49 + 1);
  v51 = *v49;
  v52 = sub_2E79000(*(__int64 **)(*a1 + 40LL));
  v53 = sub_2FE6750(v48, v51, v50, v52);
  v146 = v54;
  v153 = v53;
  *(_QWORD *)&v55 = sub_3400BD0(*a1, v167, (unsigned int)&v180, v53, v54, 0, 0);
  v149 = v55;
  v57 = sub_33FAF80(*a1, 198, (unsigned int)&v180, v182, v183, v56, v176);
  v59 = v58;
  v60 = sub_33FB310(*a1, v57, v58, &v180, v153, v146);
  v62 = v61 | v59 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v141 + 1) = v62;
  *(_QWORD *)&v141 = v60;
  v139 = v149;
  v150 = v146;
  *(_QWORD *)&v64 = sub_3406EB0(*a1, 57, (unsigned int)&v180, v153, v146, v63, v139, v141);
  v147 = v64;
  if ( !(unsigned __int8)sub_326A930(v64, DWORD2(v64), 0) )
    goto LABEL_15;
  v65 = v167;
  v168 = *a1;
  *(_QWORD *)&v66 = sub_3400BD0(*a1, v65 - 1, (unsigned int)&v180, v153, v150, 0, 0);
  *(_QWORD *)&v169 = sub_3406EB0(v168, 191, (unsigned int)&v180, v182, v183, v67, v174, v66);
  *((_QWORD *)&v169 + 1) = v68;
  sub_32B3E80((__int64)a1, v169, 1, 0, v169, v68);
  v69 = sub_3406EB0(*a1, 192, (unsigned int)&v180, v182, v183, DWORD2(v169), v169, v147);
  v170 = v69;
  v71 = v69;
  v72 = v70;
  if ( *(_DWORD *)(v69 + 24) != 328 )
  {
    v189 = (_QWORD *)v69;
    v154 = v69;
    v159 = v70;
    sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v189);
    v73 = v170;
    v71 = v154;
    v72 = v159;
    if ( *(int *)(v170 + 88) < 0 )
    {
      *(_DWORD *)(v170 + 88) = *((_DWORD *)a1 + 12);
      v138 = *((unsigned int *)a1 + 12);
      if ( v138 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v138 + 1, 8u, v154, v159);
        v138 = *((unsigned int *)a1 + 12);
        v71 = v154;
        v72 = v159;
        v73 = v170;
      }
      *(_QWORD *)(a1[5] + 8 * v138) = v73;
      ++*((_DWORD *)a1 + 12);
    }
  }
  *((_QWORD *)&v142 + 1) = v72;
  *(_QWORD *)&v142 = v71;
  v74 = sub_3406EB0(*a1, 56, (unsigned int)&v180, v182, v183, v72, v174, v142);
  v171 = v74;
  v76 = v74;
  v77 = v75;
  if ( *(_DWORD *)(v74 + 24) != 328 )
  {
    v189 = (_QWORD *)v74;
    v160 = v75;
    v155 = v74;
    sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v189);
    v78 = v171;
    v76 = v155;
    v77 = v160;
    if ( *(int *)(v171 + 88) < 0 )
    {
      *(_DWORD *)(v171 + 88) = *((_DWORD *)a1 + 12);
      v137 = *((unsigned int *)a1 + 12);
      if ( v137 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v137 + 1, 8u, v155, v160);
        v137 = *((unsigned int *)a1 + 12);
        v76 = v155;
        v77 = v160;
        v78 = v171;
      }
      *(_QWORD *)(a1[5] + 8 * v137) = v78;
      ++*((_DWORD *)a1 + 12);
    }
  }
  *((_QWORD *)&v143 + 1) = v62;
  *(_QWORD *)&v143 = v60;
  *((_QWORD *)&v140 + 1) = v77;
  *(_QWORD *)&v140 = v76;
  *(_QWORD *)&v79 = sub_3406EB0(*a1, 191, (unsigned int)&v180, v182, v183, v77, v140, v143);
  v172 = v79;
  sub_32B3E80((__int64)a1, v79, 1, 0, v80, v81);
  *(_QWORD *)&v82 = sub_3400BD0(*a1, 1, (unsigned int)&v180, v182, v183, 0, 0);
  v156 = v82;
  *(_QWORD *)&v83 = sub_34015B0(*a1, &v180, (unsigned int)v182, v183, 0, 0);
  v84 = *a1;
  v151 = v83;
  *(_QWORD *)&v85 = sub_33ED040(*a1, 17);
  *(_QWORD *)&v87 = sub_340F900(v84, 208, (unsigned int)&v180, v164, v162, v86, v176, v156, v85);
  v88 = *a1;
  v157 = v87;
  *(_QWORD *)&v89 = sub_33ED040(*a1, 17);
  *(_QWORD *)&v91 = sub_340F900(v88, 208, (unsigned int)&v180, v164, v162, v90, v176, v151, v89);
  v93 = sub_3406EB0(*a1, 187, (unsigned int)&v180, v164, v162, v92, v157, v91);
  v94 = *a1;
  v95 = v182;
  v96 = v93;
  v97 = v93;
  v99 = v98;
  v100 = *(_QWORD *)(v93 + 48) + 16LL * (unsigned int)v98;
  v101 = v183;
  v102 = *(_WORD *)v100;
  v103 = *(_QWORD *)(v100 + 8);
  LOWORD(v189) = v102;
  v190 = v103;
  if ( v102 )
  {
    v104 = ((unsigned __int16)(v102 - 17) < 0xD4u) + 205;
  }
  else
  {
    v152 = v182;
    v161 = v99;
    v136 = sub_30070B0((__int64)&v189);
    v95 = v152;
    v97 = v96;
    v99 = v161;
    v104 = 205 - (!v136 - 1);
  }
  *(_QWORD *)&v172 = sub_340EC60(v94, v104, (unsigned int)&v180, v95, v101, 0, v97, v99, v174, v172);
  *((_QWORD *)&v172 + 1) = v105 | *((_QWORD *)&v172 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v106 = sub_3400BD0(*a1, 0, (unsigned int)&v180, v182, v183, 0, 0);
  v158 = v106;
  *(_QWORD *)&v108 = sub_3406EB0(*a1, 57, (unsigned int)&v180, v182, v183, v107, v106, v172);
  v109 = *a1;
  v175 = v108;
  *(_QWORD *)&v110 = sub_33ED040(*a1, 20);
  v112 = sub_340F900(v109, 208, (unsigned int)&v180, v164, v162, v111, v176, v158, v110);
  v113 = *a1;
  v114 = v112;
  v115 = v112;
  v117 = v116;
  v118 = *(_QWORD *)(v112 + 48) + 16LL * v116;
  v119 = v182;
  v120 = *(_WORD *)v118;
  v121 = *(_QWORD *)(v118 + 8);
  v122 = v117;
  v123 = v183;
  LOWORD(v189) = v120;
  v190 = v121;
  if ( v120 )
  {
    v124 = ((unsigned __int16)(v120 - 17) < 0xD4u) + 205;
  }
  else
  {
    v165 = v182;
    v179 = v114;
    v135 = sub_30070B0((__int64)&v189);
    v119 = v165;
    v115 = v179;
    v122 = v117;
    v124 = 205 - (!v135 - 1);
  }
  v25 = sub_340EC60(v113, v124, (unsigned int)&v180, v119, v123, 0, v115, v122, v175, v172);
LABEL_16:
  if ( v180 )
    sub_B91220((__int64)&v180, v180);
  return v25;
}
