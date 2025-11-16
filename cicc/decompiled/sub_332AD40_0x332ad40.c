// Function: sub_332AD40
// Address: 0x332ad40
//
__int64 __fastcall sub_332AD40(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // rax
  __int16 v6; // dx
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r13
  int v12; // eax
  unsigned __int16 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rdx
  _DWORD *v20; // rax
  __int64 v21; // r12
  _DWORD *v22; // r15
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rdx
  unsigned int v28; // eax
  char v29; // r9
  __int64 v30; // r8
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rcx
  __int64 v33; // r8
  unsigned __int64 v34; // rax
  __int128 v35; // rax
  __int128 v36; // rax
  __int128 v37; // rax
  int v38; // r9d
  __int128 v39; // rax
  __int64 v40; // r12
  __int64 (__fastcall *v41)(__int64, __int64, __int64, _QWORD, _QWORD *); // r13
  __int64 v42; // rax
  int v43; // eax
  int v44; // edx
  int v45; // r9d
  __int128 v46; // rax
  int v47; // r9d
  __int128 v48; // rax
  int v49; // r9d
  __int64 v50; // rdx
  __int128 v51; // rax
  __int64 v52; // rdx
  __int128 v53; // rax
  int v54; // r9d
  __int128 v55; // rax
  __int128 v56; // rax
  __int128 v57; // rax
  int v58; // r9d
  __int128 v59; // rax
  __int64 v60; // r13
  __int64 v61; // r12
  int v62; // r9d
  __int128 v63; // rax
  int v64; // r9d
  __int128 v65; // rax
  int v66; // r9d
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // r13
  __int64 v70; // r12
  __int128 v71; // rax
  int v72; // r9d
  __int64 v73; // rax
  __int64 v74; // rdx
  __int128 v75; // rax
  int v76; // r9d
  __int128 v77; // rax
  __int128 v78; // rax
  int v79; // r9d
  __int128 v80; // rax
  int v81; // r9d
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r13
  __int64 v85; // r12
  __int128 v86; // rax
  int v87; // r9d
  __int128 v88; // rax
  int v89; // r9d
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r13
  __int64 v93; // r12
  int v94; // r9d
  __int128 v95; // rax
  int v96; // r9d
  __int64 v97; // rax
  _QWORD *v98; // rdx
  _QWORD *v99; // r13
  _DWORD *v100; // r12
  int v101; // r9d
  __int64 v102; // rax
  __int64 v103; // rdi
  __int64 v104; // rdx
  _QWORD *ii; // rbx
  _QWORD *j; // r12
  _QWORD *i; // r12
  _QWORD *n; // r14
  _QWORD *m; // r14
  _QWORD *k; // r13
  _QWORD *v111; // r12
  __int128 v112; // [rsp-10h] [rbp-240h]
  __int128 v113; // [rsp-10h] [rbp-240h]
  __int128 v114; // [rsp-10h] [rbp-240h]
  __int64 v115; // [rsp+0h] [rbp-230h]
  __int128 v116; // [rsp+0h] [rbp-230h]
  __int128 v117; // [rsp+10h] [rbp-220h]
  __int128 v118; // [rsp+20h] [rbp-210h]
  __int128 v119; // [rsp+30h] [rbp-200h]
  __int128 v120; // [rsp+30h] [rbp-200h]
  __int128 v121; // [rsp+40h] [rbp-1F0h]
  __int128 v122; // [rsp+50h] [rbp-1E0h]
  __int128 v123; // [rsp+60h] [rbp-1D0h]
  __int128 v124; // [rsp+60h] [rbp-1D0h]
  char v125; // [rsp+78h] [rbp-1B8h]
  int v126; // [rsp+80h] [rbp-1B0h]
  __int128 v127; // [rsp+80h] [rbp-1B0h]
  __int64 v128; // [rsp+90h] [rbp-1A0h]
  __int64 v129; // [rsp+90h] [rbp-1A0h]
  __int128 v130; // [rsp+90h] [rbp-1A0h]
  __int64 v131; // [rsp+90h] [rbp-1A0h]
  __int64 v132; // [rsp+90h] [rbp-1A0h]
  __int128 v133; // [rsp+A0h] [rbp-190h]
  __int128 v134; // [rsp+A0h] [rbp-190h]
  __int128 v135; // [rsp+A0h] [rbp-190h]
  unsigned int v136; // [rsp+A0h] [rbp-190h]
  unsigned int v137; // [rsp+A0h] [rbp-190h]
  __int128 v138; // [rsp+B0h] [rbp-180h]
  __int64 v139; // [rsp+B0h] [rbp-180h]
  __int128 v140; // [rsp+B0h] [rbp-180h]
  __int128 v141; // [rsp+C0h] [rbp-170h]
  int v142; // [rsp+D0h] [rbp-160h]
  __int128 v143; // [rsp+D0h] [rbp-160h]
  _DWORD *v144; // [rsp+E0h] [rbp-150h]
  __int64 v145; // [rsp+E0h] [rbp-150h]
  __int128 v146; // [rsp+E0h] [rbp-150h]
  unsigned int v147; // [rsp+F4h] [rbp-13Ch]
  unsigned int v148; // [rsp+100h] [rbp-130h]
  __int64 v149; // [rsp+108h] [rbp-128h]
  unsigned int v150; // [rsp+118h] [rbp-118h]
  __int64 v151; // [rsp+120h] [rbp-110h]
  __int128 v152; // [rsp+120h] [rbp-110h]
  __int64 v153; // [rsp+130h] [rbp-100h] BYREF
  int v154; // [rsp+138h] [rbp-F8h]
  unsigned int v155; // [rsp+140h] [rbp-F0h] BYREF
  _QWORD *v156; // [rsp+148h] [rbp-E8h]
  unsigned __int64 v157; // [rsp+150h] [rbp-E0h] BYREF
  unsigned int v158; // [rsp+158h] [rbp-D8h]
  unsigned __int64 v159; // [rsp+160h] [rbp-D0h] BYREF
  unsigned int v160; // [rsp+168h] [rbp-C8h]
  __int64 v161; // [rsp+170h] [rbp-C0h]
  __int64 v162; // [rsp+178h] [rbp-B8h]
  _DWORD *v163; // [rsp+180h] [rbp-B0h] BYREF
  _QWORD *v164; // [rsp+188h] [rbp-A8h]
  _DWORD *v165; // [rsp+1A0h] [rbp-90h] BYREF
  _QWORD *v166; // [rsp+1A8h] [rbp-88h]
  _DWORD *v167; // [rsp+1C0h] [rbp-70h] BYREF
  _QWORD *v168; // [rsp+1C8h] [rbp-68h]
  _DWORD *v169; // [rsp+1E0h] [rbp-50h] BYREF
  _QWORD *v170; // [rsp+1E8h] [rbp-48h]
  __int64 v171; // [rsp+1F0h] [rbp-40h]
  __int64 v172; // [rsp+1F8h] [rbp-38h]

  v3 = a1;
  v4 = *(_QWORD *)(a2 + 80);
  v153 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v153, v4, 1);
  v154 = *(_DWORD *)(a2 + 72);
  v141 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v5 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v6 = *(_WORD *)v5;
  v156 = *(_QWORD **)(v5 + 8);
  v7 = *(_QWORD *)(a2 + 48);
  LOWORD(v155) = v6;
  v149 = *(_QWORD *)(v7 + 24);
  v148 = *(unsigned __int16 *)(v7 + 16);
  v8 = sub_327FF20((unsigned __int16 *)&v155, v4);
  v151 = v9;
  v150 = v8;
  if ( !(_WORD)v8 && !v9 )
  {
    v10 = 0;
    goto LABEL_6;
  }
  v144 = sub_300AC80((unsigned __int16 *)&v155, v4);
  v126 = sub_C336C0((__int64)v144);
  v12 = sub_C336A0((__int64)v144);
  v13 = v155;
  v142 = v12;
  if ( (_WORD)v155 )
  {
    if ( (unsigned __int16)(v155 - 17) <= 0xD3u )
    {
      v170 = 0;
      v13 = word_4456580[(unsigned __int16)v155 - 1];
      LOWORD(v169) = v13;
      if ( !v13 )
        goto LABEL_13;
      goto LABEL_81;
    }
    goto LABEL_11;
  }
  if ( !sub_30070B0((__int64)&v155) )
  {
LABEL_11:
    v17 = v156;
    goto LABEL_12;
  }
  v13 = sub_3009970((__int64)&v155, v4, v14, v15, v16);
LABEL_12:
  LOWORD(v169) = v13;
  v170 = v17;
  if ( !v13 )
  {
LABEL_13:
    v161 = sub_3007260((__int64)&v169);
    LODWORD(v18) = v161;
    v162 = v19;
    goto LABEL_14;
  }
LABEL_81:
  if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
    BUG();
  v18 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
LABEL_14:
  v20 = sub_C33340();
  v21 = *(_QWORD *)(a1 + 16);
  v22 = v20;
  if ( v144 == v20 )
    sub_C3C500(&v169, (__int64)v144);
  else
    sub_C373C0(&v169, (__int64)v144);
  if ( v169 == v22 )
    sub_C3D240((__int64)&v169, 1);
  else
    sub_C35A40((__int64)&v169, 1);
  if ( v169 == v22 )
    sub_C3E660((__int64)&v167, (__int64)&v169);
  else
    sub_C3A850((__int64)&v167, (__int64 *)&v169);
  *(_QWORD *)&v138 = sub_34007B0(v21, (unsigned int)&v167, (unsigned int)&v153, v150, v151, 0, 0);
  *((_QWORD *)&v138 + 1) = v23;
  if ( (unsigned int)v168 > 0x40 && v167 )
    j_j___libc_free_0_0((unsigned __int64)v167);
  if ( v169 == v22 )
  {
    if ( v170 )
    {
      v111 = &v170[3 * *(v170 - 1)];
      if ( v170 != v111 )
      {
        do
        {
          while ( 1 )
          {
            v111 -= 3;
            if ( v22 == (_DWORD *)*v111 )
              break;
            sub_C338F0((__int64)v111);
            if ( v170 == v111 )
              goto LABEL_149;
          }
          sub_969EE0((__int64)v111);
        }
        while ( v170 != v111 );
LABEL_149:
        v3 = a1;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v111 - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v169);
  }
  v24 = *(_QWORD *)(v3 + 16);
  if ( v144 == v22 )
    sub_C3C500(&v169, (__int64)v144);
  else
    sub_C373C0(&v169, (__int64)v144);
  if ( v22 == v169 )
    sub_C3D240((__int64)&v169, 0);
  else
    sub_C35A40((__int64)&v169, 0);
  if ( v169 == v22 )
    sub_C3E660((__int64)&v167, (__int64)&v169);
  else
    sub_C3A850((__int64)&v167, (__int64 *)&v169);
  *(_QWORD *)&v123 = sub_34007B0(v24, (unsigned int)&v167, (unsigned int)&v153, v150, v151, 0, 0);
  *((_QWORD *)&v123 + 1) = v25;
  if ( (unsigned int)v168 > 0x40 && v167 )
    j_j___libc_free_0_0((unsigned __int64)v167);
  if ( v169 == v22 )
  {
    if ( v170 )
    {
      for ( i = &v170[3 * *(v170 - 1)]; v170 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v22 == (_DWORD *)*i )
            break;
          sub_C338F0((__int64)i);
          if ( v170 == i )
            goto LABEL_117;
        }
      }
LABEL_117:
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v169);
  }
  v26 = *(_QWORD *)(v3 + 16);
  if ( v144 == v22 )
    sub_C3C500(&v169, (__int64)v144);
  else
    sub_C373C0(&v169, (__int64)v144);
  if ( v22 == v169 )
    sub_C3CF20((__int64)&v169, 0);
  else
    sub_C36EF0(&v169, 0);
  if ( v169 == v22 )
    sub_C3E660((__int64)&v167, (__int64)&v169);
  else
    sub_C3A850((__int64)&v167, (__int64 *)&v169);
  *(_QWORD *)&v122 = sub_34007B0(v26, (unsigned int)&v167, (unsigned int)&v153, v150, v151, 0, 0);
  *((_QWORD *)&v122 + 1) = v27;
  if ( (unsigned int)v168 > 0x40 && v167 )
    j_j___libc_free_0_0((unsigned __int64)v167);
  if ( v169 == v22 )
  {
    if ( v170 )
    {
      for ( j = &v170[3 * *(v170 - 1)]; v170 != j; sub_969EE0((__int64)j) )
      {
        while ( 1 )
        {
          j -= 3;
          if ( v22 == (_DWORD *)*j )
            break;
          sub_C338F0((__int64)j);
          if ( v170 == j )
            goto LABEL_106;
        }
      }
LABEL_106:
      j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v169);
  }
  v158 = v18;
  v28 = v18 - 1;
  v147 = v142 - 1;
  v29 = (v18 - 1) & 0x3F;
  v30 = 1LL << ((unsigned __int8)v18 - 1);
  if ( (unsigned int)v18 <= 0x40 )
  {
    v157 = 0;
    if ( v142 == 1 )
    {
LABEL_52:
      v157 |= v30;
      goto LABEL_53;
    }
    if ( v147 <= 0x40 )
    {
      v31 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v142);
      v32 = 0;
LABEL_51:
      v157 = v32 | v31;
      goto LABEL_52;
    }
LABEL_153:
    v125 = v29;
    v132 = v30;
    v137 = v28;
    sub_C43C90(&v157, 0, v147);
    v29 = v125;
    v30 = v132;
    v28 = v137;
    goto LABEL_90;
  }
  sub_C43690((__int64)&v157, 0, 0);
  v28 = v18 - 1;
  v30 = 1LL << ((unsigned __int8)v18 - 1);
  v29 = (v18 - 1) & 0x3F;
  if ( v142 == 1 )
    goto LABEL_90;
  if ( v147 > 0x40 )
    goto LABEL_153;
  v31 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v142);
  v32 = v157;
  if ( v158 <= 0x40 )
    goto LABEL_51;
  *(_QWORD *)v157 |= v31;
LABEL_90:
  if ( v158 <= 0x40 )
    goto LABEL_52;
  *(_QWORD *)(v157 + 8LL * (v28 >> 6)) |= v30;
LABEL_53:
  v160 = v18;
  v33 = ~v30;
  if ( (unsigned int)v18 > 0x40 )
  {
    v131 = v33;
    v136 = v28;
    sub_C43690((__int64)&v159, -1, 1);
    v33 = v131;
    if ( v160 > 0x40 )
    {
      *(_QWORD *)(v159 + 8LL * (v136 >> 6)) &= v131;
      goto LABEL_58;
    }
  }
  else
  {
    v34 = 0xFFFFFFFFFFFFFFFFLL >> (63 - v29);
    if ( !(_DWORD)v18 )
      v34 = 0;
    v159 = v34;
  }
  v159 &= v33;
LABEL_58:
  *(_QWORD *)&v35 = sub_34007B0(*(_QWORD *)(v3 + 16), (unsigned int)&v159, (unsigned int)&v153, v150, v151, 0, 0);
  v133 = v35;
  *(_QWORD *)&v36 = sub_34007B0(*(_QWORD *)(v3 + 16), (unsigned int)&v157, (unsigned int)&v153, v150, v151, 0, 0);
  v121 = v36;
  sub_C43310((void **)&v163, v144, (unsigned __int64)"1.0", 3u);
  if ( v163 == v22 )
    sub_C3C790(&v169, (_QWORD **)&v163);
  else
    sub_C33EB0(&v169, (__int64 *)&v163);
  sub_3329C90(&v165, (__int64 *)&v169, v142 + 1, 1);
  if ( v169 == v22 )
  {
    if ( v170 )
    {
      for ( k = &v170[3 * *(v170 - 1)]; v170 != k; sub_969EE0((__int64)k) )
      {
        while ( 1 )
        {
          k -= 3;
          if ( v22 == (_DWORD *)*k )
            break;
          sub_C338F0((__int64)k);
          if ( v170 == k )
            goto LABEL_141;
        }
      }
LABEL_141:
      j_j_j___libc_free_0_0((unsigned __int64)(k - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v169);
  }
  *(_QWORD *)&v37 = sub_33FE6E0(*(_QWORD *)(v3 + 16), &v165, &v153, v155, v156, 0);
  *(_QWORD *)&v39 = sub_3406EB0(*(_QWORD *)(v3 + 16), 98, (unsigned int)&v153, v155, (_DWORD)v156, v38, v141, v37);
  v40 = *(_QWORD *)(v3 + 8);
  v119 = v39;
  v41 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD *))(*(_QWORD *)v40 + 528LL);
  *(_QWORD *)&v39 = *(_QWORD *)(v3 + 16);
  v128 = *(_QWORD *)(v39 + 64);
  v42 = sub_2E79000(*(__int64 **)(v39 + 40));
  v43 = v41(v40, v42, v128, v155, v156);
  LODWORD(v41) = v44;
  LODWORD(v40) = v43;
  *(_QWORD *)&v46 = sub_33FAF80(*(_QWORD *)(v3 + 16), 234, (unsigned int)&v153, v150, v151, v45, v141);
  v117 = v46;
  *(_QWORD *)&v48 = sub_3406EB0(*(_QWORD *)(v3 + 16), 186, (unsigned int)&v153, v150, v151, v47, v46, v133);
  v134 = v48;
  *(_QWORD *)&v118 = sub_3406EB0(*(_QWORD *)(v3 + 16), 56, (unsigned int)&v153, v150, v151, v49, v48, v138);
  v129 = *(_QWORD *)(v3 + 16);
  *((_QWORD *)&v118 + 1) = v50;
  *(_QWORD *)&v51 = sub_33ED040(v129, 13);
  *(_QWORD *)&v130 = sub_340F900(v129, 208, (unsigned int)&v153, v40, (_DWORD)v41, DWORD2(v118), v118, v138, v51);
  v139 = *(_QWORD *)(v3 + 16);
  *((_QWORD *)&v130 + 1) = v52;
  *(_QWORD *)&v53 = sub_33ED040(v139, 12);
  *(_QWORD *)&v55 = sub_340F900(v139, 208, (unsigned int)&v153, v40, (_DWORD)v41, v54, v134, v123, v53);
  v140 = v55;
  *(_QWORD *)&v56 = sub_3401400(*(_QWORD *)(v3 + 16), v126, (unsigned int)&v153, v148, v149, 0, 0);
  LODWORD(v115) = 0;
  v124 = v56;
  *(_QWORD *)&v57 = sub_3400BD0(*(_QWORD *)(v3 + 16), 0, (unsigned int)&v153, v148, v149, 0, v115);
  v127 = v57;
  *(_QWORD *)&v59 = sub_33FAF80(*(_QWORD *)(v3 + 16), 234, (unsigned int)&v153, v150, v151, v58, v119);
  v60 = *((_QWORD *)&v59 + 1);
  v61 = v59;
  *(_QWORD *)&v63 = sub_340F900(*(_QWORD *)(v3 + 16), 205, (unsigned int)&v153, v150, v151, v62, v140, v59, v117);
  *((_QWORD *)&v112 + 1) = v60;
  *(_QWORD *)&v112 = v61;
  v120 = v63;
  *(_QWORD *)&v65 = sub_3406EB0(*(_QWORD *)(v3 + 16), 186, (unsigned int)&v153, v150, v151, v64, v112, v122);
  v67 = sub_340F900(*(_QWORD *)(v3 + 16), 205, (unsigned int)&v153, v150, v151, v66, v140, v65, v134);
  v69 = v68;
  v70 = v67;
  *(_QWORD *)&v71 = sub_3400E40(*(_QWORD *)(v3 + 16), v147, v150, v151, &v153);
  *((_QWORD *)&v113 + 1) = v69;
  *(_QWORD *)&v113 = v70;
  v73 = sub_3406EB0(*(_QWORD *)(v3 + 16), 192, (unsigned int)&v153, v150, v151, v72, v113, v71);
  *(_QWORD *)&v75 = sub_33FB160(*(_QWORD *)(v3 + 16), v73, v74, &v153, v148, v149);
  *(_QWORD *)&v77 = sub_3406EB0(*(_QWORD *)(v3 + 16), 56, (unsigned int)&v153, v148, v149, v76, v75, v124);
  v135 = v77;
  *(_QWORD *)&v78 = sub_3400BD0(*(_QWORD *)(v3 + 16), ~v142, (unsigned int)&v153, v148, v149, 0, 0);
  *(_QWORD *)&v80 = sub_340F900(*(_QWORD *)(v3 + 16), 205, (unsigned int)&v153, v148, v149, v79, v140, v78, v127);
  v143 = v80;
  v82 = sub_3406EB0(*(_QWORD *)(v3 + 16), 186, (unsigned int)&v153, v150, v151, v81, v120, v121);
  v84 = v83;
  v85 = v82;
  sub_C43310((void **)&v167, v144, (unsigned __int64)"0.5", 3u);
  v145 = *(_QWORD *)(v3 + 16);
  if ( v167 == v22 )
    sub_C3E660((__int64)&v169, (__int64)&v167);
  else
    sub_C3A850((__int64)&v169, (__int64 *)&v167);
  *(_QWORD *)&v86 = sub_34007B0(v145, (unsigned int)&v169, (unsigned int)&v153, v150, v151, 0, 0);
  if ( (unsigned int)v170 > 0x40 && v169 )
  {
    v146 = v86;
    j_j___libc_free_0_0((unsigned __int64)v169);
    v86 = v146;
  }
  *((_QWORD *)&v114 + 1) = v84;
  *(_QWORD *)&v114 = v85;
  *(_QWORD *)&v88 = sub_3406EB0(*(_QWORD *)(v3 + 16), 187, (unsigned int)&v153, v150, v151, v87, v114, v86);
  v90 = sub_33FAF80(*(_QWORD *)(v3 + 16), 234, (unsigned int)&v153, v155, (_DWORD)v156, v89, v88);
  v92 = v91;
  v93 = v90;
  *(_QWORD *)&v95 = sub_3406EB0(*(_QWORD *)(v3 + 16), 56, (unsigned int)&v153, v148, v149, v94, v135, v143);
  *((_QWORD *)&v116 + 1) = v92;
  *(_QWORD *)&v116 = v93;
  v152 = v95;
  v97 = sub_340F900(*(_QWORD *)(v3 + 16), 205, (unsigned int)&v153, v155, (_DWORD)v156, v96, v130, v141, v116);
  v99 = v98;
  v100 = (_DWORD *)v97;
  v102 = sub_340F900(*(_QWORD *)(v3 + 16), 205, (unsigned int)&v153, v148, v149, v101, v130, v127, v152);
  v103 = *(_QWORD *)(v3 + 16);
  v172 = v104;
  v169 = v100;
  v170 = v99;
  v171 = v102;
  v10 = sub_3411660(v103, &v169, 2, &v153);
  if ( v167 == v22 )
  {
    if ( v168 )
    {
      for ( m = &v168[3 * *(v168 - 1)]; v168 != m; sub_969EE0((__int64)m) )
      {
        while ( 1 )
        {
          m -= 3;
          if ( v22 == (_DWORD *)*m )
            break;
          sub_C338F0((__int64)m);
          if ( v168 == m )
            goto LABEL_133;
        }
      }
LABEL_133:
      j_j_j___libc_free_0_0((unsigned __int64)(m - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v167);
  }
  if ( v165 == v22 )
  {
    if ( v166 )
    {
      for ( n = &v166[3 * *(v166 - 1)]; v166 != n; sub_969EE0((__int64)n) )
      {
        while ( 1 )
        {
          n -= 3;
          if ( v22 == (_DWORD *)*n )
            break;
          sub_C338F0((__int64)n);
          if ( v166 == n )
            goto LABEL_126;
        }
      }
LABEL_126:
      j_j_j___libc_free_0_0((unsigned __int64)(n - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v165);
  }
  if ( v163 == v22 )
  {
    if ( v164 )
    {
      for ( ii = &v164[3 * *(v164 - 1)]; v164 != ii; sub_969EE0((__int64)ii) )
      {
        while ( 1 )
        {
          ii -= 3;
          if ( v22 == (_DWORD *)*ii )
            break;
          sub_C338F0((__int64)ii);
          if ( v164 == ii )
            goto LABEL_98;
        }
      }
LABEL_98:
      j_j_j___libc_free_0_0((unsigned __int64)(ii - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v163);
  }
  if ( v160 > 0x40 && v159 )
    j_j___libc_free_0_0(v159);
  if ( v158 > 0x40 && v157 )
    j_j___libc_free_0_0(v157);
LABEL_6:
  if ( v153 )
    sub_B91220((__int64)&v153, v153);
  return v10;
}
