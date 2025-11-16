// Function: sub_346C5C0
// Address: 0x346c5c0
//
unsigned __int8 *__fastcall sub_346C5C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8)
{
  int v9; // r15d
  unsigned __int16 *v11; // rax
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rdx
  unsigned int v17; // eax
  unsigned __int16 v18; // bx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rdx
  unsigned int v28; // ebx
  __int128 v29; // rax
  __int64 v30; // r14
  __int128 v31; // rax
  __int64 v32; // r9
  int v33; // r9d
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // r14
  unsigned __int64 v37; // rax
  __int128 v38; // rax
  __int64 v39; // r9
  unsigned __int8 *v40; // r14
  unsigned int v41; // edx
  unsigned int v42; // ebx
  unsigned int v43; // edx
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int16 v50; // ax
  __int64 v51; // rdx
  unsigned __int8 *v52; // rax
  __int64 v53; // rdx
  __int128 v54; // rax
  int v55; // r9d
  __int128 v56; // rax
  __int128 v57; // rax
  __int64 v58; // r15
  __int64 v59; // r14
  __int128 v60; // rax
  __int64 v61; // r9
  unsigned __int8 *v62; // r14
  __int64 v63; // rdx
  __int64 v64; // rbx
  __int64 (__fastcall *v65)(__int64, __int64, __int64, __int64, __int64); // r15
  unsigned __int16 *v66; // rax
  __int64 v67; // rax
  int v68; // eax
  __int64 v69; // rdx
  __int64 v70; // r15
  __int64 v71; // rdx
  __int128 v72; // rax
  __int64 v73; // r14
  __int64 v74; // rdx
  __int64 v75; // r15
  __int64 (__fastcall *v76)(__int64, __int64, __int64, __int64, _QWORD); // rbx
  __int64 v77; // rax
  int v78; // eax
  __int64 v79; // rdx
  __int64 v80; // rbx
  __int128 v81; // rax
  __int64 v82; // r9
  __int128 v83; // rax
  __int64 v84; // r9
  unsigned int v85; // edx
  __int128 v86; // rax
  __int64 v87; // r9
  __int64 v88; // r14
  unsigned int v89; // edx
  __int64 v90; // rcx
  __int64 v91; // rdx
  __int16 v92; // ax
  __int64 v93; // rdx
  __int64 v94; // r15
  unsigned int v95; // esi
  __int128 v96; // rax
  __int64 v97; // r9
  unsigned __int8 *v98; // rax
  __int64 v99; // rdx
  __int64 v100; // r15
  unsigned __int8 *v101; // r14
  __int64 v102; // rdx
  __int16 v103; // ax
  __int64 v104; // rdx
  __int64 v105; // rsi
  __int64 v106; // rax
  unsigned __int16 v107; // r15
  __int64 v108; // r14
  unsigned int v109; // edx
  unsigned int v110; // ebx
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rdx
  __int128 v114; // rax
  __int64 v115; // r9
  unsigned int v116; // edx
  int v117; // r9d
  unsigned int v118; // edx
  __int64 v119; // r9
  int v120; // r9d
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  __int128 v124; // [rsp-30h] [rbp-1E0h]
  __int128 v125; // [rsp-20h] [rbp-1D0h]
  __int128 v126; // [rsp-20h] [rbp-1D0h]
  __int128 v127; // [rsp-20h] [rbp-1D0h]
  __int128 v128; // [rsp-10h] [rbp-1C0h]
  __int128 v129; // [rsp-10h] [rbp-1C0h]
  __int128 v130; // [rsp+0h] [rbp-1B0h]
  __int128 v131; // [rsp+0h] [rbp-1B0h]
  __int64 v132; // [rsp+20h] [rbp-190h]
  __int128 v133; // [rsp+20h] [rbp-190h]
  __int128 v134; // [rsp+30h] [rbp-180h]
  __int128 v135; // [rsp+40h] [rbp-170h]
  __int64 v136; // [rsp+50h] [rbp-160h]
  __int64 v137; // [rsp+50h] [rbp-160h]
  int v138; // [rsp+5Ch] [rbp-154h]
  __int128 v139; // [rsp+60h] [rbp-150h]
  __int64 v140; // [rsp+70h] [rbp-140h]
  __int64 v141; // [rsp+78h] [rbp-138h]
  __int64 v142; // [rsp+80h] [rbp-130h]
  __int64 v143; // [rsp+80h] [rbp-130h]
  __int128 v144; // [rsp+90h] [rbp-120h]
  __int128 v145; // [rsp+A0h] [rbp-110h]
  unsigned __int8 *v146; // [rsp+A0h] [rbp-110h]
  unsigned int v147; // [rsp+B8h] [rbp-F8h]
  unsigned __int8 *v148; // [rsp+C0h] [rbp-F0h]
  __int64 v149; // [rsp+C8h] [rbp-E8h]
  unsigned __int64 v150; // [rsp+C8h] [rbp-E8h]
  __int64 v151; // [rsp+D0h] [rbp-E0h]
  unsigned int v152; // [rsp+D8h] [rbp-D8h]
  __int128 v153; // [rsp+E0h] [rbp-D0h]
  unsigned __int64 v154; // [rsp+E8h] [rbp-C8h]
  unsigned __int8 *v157; // [rsp+100h] [rbp-B0h]
  __int64 v158; // [rsp+130h] [rbp-80h] BYREF
  __int64 v159; // [rsp+138h] [rbp-78h]
  unsigned int v160; // [rsp+140h] [rbp-70h] BYREF
  __int64 v161; // [rsp+148h] [rbp-68h]
  __int64 v162; // [rsp+150h] [rbp-60h]
  __int64 v163; // [rsp+158h] [rbp-58h]
  unsigned __int16 v164; // [rsp+160h] [rbp-50h] BYREF
  __int64 v165; // [rsp+168h] [rbp-48h]
  unsigned __int64 v166; // [rsp+170h] [rbp-40h] BYREF
  __int64 v167; // [rsp+178h] [rbp-38h]

  v158 = a2;
  v9 = (unsigned __int16)a2;
  v159 = a3;
  v11 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  LOWORD(v160) = v12;
  v161 = v13;
  if ( (_WORD)v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0xD3u )
    {
LABEL_3:
      v14 = v159;
      goto LABEL_4;
    }
    v14 = 0;
    LOWORD(v9) = word_4456580[v9 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v158) )
      goto LABEL_3;
    v50 = sub_3009970((__int64)&v158, a5, v47, v48, v49);
    v12 = (unsigned __int16)v160;
    LOWORD(v9) = v50;
    v14 = v51;
  }
LABEL_4:
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 17) <= 0xD3u )
    {
      v15 = 0;
      LOWORD(v12) = word_4456580[v12 - 1];
      goto LABEL_7;
    }
  }
  else if ( sub_30070B0((__int64)&v160) )
  {
    LOWORD(v12) = sub_3009970((__int64)&v160, a5, v44, v45, v46);
    goto LABEL_7;
  }
  v15 = v161;
LABEL_7:
  if ( (_WORD)v12 == (_WORD)v9 && ((_WORD)v12 || v14 == v15) )
    return (unsigned __int8 *)a4;
  v17 = sub_327FF20((unsigned __int16 *)&v158, a5);
  v18 = v160;
  v151 = v19;
  v152 = v17;
  if ( (_WORD)v160 )
  {
    if ( (unsigned __int16)(v160 - 17) <= 0xD3u )
    {
      v167 = 0;
      v18 = word_4456580[(unsigned __int16)v160 - 1];
      LOWORD(v166) = v18;
      if ( !v18 )
        goto LABEL_15;
      goto LABEL_37;
    }
    goto LABEL_13;
  }
  if ( !sub_30070B0((__int64)&v160) )
  {
LABEL_13:
    v23 = v161;
    goto LABEL_14;
  }
  v18 = sub_3009970((__int64)&v160, a5, v20, v21, v22);
LABEL_14:
  LOWORD(v166) = v18;
  v167 = v23;
  if ( !v18 )
  {
LABEL_15:
    v162 = sub_3007260((__int64)&v166);
    LODWORD(v24) = v162;
    v163 = v25;
    goto LABEL_16;
  }
LABEL_37:
  if ( v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
    goto LABEL_70;
  v24 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
LABEL_16:
  v138 = v24;
  v26 = sub_327FF20((unsigned __int16 *)&v160, a5);
  v140 = v27;
  v147 = v26;
  v28 = v24 - 1;
  *(_QWORD *)&v29 = sub_33FB890(a8, v26, v27, a4, a5, a7);
  LODWORD(v167) = v24;
  v145 = v29;
  v30 = 1LL << ((unsigned __int8)v24 - 1);
  if ( (unsigned int)v24 > 0x40 )
  {
    sub_C43690((__int64)&v166, 0, 0);
    if ( (unsigned int)v167 > 0x40 )
    {
      *(_QWORD *)(v166 + 8LL * (v28 >> 6)) |= v30;
      goto LABEL_19;
    }
  }
  else
  {
    v166 = 0;
  }
  v166 |= v30;
LABEL_19:
  *(_QWORD *)&v31 = sub_34007B0(a8, (__int64)&v166, a6, v147, v140, 0, a7, 0);
  *(_QWORD *)&v153 = sub_3406EB0((_QWORD *)a8, 0xBAu, a6, v147, v140, v32, v145, v31);
  *((_QWORD *)&v153 + 1) = v34;
  if ( (unsigned int)v167 > 0x40 && v166 )
    j_j___libc_free_0_0(v166);
  v35 = 1;
  if ( (_WORD)v160 == 1
    || (_WORD)v160 && (v35 = (unsigned __int16)v160, *(_QWORD *)(a1 + 8LL * (unsigned __int16)v160 + 112)) )
  {
    if ( (*(_BYTE *)(a1 + 500 * v35 + 6659) & 0xFB) == 0 )
    {
      v146 = sub_33FAF80(a8, 245, a6, v160, v161, v33, a7);
      goto LABEL_47;
    }
  }
  v36 = ~v30;
  LODWORD(v167) = v24;
  if ( (unsigned int)v24 <= 0x40 )
  {
    v37 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v24 - 1) & 0x3F));
    if ( !(_DWORD)v24 )
      v37 = 0;
    v166 = v37;
    goto LABEL_28;
  }
  sub_C43690((__int64)&v166, -1, 1);
  if ( (unsigned int)v167 <= 0x40 )
  {
LABEL_28:
    v166 &= v36;
    goto LABEL_29;
  }
  *(_QWORD *)(v166 + 8LL * (v28 >> 6)) &= v36;
LABEL_29:
  *(_QWORD *)&v38 = sub_34007B0(a8, (__int64)&v166, a6, v147, v140, 0, a7, 0);
  v40 = sub_3406EB0((_QWORD *)a8, 0xBAu, a6, v147, v140, v39, v145, v38);
  v42 = v41;
  if ( (unsigned int)v167 > 0x40 && v166 )
    j_j___libc_free_0_0(v166);
  v146 = sub_33FB890(a8, v160, v161, (__int64)v40, v42, a7);
LABEL_47:
  v141 = v43;
  v52 = sub_3406EE0((_QWORD *)a8, (__int64)v146, v43, a6, (unsigned int)v158, v159, a7);
  *(_QWORD *)&v54 = sub_3406EE0((_QWORD *)a8, (__int64)v52, v53, a6, v160, v161, a7);
  v139 = v54;
  *(_QWORD *)&v56 = sub_33FAF80(a8, 234, a6, v152, v151, v55, a7);
  v144 = v56;
  *(_QWORD *)&v57 = sub_3400BD0(a8, 1, a6, v152, v151, 0, a7, 0);
  v58 = *((_QWORD *)&v57 + 1);
  v59 = v57;
  v135 = v57;
  *(_QWORD *)&v60 = sub_34015B0(a8, a6, v152, v151, 0, 0, a7);
  *((_QWORD *)&v128 + 1) = v58;
  *(_QWORD *)&v128 = v59;
  v134 = v60;
  v62 = sub_3406EB0((_QWORD *)a8, 0xBAu, a6, v152, v151, v61, v144, v128);
  v64 = v63;
  v65 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 528LL);
  v66 = (unsigned __int16 *)(*((_QWORD *)v62 + 6) + 16LL * (unsigned int)v63);
  v142 = *(_QWORD *)(a8 + 64);
  v132 = *((_QWORD *)v66 + 1);
  v136 = *v66;
  v67 = sub_2E79000(*(__int64 **)(a8 + 40));
  v68 = v65(a1, v67, v142, v136, v132);
  v70 = v69;
  LODWORD(v136) = v68;
  *(_QWORD *)&v133 = sub_3400BD0(a8, 0, a6, v152, v151, 0, a7, 0);
  *((_QWORD *)&v133 + 1) = v71;
  *(_QWORD *)&v72 = sub_33ED040((_QWORD *)a8, 0x16u);
  *((_QWORD *)&v124 + 1) = v64;
  *(_QWORD *)&v124 = v62;
  v73 = sub_340F900((_QWORD *)a8, 0xD0u, a6, v136, v70, *((__int64 *)&v133 + 1), v124, v133, v72);
  v75 = v74;
  v143 = *(_QWORD *)(a8 + 64);
  v76 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)a1 + 528LL);
  *(_QWORD *)&v133 = *(_QWORD *)(*((_QWORD *)v146 + 6) + 16 * v141 + 8);
  v137 = *(unsigned __int16 *)(*((_QWORD *)v146 + 6) + 16 * v141);
  v77 = sub_2E79000(*(__int64 **)(a8 + 40));
  v78 = v76(a1, v77, v143, v137, v133);
  v80 = v79;
  LODWORD(v143) = v78;
  *(_QWORD *)&v81 = sub_33ED040((_QWORD *)a8, 9u);
  *((_QWORD *)&v125 + 1) = v141;
  *(_QWORD *)&v125 = v146;
  *(_QWORD *)&v83 = sub_340F900((_QWORD *)a8, 0xD0u, a6, v143, v80, v82, v125, v139, v81);
  *((_QWORD *)&v130 + 1) = v75;
  *(_QWORD *)&v130 = v73;
  v149 = *((_QWORD *)&v83 + 1);
  v148 = sub_3406EB0((_QWORD *)a8, 0xBBu, a6, (unsigned int)v143, v80, v84, v83, v130);
  v150 = v85 | v149 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v86 = sub_33ED040((_QWORD *)a8, 2u);
  *((_QWORD *)&v126 + 1) = v141;
  *(_QWORD *)&v126 = v146;
  v88 = sub_340F900((_QWORD *)a8, 0xD0u, a6, v143, v80, v87, v126, v139, v86);
  v90 = v89;
  v91 = *(_QWORD *)(v88 + 48) + 16LL * v89;
  v92 = *(_WORD *)v91;
  v93 = *(_QWORD *)(v91 + 8);
  v94 = v90;
  LOWORD(v166) = v92;
  v167 = v93;
  if ( v92 )
    v95 = ((unsigned __int16)(v92 - 17) < 0xD4u) + 205;
  else
    v95 = 205 - (!sub_30070B0((__int64)&v166) - 1);
  *(_QWORD *)&v96 = sub_340EC60((_QWORD *)a8, v95, a6, v152, v151, 0, v88, v94, v135, v134);
  v98 = sub_3406EB0((_QWORD *)a8, 0x38u, a6, v152, v151, v97, v144, v96);
  v100 = v99;
  v101 = v98;
  v102 = *((_QWORD *)v148 + 6) + 16LL * (unsigned int)v150;
  v103 = *(_WORD *)v102;
  v104 = *(_QWORD *)(v102 + 8);
  LOWORD(v166) = v103;
  v167 = v104;
  if ( v103 )
    v105 = (unsigned int)((unsigned __int16)(v103 - 17) < 0xD4u) + 205;
  else
    v105 = 205 - ((unsigned int)!sub_30070B0((__int64)&v166) - 1);
  *((_QWORD *)&v131 + 1) = v100;
  *(_QWORD *)&v131 = v101;
  v106 = sub_340EC60((_QWORD *)a8, v105, a6, v152, v151, 0, (__int64)v148, v150, v144, v131);
  v107 = v158;
  v108 = v106;
  v110 = v109;
  if ( (_WORD)v158 )
  {
    if ( (unsigned __int16)(v158 - 17) <= 0xD3u )
    {
      v165 = 0;
      v107 = word_4456580[(unsigned __int16)v158 - 1];
      v164 = v107;
      if ( !v107 )
        goto LABEL_55;
      goto LABEL_66;
    }
    goto LABEL_53;
  }
  if ( !sub_30070B0((__int64)&v158) )
  {
LABEL_53:
    v111 = v159;
    goto LABEL_54;
  }
  v107 = sub_3009970((__int64)&v158, v105, v121, v122, v123);
LABEL_54:
  v164 = v107;
  v165 = v111;
  if ( v107 )
  {
LABEL_66:
    if ( v107 != 1 && (unsigned __int16)(v107 - 504) > 7u )
    {
      v112 = *(_QWORD *)&byte_444C4A0[16 * v107 - 16];
      goto LABEL_56;
    }
LABEL_70:
    BUG();
  }
LABEL_55:
  v112 = sub_3007260((__int64)&v164);
  v166 = v112;
  v167 = v113;
LABEL_56:
  *(_QWORD *)&v114 = sub_3400E40(a8, v138 - (int)v112, v147, v140, a6, a7);
  sub_3406EB0((_QWORD *)a8, 0xC0u, a6, v147, v140, v115, v153, v114);
  v154 = v116 | *((_QWORD *)&v153 + 1) & 0xFFFFFFFF00000000LL;
  v157 = sub_33FAF80(a8, 216, a6, v152, v151, v117, a7);
  *((_QWORD *)&v129 + 1) = v118 | v154 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v129 = v157;
  *((_QWORD *)&v127 + 1) = v110 | a5 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v127 = v108;
  sub_3406EB0((_QWORD *)a8, 0xBBu, a6, v152, v151, v119, v127, v129);
  return sub_33FAF80(a8, 234, a6, (unsigned int)v158, v159, v120, a7);
}
