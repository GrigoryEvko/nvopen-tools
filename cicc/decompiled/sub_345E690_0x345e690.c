// Function: sub_345E690
// Address: 0x345e690
//
unsigned __int8 *__fastcall sub_345E690(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rsi
  __int64 *v7; // rdi
  __int16 *v8; // rax
  __int16 v9; // dx
  const __m128i *v10; // rax
  __m128i v11; // xmm0
  unsigned __int32 v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // eax
  unsigned __int16 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // r10
  __int64 v20; // rdx
  unsigned __int64 v21; // r15
  unsigned int v22; // edx
  __int128 v23; // rax
  __int64 v24; // r9
  unsigned int v25; // edx
  unsigned __int8 *v26; // r14
  unsigned __int64 v27; // r15
  __int64 v28; // rax
  __int128 v29; // rax
  __int64 v30; // r9
  unsigned int v31; // edx
  __int64 v32; // r9
  unsigned int v33; // edx
  __int128 v34; // rax
  __int64 v35; // r9
  int v36; // r9d
  __int128 v37; // rax
  __int64 v38; // r9
  unsigned int v39; // edx
  __int64 v40; // r15
  __int128 v41; // rax
  __int64 v42; // r9
  unsigned int v43; // edx
  unsigned __int64 v44; // r15
  __int128 v45; // rax
  __int64 v46; // r9
  unsigned int v47; // edx
  __int128 v48; // rax
  __int64 v49; // r9
  unsigned int v50; // edx
  __int64 v51; // r9
  unsigned int v52; // edx
  __int128 v53; // rax
  __int64 v54; // r9
  unsigned int v55; // edx
  unsigned __int64 v56; // r15
  __int128 v57; // rax
  __int64 v58; // r9
  unsigned int v59; // edx
  unsigned __int64 v60; // r15
  __int128 v61; // rax
  __int64 v62; // r9
  unsigned int v63; // edx
  __int128 v64; // rax
  __int64 v65; // r9
  unsigned int v66; // edx
  __int64 v67; // r9
  unsigned int v68; // edx
  __int128 v69; // rax
  __int64 v70; // r9
  unsigned int v71; // edx
  unsigned __int64 v72; // r15
  __int128 v73; // rax
  __int64 v74; // r9
  unsigned int v75; // edx
  unsigned __int64 v76; // r15
  __int128 v77; // rax
  __int64 v78; // r9
  unsigned int v79; // edx
  __int128 v80; // rax
  __int64 v81; // r9
  unsigned int v82; // edx
  __int64 v83; // r9
  unsigned __int8 *v84; // r12
  unsigned int v86; // edx
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  unsigned __int32 v90; // edx
  __int128 v91; // [rsp-30h] [rbp-260h]
  __int128 v92; // [rsp-30h] [rbp-260h]
  __int128 v93; // [rsp-30h] [rbp-260h]
  __int128 v94; // [rsp-30h] [rbp-260h]
  __int128 v95; // [rsp-30h] [rbp-260h]
  __int128 v96; // [rsp-30h] [rbp-260h]
  __int128 v97; // [rsp-20h] [rbp-250h]
  __int128 v98; // [rsp-20h] [rbp-250h]
  __int128 v99; // [rsp-20h] [rbp-250h]
  __int128 v100; // [rsp-20h] [rbp-250h]
  __int128 v101; // [rsp-10h] [rbp-240h]
  __int128 v102; // [rsp-10h] [rbp-240h]
  __int128 v103; // [rsp-10h] [rbp-240h]
  __int128 v104; // [rsp-10h] [rbp-240h]
  unsigned __int8 *v105; // [rsp+10h] [rbp-220h]
  unsigned int v106; // [rsp+18h] [rbp-218h]
  __int64 v107; // [rsp+20h] [rbp-210h]
  unsigned int v108; // [rsp+30h] [rbp-200h]
  unsigned __int8 *v109; // [rsp+38h] [rbp-1F8h]
  unsigned int v110; // [rsp+40h] [rbp-1F0h]
  unsigned int v111; // [rsp+44h] [rbp-1ECh]
  unsigned int v112; // [rsp+48h] [rbp-1E8h]
  unsigned int v113; // [rsp+50h] [rbp-1E0h]
  __int128 v114; // [rsp+50h] [rbp-1E0h]
  int v115; // [rsp+50h] [rbp-1E0h]
  unsigned __int64 v116; // [rsp+68h] [rbp-1C8h]
  __int128 v117; // [rsp+70h] [rbp-1C0h]
  unsigned __int64 v118; // [rsp+78h] [rbp-1B8h]
  unsigned __int64 v119; // [rsp+78h] [rbp-1B8h]
  unsigned __int8 *v120; // [rsp+80h] [rbp-1B0h]
  unsigned __int8 *v121; // [rsp+B0h] [rbp-180h]
  unsigned __int8 *v122; // [rsp+D0h] [rbp-160h]
  unsigned __int8 *v123; // [rsp+E0h] [rbp-150h]
  unsigned __int8 *v124; // [rsp+100h] [rbp-130h]
  unsigned __int8 *v125; // [rsp+120h] [rbp-110h]
  unsigned __int8 *v126; // [rsp+130h] [rbp-100h]
  unsigned __int8 *v127; // [rsp+150h] [rbp-E0h]
  unsigned __int8 *v128; // [rsp+170h] [rbp-C0h]
  unsigned __int8 *v129; // [rsp+180h] [rbp-B0h]
  __int64 v130; // [rsp+190h] [rbp-A0h] BYREF
  int v131; // [rsp+198h] [rbp-98h]
  unsigned int v132; // [rsp+1A0h] [rbp-90h] BYREF
  __int64 v133; // [rsp+1A8h] [rbp-88h]
  unsigned __int64 v134; // [rsp+1B0h] [rbp-80h] BYREF
  unsigned int v135; // [rsp+1B8h] [rbp-78h]
  unsigned __int64 v136; // [rsp+1C0h] [rbp-70h] BYREF
  unsigned int v137; // [rsp+1C8h] [rbp-68h]
  unsigned __int64 v138; // [rsp+1D0h] [rbp-60h] BYREF
  unsigned int v139; // [rsp+1D8h] [rbp-58h]
  unsigned __int64 v140; // [rsp+1E0h] [rbp-50h] BYREF
  __int64 v141; // [rsp+1E8h] [rbp-48h]
  __int64 v142; // [rsp+1F0h] [rbp-40h]
  __int64 v143; // [rsp+1F8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v130 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v130, v6, 1);
  v7 = (__int64 *)a3[5];
  v131 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v133 = *((_QWORD *)v8 + 1);
  v10 = *(const __m128i **)(a2 + 40);
  LOWORD(v132) = v9;
  v11 = _mm_loadu_si128(v10);
  v116 = v11.m128i_u64[1];
  v105 = (unsigned __int8 *)v10->m128i_i64[0];
  v12 = v10->m128i_u32[2];
  v13 = sub_2E79000(v7);
  v14 = v132;
  v15 = sub_2FE6750(a1, v132, v133, v13);
  v16 = v132;
  v107 = v17;
  v106 = v15;
  if ( (_WORD)v132 )
  {
    if ( (unsigned __int16)(v132 - 17) <= 0xD3u )
    {
      v16 = word_4456580[(unsigned __int16)v132 - 1];
      v141 = 0;
      LOWORD(v140) = v16;
      if ( !v16 )
        goto LABEL_7;
      goto LABEL_44;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v132) )
  {
LABEL_5:
    v18 = v133;
    goto LABEL_6;
  }
  v16 = sub_3009970((__int64)&v132, v14, v87, v88, v89);
LABEL_6:
  LOWORD(v140) = v16;
  v141 = v18;
  if ( !v16 )
  {
LABEL_7:
    v142 = sub_3007260((__int64)&v140);
    LODWORD(v19) = v142;
    v143 = v20;
    goto LABEL_8;
  }
LABEL_44:
  if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
    BUG();
  v19 = *(_QWORD *)&byte_444C4A0[16 * v16 - 16];
LABEL_8:
  v111 = v19;
  v21 = 0;
  v112 = v19 - 1;
  if ( (unsigned int)v19 <= 7 )
  {
    v115 = v19;
    v109 = sub_3400BD0((__int64)a3, 0, (__int64)&v130, v132, v133, 0, v11, 0);
    v108 = v86;
    v118 = v86;
    if ( v115 )
      goto LABEL_11;
LABEL_50:
    v84 = v109;
  }
  else
  {
    if ( ((unsigned int)v19 & ((_DWORD)v19 - 1)) != 0 )
    {
      v109 = sub_3400BD0((__int64)a3, 0, (__int64)&v130, v132, v133, 0, v11, 0);
      v108 = v22;
      v118 = v22;
LABEL_11:
      v113 = 0;
      v110 = v112;
      while ( 1 )
      {
        if ( v113 < v112 )
        {
          *(_QWORD *)&v23 = sub_3400BD0((__int64)a3, v110, (__int64)&v130, v106, v107, 0, v11, 0);
          v116 = v12 | v116 & 0xFFFFFFFF00000000LL;
          v26 = sub_3406EB0(a3, 0xBEu, (__int64)&v130, v132, v133, v24, __PAIR128__(v116, (unsigned __int64)v105), v23);
        }
        else
        {
          *(_QWORD *)&v34 = sub_3400BD0((__int64)a3, -v110, (__int64)&v130, v106, v107, 0, v11, 0);
          v116 = v12 | v116 & 0xFFFFFFFF00000000LL;
          v26 = sub_3406EB0(a3, 0xC0u, (__int64)&v130, v132, v133, v35, __PAIR128__(v116, (unsigned __int64)v105), v34);
        }
        LODWORD(v141) = v111;
        v27 = v25 | v21 & 0xFFFFFFFF00000000LL;
        v28 = 1LL << v112;
        if ( v111 <= 0x40 )
          break;
        sub_C43690((__int64)&v140, 0, 0);
        v28 = 1LL << v112;
        if ( (unsigned int)v141 <= 0x40 )
          goto LABEL_15;
        *(_QWORD *)(v140 + 8LL * (v112 >> 6)) |= 1LL << v112;
LABEL_16:
        *(_QWORD *)&v29 = sub_34007B0((__int64)a3, (__int64)&v140, (__int64)&v130, v132, v133, 0, v11, 0);
        *((_QWORD *)&v91 + 1) = v27;
        *(_QWORD *)&v91 = v26;
        v120 = sub_3406EB0(a3, 0xBAu, (__int64)&v130, v132, v133, v30, v91, v29);
        v21 = v31 | v27 & 0xFFFFFFFF00000000LL;
        *((_QWORD *)&v101 + 1) = v21;
        *(_QWORD *)&v101 = v120;
        v119 = v108 | v118 & 0xFFFFFFFF00000000LL;
        *((_QWORD *)&v97 + 1) = v119;
        *(_QWORD *)&v97 = v109;
        v109 = sub_3406EB0(a3, 0xBBu, (__int64)&v130, v132, v133, v32, v97, v101);
        v108 = v33;
        v118 = v33 | v119 & 0xFFFFFFFF00000000LL;
        if ( (unsigned int)v141 > 0x40 && v140 )
          j_j___libc_free_0_0(v140);
        ++v113;
        --v112;
        v110 -= 2;
        if ( v111 <= v113 )
          goto LABEL_50;
      }
      v140 = 0;
LABEL_15:
      v140 |= v28;
      goto LABEL_16;
    }
    LODWORD(v141) = 8;
    v140 = 15;
    sub_C47700((__int64)&v134, v19, (__int64)&v140);
    if ( (unsigned int)v141 > 0x40 && v140 )
      j_j___libc_free_0_0(v140);
    LODWORD(v141) = 8;
    v140 = 51;
    sub_C47700((__int64)&v136, v111, (__int64)&v140);
    if ( (unsigned int)v141 > 0x40 && v140 )
      j_j___libc_free_0_0(v140);
    LODWORD(v141) = 8;
    v140 = 85;
    sub_C47700((__int64)&v138, v111, (__int64)&v140);
    if ( (unsigned int)v141 > 0x40 && v140 )
      j_j___libc_free_0_0(v140);
    if ( v111 != 8 )
    {
      v105 = sub_33FAF80((__int64)a3, 197, (__int64)&v130, v132, v133, v36, v11);
      v12 = v90;
    }
    *(_QWORD *)&v37 = sub_3400BD0((__int64)a3, 4, (__int64)&v130, v106, v107, 0, v11, 0);
    *((_QWORD *)&v92 + 1) = v12;
    *(_QWORD *)&v92 = v105;
    v129 = sub_3406EB0(a3, 0xC0u, (__int64)&v130, v132, v133, v38, v92, v37);
    v40 = v39;
    *(_QWORD *)&v41 = sub_34007B0((__int64)a3, (__int64)&v134, (__int64)&v130, v132, v133, 0, v11, 0);
    *((_QWORD *)&v93 + 1) = v40;
    *(_QWORD *)&v93 = v129;
    v128 = sub_3406EB0(a3, 0xBAu, (__int64)&v130, v132, v133, v42, v93, v41);
    v44 = v43 | v40 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v45 = sub_34007B0((__int64)a3, (__int64)&v134, (__int64)&v130, v132, v133, 0, v11, 0);
    *((_QWORD *)&v94 + 1) = v12;
    *(_QWORD *)&v94 = v105;
    *(_QWORD *)&v114 = sub_3406EB0(a3, 0xBAu, (__int64)&v130, v132, v133, v46, v94, v45);
    *((_QWORD *)&v114 + 1) = v47;
    *(_QWORD *)&v48 = sub_3400BD0((__int64)a3, 4, (__int64)&v130, v106, v107, 0, v11, 0);
    v127 = sub_3406EB0(a3, 0xBEu, (__int64)&v130, v132, v133, v49, v114, v48);
    *((_QWORD *)&v114 + 1) = v50 | *((_QWORD *)&v114 + 1) & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v102 + 1) = *((_QWORD *)&v114 + 1);
    *(_QWORD *)&v102 = v127;
    *((_QWORD *)&v98 + 1) = v44;
    *(_QWORD *)&v98 = v128;
    *(_QWORD *)&v117 = sub_3406EB0(a3, 0xBBu, (__int64)&v130, v132, v133, v51, v98, v102);
    *((_QWORD *)&v117 + 1) = v52;
    *(_QWORD *)&v53 = sub_3400BD0((__int64)a3, 2, (__int64)&v130, v106, v107, 0, v11, 0);
    v126 = sub_3406EB0(a3, 0xC0u, (__int64)&v130, v132, v133, v54, v117, v53);
    v56 = v55 | v44 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v57 = sub_34007B0((__int64)a3, (__int64)&v136, (__int64)&v130, v132, v133, 0, v11, 0);
    *((_QWORD *)&v95 + 1) = v56;
    *(_QWORD *)&v95 = v126;
    v125 = sub_3406EB0(a3, 0xBAu, (__int64)&v130, v132, v133, v58, v95, v57);
    v60 = v59 | v56 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v61 = sub_34007B0((__int64)a3, (__int64)&v136, (__int64)&v130, v132, v133, 0, v11, 0);
    *(_QWORD *)&v114 = sub_3406EB0(a3, 0xBAu, (__int64)&v130, v132, v133, v62, v117, v61);
    *((_QWORD *)&v114 + 1) = v63 | *((_QWORD *)&v114 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v64 = sub_3400BD0((__int64)a3, 2, (__int64)&v130, v106, v107, 0, v11, 0);
    v124 = sub_3406EB0(a3, 0xBEu, (__int64)&v130, v132, v133, v65, v114, v64);
    *((_QWORD *)&v114 + 1) = v66 | *((_QWORD *)&v114 + 1) & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v103 + 1) = *((_QWORD *)&v114 + 1);
    *(_QWORD *)&v103 = v124;
    *((_QWORD *)&v99 + 1) = v60;
    *(_QWORD *)&v99 = v125;
    *(_QWORD *)&v117 = sub_3406EB0(a3, 0xBBu, (__int64)&v130, v132, v133, v67, v99, v103);
    *((_QWORD *)&v117 + 1) = v68 | *((_QWORD *)&v117 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v69 = sub_3400BD0((__int64)a3, 1, (__int64)&v130, v106, v107, 0, v11, 0);
    v123 = sub_3406EB0(a3, 0xC0u, (__int64)&v130, v132, v133, v70, v117, v69);
    v72 = v71 | v60 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v73 = sub_34007B0((__int64)a3, (__int64)&v138, (__int64)&v130, v132, v133, 0, v11, 0);
    *((_QWORD *)&v96 + 1) = v72;
    *(_QWORD *)&v96 = v123;
    v122 = sub_3406EB0(a3, 0xBAu, (__int64)&v130, v132, v133, v74, v96, v73);
    v76 = v75 | v72 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v77 = sub_34007B0((__int64)a3, (__int64)&v138, (__int64)&v130, v132, v133, 0, v11, 0);
    *(_QWORD *)&v114 = sub_3406EB0(a3, 0xBAu, (__int64)&v130, v132, v133, v78, v117, v77);
    *((_QWORD *)&v114 + 1) = v79 | *((_QWORD *)&v114 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v80 = sub_3400BD0((__int64)a3, 1, (__int64)&v130, v106, v107, 0, v11, 0);
    v121 = sub_3406EB0(a3, 0xBEu, (__int64)&v130, v132, v133, v81, v114, v80);
    *((_QWORD *)&v104 + 1) = v82 | *((_QWORD *)&v114 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v104 = v121;
    *((_QWORD *)&v100 + 1) = v76;
    *(_QWORD *)&v100 = v122;
    v84 = sub_3406EB0(a3, 0xBBu, (__int64)&v130, v132, v133, v83, v100, v104);
    if ( v139 > 0x40 && v138 )
      j_j___libc_free_0_0(v138);
    if ( v137 > 0x40 && v136 )
      j_j___libc_free_0_0(v136);
    if ( v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
  }
  if ( v130 )
    sub_B91220((__int64)&v130, v130);
  return v84;
}
