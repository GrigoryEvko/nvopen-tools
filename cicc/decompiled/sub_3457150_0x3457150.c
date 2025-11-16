// Function: sub_3457150
// Address: 0x3457150
//
unsigned __int8 *__fastcall sub_3457150(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rsi
  __int64 *v7; // rdi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  const __m128i *v13; // rax
  unsigned __int16 v14; // r13
  __int64 v15; // rdx
  __int128 v16; // xmm0
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int8 *v22; // r12
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int128 v30; // rax
  __int64 v31; // r9
  __int128 v32; // rax
  __int64 v33; // r9
  unsigned int v34; // edx
  __int64 v35; // r9
  unsigned int v36; // edx
  __int64 v37; // r9
  unsigned int v38; // edx
  __int128 v39; // rax
  __int64 v40; // r9
  __int128 v41; // rax
  __int64 v42; // r9
  unsigned int v43; // edx
  __int64 v44; // r9
  unsigned __int8 *v45; // rax
  unsigned int v46; // edx
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned int v49; // edx
  __int64 v50; // r9
  unsigned int v51; // edx
  __int64 v52; // r9
  unsigned __int8 *v53; // rax
  unsigned int v54; // edx
  __int64 v55; // r9
  __int64 (__fastcall *v56)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int16 v57; // ax
  unsigned __int8 *v58; // r10
  unsigned int v59; // r11d
  __int64 v60; // rdx
  unsigned __int8 v61; // al
  unsigned int v62; // r9d
  _QWORD *v63; // rbx
  unsigned __int8 *v64; // r12
  unsigned __int64 v65; // r13
  __int128 v66; // rax
  __int64 v67; // r9
  __int128 v68; // rax
  __int64 v69; // r9
  unsigned int v70; // edx
  unsigned __int64 v71; // rax
  __int128 v72; // rax
  __int64 v73; // r9
  unsigned __int8 *v74; // rax
  __int64 v75; // rdx
  unsigned __int8 *v76; // r10
  unsigned __int8 *v77; // r8
  __int64 v78; // r9
  unsigned int v79; // r11d
  unsigned int v80; // edx
  __int128 v81; // [rsp-40h] [rbp-1C0h]
  __int128 v82; // [rsp-40h] [rbp-1C0h]
  __int128 v83; // [rsp-40h] [rbp-1C0h]
  __int128 v84; // [rsp-40h] [rbp-1C0h]
  __int128 v85; // [rsp-30h] [rbp-1B0h]
  __int128 v86; // [rsp-30h] [rbp-1B0h]
  __int128 v87; // [rsp-30h] [rbp-1B0h]
  __int128 v88; // [rsp-30h] [rbp-1B0h]
  __int128 v89; // [rsp-30h] [rbp-1B0h]
  __int128 v90; // [rsp-30h] [rbp-1B0h]
  __int128 v91; // [rsp-30h] [rbp-1B0h]
  __int128 v92; // [rsp-30h] [rbp-1B0h]
  __int128 v93; // [rsp-20h] [rbp-1A0h]
  __int128 v94; // [rsp-20h] [rbp-1A0h]
  __int128 v95; // [rsp-20h] [rbp-1A0h]
  __int128 v96; // [rsp-20h] [rbp-1A0h]
  __int128 v97; // [rsp-20h] [rbp-1A0h]
  __int128 v98; // [rsp-20h] [rbp-1A0h]
  __int128 v99; // [rsp-20h] [rbp-1A0h]
  __int128 v100; // [rsp-20h] [rbp-1A0h]
  __int128 v101; // [rsp+0h] [rbp-180h]
  unsigned int v102; // [rsp+0h] [rbp-180h]
  unsigned int v103; // [rsp+0h] [rbp-180h]
  __int128 v104; // [rsp+10h] [rbp-170h]
  unsigned int v105; // [rsp+10h] [rbp-170h]
  unsigned __int8 *v106; // [rsp+10h] [rbp-170h]
  unsigned __int8 *v107; // [rsp+10h] [rbp-170h]
  unsigned int v108; // [rsp+10h] [rbp-170h]
  __int64 v109; // [rsp+18h] [rbp-168h]
  __int128 v110; // [rsp+20h] [rbp-160h]
  unsigned __int8 *v111; // [rsp+20h] [rbp-160h]
  __int128 v112; // [rsp+20h] [rbp-160h]
  unsigned int v113; // [rsp+30h] [rbp-150h]
  unsigned __int8 *v114; // [rsp+30h] [rbp-150h]
  unsigned __int8 *v115; // [rsp+30h] [rbp-150h]
  __int64 v116; // [rsp+38h] [rbp-148h]
  __int64 v117; // [rsp+38h] [rbp-148h]
  unsigned __int8 *v118; // [rsp+40h] [rbp-140h]
  unsigned int v119; // [rsp+48h] [rbp-138h]
  __int64 v120; // [rsp+50h] [rbp-130h]
  __int128 v121; // [rsp+60h] [rbp-120h]
  __int128 v122; // [rsp+60h] [rbp-120h]
  __int128 v123; // [rsp+60h] [rbp-120h]
  __int128 v124; // [rsp+70h] [rbp-110h]
  unsigned __int8 *v125; // [rsp+A0h] [rbp-E0h]
  unsigned __int8 *v126; // [rsp+B0h] [rbp-D0h]
  unsigned __int8 *v127; // [rsp+C0h] [rbp-C0h]
  unsigned __int8 *v128; // [rsp+D0h] [rbp-B0h]
  unsigned __int8 *v129; // [rsp+E0h] [rbp-A0h]
  __int64 v130; // [rsp+F0h] [rbp-90h] BYREF
  int v131; // [rsp+F8h] [rbp-88h]
  __int64 v132; // [rsp+100h] [rbp-80h] BYREF
  __int64 v133; // [rsp+108h] [rbp-78h]
  unsigned __int64 v134; // [rsp+110h] [rbp-70h] BYREF
  unsigned int v135; // [rsp+118h] [rbp-68h]
  __int64 v136; // [rsp+120h] [rbp-60h]
  __int64 v137; // [rsp+128h] [rbp-58h]
  unsigned __int64 v138; // [rsp+130h] [rbp-50h] BYREF
  __int64 v139; // [rsp+138h] [rbp-48h]

  v6 = *(_QWORD *)(a2 + 80);
  v130 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v130, v6, 1);
  v7 = (__int64 *)a3[5];
  v131 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v132) = v9;
  v133 = v10;
  v11 = sub_2E79000(v7);
  v12 = (unsigned int)v132;
  v119 = sub_2FE6750(a1, (unsigned int)v132, v133, v11);
  v13 = *(const __m128i **)(a2 + 40);
  v14 = v132;
  v120 = v15;
  v16 = (__int128)_mm_loadu_si128(v13);
  v17 = v13[2].m128i_i64[1];
  v18 = v13[3].m128i_i64[0];
  v124 = (__int128)_mm_loadu_si128(v13 + 5);
  if ( (_WORD)v132 )
  {
    if ( (unsigned __int16)(v132 - 17) <= 0xD3u )
    {
      v139 = 0;
      v14 = word_4456580[(unsigned __int16)v132 - 1];
      LOWORD(v138) = v14;
      if ( !v14 )
        goto LABEL_7;
      goto LABEL_15;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v132) )
  {
LABEL_5:
    v19 = v133;
    goto LABEL_6;
  }
  v14 = sub_3009970((__int64)&v132, v12, v24, v25, v26);
LABEL_6:
  LOWORD(v138) = v14;
  v139 = v19;
  if ( !v14 )
  {
LABEL_7:
    v136 = sub_3007260((__int64)&v138);
    LODWORD(v20) = v136;
    v137 = v21;
    goto LABEL_8;
  }
LABEL_15:
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    BUG();
  v20 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
LABEL_8:
  if ( (unsigned int)v20 > 0x80 || (v20 & 7) != 0 )
  {
    v22 = 0;
  }
  else
  {
    v135 = 8;
    v134 = 85;
    sub_C47700((__int64)&v138, v20, (__int64)&v134);
    *(_QWORD *)&v104 = sub_34007B0((__int64)a3, (__int64)&v138, (__int64)&v130, v132, v133, 0, (__m128i)v16, 0);
    *((_QWORD *)&v104 + 1) = v27;
    if ( (unsigned int)v139 > 0x40 && v138 )
      j_j___libc_free_0_0(v138);
    if ( v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
    v135 = 8;
    v134 = 51;
    sub_C47700((__int64)&v138, v20, (__int64)&v134);
    *(_QWORD *)&v110 = sub_34007B0((__int64)a3, (__int64)&v138, (__int64)&v130, v132, v133, 0, (__m128i)v16, 0);
    *((_QWORD *)&v110 + 1) = v28;
    if ( (unsigned int)v139 > 0x40 && v138 )
      j_j___libc_free_0_0(v138);
    if ( v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
    v135 = 8;
    v134 = 15;
    sub_C47700((__int64)&v138, v20, (__int64)&v134);
    *(_QWORD *)&v101 = sub_34007B0((__int64)a3, (__int64)&v138, (__int64)&v130, v132, v133, 0, (__m128i)v16, 0);
    *((_QWORD *)&v101 + 1) = v29;
    if ( (unsigned int)v139 > 0x40 && v138 )
      j_j___libc_free_0_0(v138);
    if ( v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
    *(_QWORD *)&v30 = sub_3400BD0((__int64)a3, 1, (__int64)&v130, v119, v120, 0, (__m128i)v16, 0);
    *((_QWORD *)&v85 + 1) = v18;
    *(_QWORD *)&v85 = v17;
    *(_QWORD *)&v32 = sub_33FC130(a3, 398, (__int64)&v130, (unsigned int)v132, v133, v31, v16, v30, v85, v124);
    *((_QWORD *)&v93 + 1) = v18;
    *(_QWORD *)&v93 = v17;
    v129 = sub_33FC130(a3, 396, (__int64)&v130, (unsigned int)v132, v133, v33, v32, v104, v93, v124);
    *((_QWORD *)&v94 + 1) = v18;
    *(_QWORD *)&v94 = v17;
    *((_QWORD *)&v86 + 1) = v34;
    *(_QWORD *)&v86 = v129;
    *(_QWORD *)&v121 = sub_33FC130(a3, 404, (__int64)&v130, (unsigned int)v132, v133, v35, v16, v86, v94, v124);
    *((_QWORD *)&v95 + 1) = v18;
    *(_QWORD *)&v95 = v17;
    *((_QWORD *)&v121 + 1) = v36 | *((_QWORD *)&v16 + 1) & 0xFFFFFFFF00000000LL;
    v128 = sub_33FC130(a3, 396, (__int64)&v130, (unsigned int)v132, v133, v37, v121, v110, v95, v124);
    v109 = v38;
    *(_QWORD *)&v39 = sub_3400BD0((__int64)a3, 2, (__int64)&v130, v119, v120, 0, (__m128i)v16, 0);
    *((_QWORD *)&v87 + 1) = v18;
    *(_QWORD *)&v87 = v17;
    *(_QWORD *)&v41 = sub_33FC130(a3, 398, (__int64)&v130, (unsigned int)v132, v133, v40, v121, v39, v87, v124);
    *((_QWORD *)&v96 + 1) = v18;
    *(_QWORD *)&v96 = v17;
    v127 = sub_33FC130(a3, 396, (__int64)&v130, (unsigned int)v132, v133, v42, v41, v110, v96, v124);
    *((_QWORD *)&v97 + 1) = v18;
    *(_QWORD *)&v97 = v17;
    *((_QWORD *)&v88 + 1) = v43;
    *(_QWORD *)&v88 = v127;
    *((_QWORD *)&v81 + 1) = v109;
    *(_QWORD *)&v81 = v128;
    v45 = sub_33FC130(a3, 395, (__int64)&v130, (unsigned int)v132, v133, v44, v81, v88, v97, v124);
    v105 = v46;
    v111 = v45;
    *(_QWORD *)&v47 = sub_3400BD0((__int64)a3, 4, (__int64)&v130, v119, v120, 0, (__m128i)v16, 0);
    *((_QWORD *)&v89 + 1) = v18;
    *(_QWORD *)&v122 = v111;
    *(_QWORD *)&v89 = v17;
    *((_QWORD *)&v122 + 1) = v105 | *((_QWORD *)&v121 + 1) & 0xFFFFFFFF00000000LL;
    v126 = sub_33FC130(
             a3,
             398,
             (__int64)&v130,
             (unsigned int)v132,
             v133,
             v48,
             __PAIR128__(*((unsigned __int64 *)&v122 + 1), (unsigned __int64)v111),
             v47,
             v89,
             v124);
    *((_QWORD *)&v98 + 1) = v18;
    *(_QWORD *)&v98 = v17;
    *((_QWORD *)&v90 + 1) = v49;
    *(_QWORD *)&v90 = v126;
    v125 = sub_33FC130(a3, 395, (__int64)&v130, (unsigned int)v132, v133, v50, v122, v90, v98, v124);
    *((_QWORD *)&v99 + 1) = v18;
    *(_QWORD *)&v99 = v17;
    *((_QWORD *)&v82 + 1) = v51;
    *(_QWORD *)&v82 = v125;
    v53 = sub_33FC130(a3, 396, (__int64)&v130, (unsigned int)v132, v133, v52, v82, v101, v99, v124);
    if ( (unsigned int)v20 > 8 )
    {
      v102 = v54;
      v55 = a3[8];
      v106 = v53;
      v56 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
      if ( v56 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v138, a1, v55, v132, v133);
        v57 = v139;
        v58 = v106;
        v59 = v102;
      }
      else
      {
        v57 = v56(a1, v55, v132, v133);
        v59 = v102;
        v58 = v106;
      }
      v60 = 1;
      if ( (v57 == 1 || v57 && (v60 = v57, *(_QWORD *)(a1 + 8LL * v57 + 112)))
        && ((v61 = *(_BYTE *)(a1 + 500 * v60 + 6813), v61 <= 1u) || v61 == 4) )
      {
        v103 = v59;
        v107 = v58;
        v135 = 8;
        v134 = 1;
        sub_C47700((__int64)&v138, v20, (__int64)&v134);
        v74 = sub_34007B0((__int64)a3, (__int64)&v138, (__int64)&v130, v132, v133, 0, (__m128i)v16, 0);
        v76 = v107;
        v77 = v74;
        v78 = v75;
        v79 = v103;
        if ( (unsigned int)v139 > 0x40 && v138 )
        {
          v114 = v74;
          v116 = v75;
          j_j___libc_free_0_0(v138);
          v79 = v103;
          v76 = v107;
          v77 = v114;
          v78 = v116;
        }
        if ( v135 > 0x40 && v134 )
        {
          v108 = v79;
          v118 = v76;
          v115 = v77;
          v117 = v78;
          j_j___libc_free_0_0(v134);
          v79 = v108;
          v76 = v118;
          v77 = v115;
          v78 = v117;
        }
        *((_QWORD *)&v100 + 1) = v18;
        *(_QWORD *)&v100 = v17;
        *((_QWORD *)&v92 + 1) = v78;
        *(_QWORD *)&v92 = v77;
        *(_QWORD *)&v112 = sub_33FC130(
                             a3,
                             399,
                             (__int64)&v130,
                             (unsigned int)v132,
                             v133,
                             v78,
                             __PAIR128__(v79 | *((_QWORD *)&v122 + 1) & 0xFFFFFFFF00000000LL, (unsigned __int64)v76),
                             v92,
                             v100,
                             v124);
        *((_QWORD *)&v112 + 1) = v80;
      }
      else
      {
        *(_QWORD *)&v123 = v17;
        *((_QWORD *)&v123 + 1) = v18;
        v62 = 8;
        v63 = a3;
        v64 = v58;
        v65 = v59;
        do
        {
          v113 = v62;
          *(_QWORD *)&v66 = sub_3400E40((__int64)v63, v62, v132, v133, (__int64)&v130, (__m128i)v16);
          *((_QWORD *)&v83 + 1) = v65;
          *(_QWORD *)&v83 = v64;
          *(_QWORD *)&v68 = sub_33FC130(v63, 402, (__int64)&v130, (unsigned int)v132, v133, v67, v83, v66, v123, v124);
          *((_QWORD *)&v84 + 1) = v65;
          *(_QWORD *)&v84 = v64;
          v64 = sub_33FC130(v63, 395, (__int64)&v130, (unsigned int)v132, v133, v69, v84, v68, v123, v124);
          v62 = 2 * v113;
          v71 = v70 | v65 & 0xFFFFFFFF00000000LL;
          v65 = v71;
        }
        while ( (unsigned int)v20 > 2 * v113 );
        *((_QWORD *)&v112 + 1) = v71;
        *(_QWORD *)&v112 = v64;
        a3 = v63;
      }
      *(_QWORD *)&v72 = sub_3400BD0(
                          (__int64)a3,
                          (unsigned int)(v20 - 8),
                          (__int64)&v130,
                          v119,
                          v120,
                          0,
                          (__m128i)v16,
                          0);
      *((_QWORD *)&v91 + 1) = v18;
      *(_QWORD *)&v91 = v17;
      v22 = sub_33FC130(a3, 398, (__int64)&v130, (unsigned int)v132, v133, v73, v112, v72, v91, v124);
    }
    else
    {
      v22 = v53;
    }
  }
  if ( v130 )
    sub_B91220((__int64)&v130, v130);
  return v22;
}
