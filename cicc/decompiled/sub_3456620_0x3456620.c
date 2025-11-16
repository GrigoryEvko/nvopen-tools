// Function: sub_3456620
// Address: 0x3456620
//
unsigned __int8 *__fastcall sub_3456620(__int64 a1, __int64 a2, _QWORD *a3)
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
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int8 *v20; // r12
  unsigned __int8 *v22; // r14
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int128 v27; // rax
  __int64 v28; // r9
  __int128 v29; // rax
  __int64 v30; // r9
  __int128 v31; // rax
  __int64 v32; // r9
  unsigned __int8 *v33; // r15
  unsigned int v34; // edx
  unsigned int v35; // r14d
  __int128 v36; // rax
  __int64 v37; // r9
  __int128 v38; // rax
  __int64 v39; // r9
  unsigned __int8 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r15
  unsigned __int8 *v43; // r14
  __int64 v44; // r9
  __int128 v45; // rax
  __int64 v46; // r9
  unsigned __int8 *v47; // r15
  int v48; // edx
  __int128 v49; // rax
  unsigned __int8 *v50; // r14
  __int64 v51; // r15
  __int64 v52; // r9
  __int128 v53; // rax
  __int64 v54; // r9
  __int128 v55; // rax
  __int64 v56; // r9
  unsigned __int8 *v57; // rax
  unsigned int v58; // edx
  unsigned __int8 *v59; // r10
  unsigned int v60; // r11d
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 (__fastcall *v65)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int16 v66; // ax
  unsigned __int8 *v67; // r10
  unsigned int v68; // r11d
  __int64 v69; // rdx
  unsigned __int8 v70; // al
  unsigned __int8 *v71; // r14
  unsigned int v72; // ebx
  unsigned __int64 v73; // r15
  __int64 v74; // rsi
  __int128 v75; // rax
  __int64 v76; // r9
  __int128 v77; // rax
  __int64 v78; // r9
  unsigned int v79; // edx
  __int128 v80; // rax
  __int64 v81; // r9
  __int128 v82; // rax
  __int64 v83; // r9
  unsigned __int8 *v84; // r10
  unsigned int v85; // r11d
  unsigned int v86; // edx
  unsigned __int8 *v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r15
  unsigned __int8 *v90; // r14
  __int128 v91; // rax
  __int64 v92; // r9
  __int128 v93; // rax
  __int64 v94; // r9
  __int128 v95; // rax
  __int64 v96; // r9
  bool v97; // al
  __int128 v98; // [rsp-20h] [rbp-140h]
  __int128 v99; // [rsp-20h] [rbp-140h]
  __int128 v100; // [rsp-10h] [rbp-130h]
  __int128 v101; // [rsp-10h] [rbp-130h]
  __int128 v102; // [rsp-10h] [rbp-130h]
  __int128 v103; // [rsp-10h] [rbp-130h]
  __int128 v104; // [rsp+0h] [rbp-120h]
  __int128 v105; // [rsp+0h] [rbp-120h]
  __int128 v106; // [rsp+0h] [rbp-120h]
  __int128 v107; // [rsp+10h] [rbp-110h]
  unsigned int v108; // [rsp+10h] [rbp-110h]
  unsigned int v109; // [rsp+10h] [rbp-110h]
  unsigned int v110; // [rsp+10h] [rbp-110h]
  __int128 v111; // [rsp+20h] [rbp-100h]
  unsigned __int8 *v112; // [rsp+20h] [rbp-100h]
  unsigned __int8 *v113; // [rsp+20h] [rbp-100h]
  unsigned int v114; // [rsp+20h] [rbp-100h]
  unsigned __int8 *v115; // [rsp+20h] [rbp-100h]
  __int128 v116; // [rsp+30h] [rbp-F0h]
  __int128 v117; // [rsp+30h] [rbp-F0h]
  unsigned int v118; // [rsp+30h] [rbp-F0h]
  unsigned __int8 *v119; // [rsp+40h] [rbp-E0h]
  unsigned int v120; // [rsp+48h] [rbp-D8h]
  __int64 v121; // [rsp+50h] [rbp-D0h]
  unsigned int v122; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *v123; // [rsp+58h] [rbp-C8h]
  __int128 v124; // [rsp+60h] [rbp-C0h]
  __int128 v125; // [rsp+60h] [rbp-C0h]
  __int64 v126; // [rsp+68h] [rbp-B8h]
  __int64 v127; // [rsp+90h] [rbp-90h] BYREF
  int v128; // [rsp+98h] [rbp-88h]
  __int64 v129; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v130; // [rsp+A8h] [rbp-78h]
  unsigned __int64 v131; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v132; // [rsp+B8h] [rbp-68h]
  __int64 v133; // [rsp+C0h] [rbp-60h]
  __int64 v134; // [rsp+C8h] [rbp-58h]
  unsigned __int64 v135; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v136; // [rsp+D8h] [rbp-48h]

  v6 = *(_QWORD *)(a2 + 80);
  v127 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v127, v6, 1);
  v7 = (__int64 *)a3[5];
  v128 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v129) = v9;
  v130 = v10;
  v11 = sub_2E79000(v7);
  v12 = (unsigned int)v129;
  v120 = sub_2FE6750(a1, (unsigned int)v129, v130, v11);
  v13 = *(const __m128i **)(a2 + 40);
  v14 = v129;
  v121 = v15;
  v16 = (__int128)_mm_loadu_si128(v13);
  if ( (_WORD)v129 )
  {
    if ( (unsigned __int16)(v129 - 17) <= 0xD3u )
    {
      v136 = 0;
      v14 = word_4456580[(unsigned __int16)v129 - 1];
      LOWORD(v135) = v14;
      if ( !v14 )
        goto LABEL_7;
      goto LABEL_38;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v129) )
  {
LABEL_5:
    v17 = v130;
    goto LABEL_6;
  }
  v14 = sub_3009970((__int64)&v129, v12, v61, v62, v63);
LABEL_6:
  LOWORD(v135) = v14;
  v136 = v17;
  if ( !v14 )
  {
LABEL_7:
    v18 = sub_3007260((__int64)&v135);
    v133 = v18;
    v134 = v19;
    goto LABEL_8;
  }
LABEL_38:
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    BUG();
  v18 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
LABEL_8:
  v122 = v18;
  if ( (unsigned int)v18 > 0x80 || (v18 & 7) != 0 )
    goto LABEL_10;
  if ( (_WORD)v129 )
  {
    if ( (unsigned __int16)(v129 - 17) > 0xD3u )
      goto LABEL_17;
  }
  else if ( !sub_30070B0((__int64)&v129) )
  {
    goto LABEL_17;
  }
  if ( !sub_34447D0(a1, (unsigned int)v129, v130) )
  {
LABEL_10:
    v20 = 0;
    goto LABEL_11;
  }
LABEL_17:
  v132 = 8;
  v131 = 85;
  sub_C47700((__int64)&v135, v122, (__int64)&v131);
  v22 = sub_34007B0((__int64)a3, (__int64)&v135, (__int64)&v127, v129, v130, 0, (__m128i)v16, 0);
  v24 = v23;
  if ( (unsigned int)v136 > 0x40 && v135 )
    j_j___libc_free_0_0(v135);
  if ( v132 > 0x40 && v131 )
    j_j___libc_free_0_0(v131);
  v132 = 8;
  v131 = 51;
  sub_C47700((__int64)&v135, v122, (__int64)&v131);
  *(_QWORD *)&v111 = sub_34007B0((__int64)a3, (__int64)&v135, (__int64)&v127, v129, v130, 0, (__m128i)v16, 0);
  *((_QWORD *)&v111 + 1) = v25;
  if ( (unsigned int)v136 > 0x40 && v135 )
    j_j___libc_free_0_0(v135);
  if ( v132 > 0x40 && v131 )
    j_j___libc_free_0_0(v131);
  v132 = 8;
  v131 = 15;
  sub_C47700((__int64)&v135, v122, (__int64)&v131);
  *(_QWORD *)&v107 = sub_34007B0((__int64)a3, (__int64)&v135, (__int64)&v127, v129, v130, 0, (__m128i)v16, 0);
  *((_QWORD *)&v107 + 1) = v26;
  if ( (unsigned int)v136 > 0x40 && v135 )
    j_j___libc_free_0_0(v135);
  if ( v132 > 0x40 && v131 )
    j_j___libc_free_0_0(v131);
  *(_QWORD *)&v27 = sub_3400BD0((__int64)a3, 1, (__int64)&v127, v120, v121, 0, (__m128i)v16, 0);
  *(_QWORD *)&v29 = sub_3406EB0(a3, 0xC0u, (__int64)&v127, (unsigned int)v129, v130, v28, v16, v27);
  *((_QWORD *)&v104 + 1) = v24;
  *(_QWORD *)&v104 = v22;
  *(_QWORD *)&v31 = sub_3406EB0(a3, 0xBAu, (__int64)&v127, (unsigned int)v129, v130, v30, v29, v104);
  v33 = sub_3406EB0(a3, 0x39u, (__int64)&v127, (unsigned int)v129, v130, v32, v16, v31);
  v35 = v34;
  *(_QWORD *)&v36 = sub_3400BD0((__int64)a3, 2, (__int64)&v127, v120, v121, 0, (__m128i)v16, 0);
  *(_QWORD *)&v124 = v33;
  *((_QWORD *)&v124 + 1) = v35 | *((_QWORD *)&v16 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v38 = sub_3406EB0(
                      a3,
                      0xC0u,
                      (__int64)&v127,
                      (unsigned int)v129,
                      v130,
                      v37,
                      __PAIR128__(*((unsigned __int64 *)&v124 + 1), (unsigned __int64)v33),
                      v36);
  v40 = sub_3406EB0(a3, 0xBAu, (__int64)&v127, (unsigned int)v129, v130, v39, v38, v111);
  v42 = v41;
  v43 = v40;
  *(_QWORD *)&v45 = sub_3406EB0(a3, 0xBAu, (__int64)&v127, (unsigned int)v129, v130, v44, v124, v111);
  *((_QWORD *)&v105 + 1) = v42;
  *(_QWORD *)&v105 = v43;
  v47 = sub_3406EB0(a3, 0x38u, (__int64)&v127, (unsigned int)v129, v130, v46, v45, v105);
  LODWORD(v43) = v48;
  *(_QWORD *)&v49 = sub_3400BD0((__int64)a3, 4, (__int64)&v127, v120, v121, 0, (__m128i)v16, 0);
  v100 = v49;
  *((_QWORD *)&v49 + 1) = (unsigned int)v43 | *((_QWORD *)&v124 + 1) & 0xFFFFFFFF00000000LL;
  v50 = v47;
  v126 = *((_QWORD *)&v49 + 1);
  v51 = *((_QWORD *)&v49 + 1);
  *((_QWORD *)&v98 + 1) = *((_QWORD *)&v49 + 1);
  *(_QWORD *)&v98 = v50;
  *(_QWORD *)&v53 = sub_3406EB0(a3, 0xC0u, (__int64)&v127, (unsigned int)v129, v130, v52, v98, v100);
  *((_QWORD *)&v101 + 1) = v51;
  *(_QWORD *)&v101 = v50;
  *(_QWORD *)&v55 = sub_3406EB0(a3, 0x38u, (__int64)&v127, (unsigned int)v129, v130, v54, v101, v53);
  v57 = sub_3406EB0(a3, 0xBAu, (__int64)&v127, (unsigned int)v129, v130, v56, v55, v107);
  v59 = v57;
  v60 = v58;
  if ( v122 <= 8 )
  {
    v20 = v57;
    goto LABEL_11;
  }
  if ( v122 != 16 )
    goto LABEL_46;
  if ( (_WORD)v129 )
  {
    if ( (unsigned __int16)(v129 - 17) > 0xD3u )
    {
LABEL_66:
      v118 = v60;
      v123 = v59;
      v87 = sub_3400BD0((__int64)a3, 255, (__int64)&v127, (unsigned int)v129, v130, 0, (__m128i)v16, 0);
      v89 = v88;
      v90 = v87;
      *(_QWORD *)&v91 = sub_3400BD0((__int64)a3, 8, (__int64)&v127, v120, v121, 0, (__m128i)v16, 0);
      *(_QWORD *)&v125 = v123;
      *((_QWORD *)&v125 + 1) = v118 | v126 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v93 = sub_3406EB0(
                          a3,
                          0xC0u,
                          (__int64)&v127,
                          (unsigned int)v129,
                          v130,
                          v92,
                          __PAIR128__(*((unsigned __int64 *)&v125 + 1), (unsigned __int64)v123),
                          v91);
      *(_QWORD *)&v95 = sub_3406EB0(a3, 0x38u, (__int64)&v127, (unsigned int)v129, v130, v94, v125, v93);
      *((_QWORD *)&v106 + 1) = v89;
      *(_QWORD *)&v106 = v90;
      v20 = sub_3406EB0(a3, 0xBAu, (__int64)&v127, (unsigned int)v129, v130, v96, v95, v106);
      goto LABEL_11;
    }
  }
  else
  {
    v115 = v57;
    v110 = v58;
    v97 = sub_30070B0((__int64)&v129);
    v59 = v115;
    v60 = v110;
    if ( !v97 )
      goto LABEL_66;
  }
LABEL_46:
  v108 = v60;
  v64 = a3[8];
  v112 = v59;
  v65 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
  if ( v65 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v135, a1, v64, v129, v130);
    v66 = v136;
    v67 = v112;
    v68 = v108;
  }
  else
  {
    v66 = v65(a1, v64, v129, v130);
    v68 = v108;
    v67 = v112;
  }
  v69 = 1;
  if ( (v66 == 1 || v66 && (v69 = v66, *(_QWORD *)(a1 + 8LL * v66 + 112)))
    && ((v70 = *(_BYTE *)(a1 + 500 * v69 + 6472), v70 <= 1u) || v70 == 4) )
  {
    v109 = v68;
    v113 = v67;
    v132 = 8;
    v131 = 1;
    sub_C47700((__int64)&v135, v122, (__int64)&v131);
    *(_QWORD *)&v82 = sub_34007B0((__int64)a3, (__int64)&v135, (__int64)&v127, v129, v130, 0, (__m128i)v16, 0);
    v84 = v113;
    v85 = v109;
    if ( (unsigned int)v136 > 0x40 && v135 )
    {
      v116 = v82;
      j_j___libc_free_0_0(v135);
      v85 = v109;
      v84 = v113;
      v82 = v116;
    }
    if ( v132 > 0x40 && v131 )
    {
      v114 = v85;
      v119 = v84;
      v117 = v82;
      j_j___libc_free_0_0(v131);
      v85 = v114;
      v84 = v119;
      v82 = v117;
    }
    v71 = sub_3406EB0(
            a3,
            0x3Au,
            (__int64)&v127,
            (unsigned int)v129,
            v130,
            v83,
            __PAIR128__(v85 | v126 & 0xFFFFFFFF00000000LL, (unsigned __int64)v84),
            v82);
    v73 = v86;
  }
  else
  {
    v71 = v67;
    v72 = 8;
    v73 = v68;
    do
    {
      v74 = v72;
      v72 *= 2;
      *(_QWORD *)&v75 = sub_3400E40((__int64)a3, v74, v129, v130, (__int64)&v127, (__m128i)v16);
      *((_QWORD *)&v102 + 1) = v73;
      *(_QWORD *)&v102 = v71;
      *(_QWORD *)&v77 = sub_3406EB0(a3, 0xBEu, (__int64)&v127, (unsigned int)v129, v130, v76, v102, v75);
      *((_QWORD *)&v103 + 1) = v73;
      *(_QWORD *)&v103 = v71;
      v71 = sub_3406EB0(a3, 0x38u, (__int64)&v127, (unsigned int)v129, v130, v78, v103, v77);
      v73 = v79 | v73 & 0xFFFFFFFF00000000LL;
    }
    while ( v122 > v72 );
  }
  *(_QWORD *)&v80 = sub_3400BD0((__int64)a3, v122 - 8, (__int64)&v127, v120, v121, 0, (__m128i)v16, 0);
  *((_QWORD *)&v99 + 1) = v73;
  *(_QWORD *)&v99 = v71;
  v20 = sub_3406EB0(a3, 0xC0u, (__int64)&v127, (unsigned int)v129, v130, v81, v99, v80);
LABEL_11:
  if ( v127 )
    sub_B91220((__int64)&v127, v127);
  return v20;
}
