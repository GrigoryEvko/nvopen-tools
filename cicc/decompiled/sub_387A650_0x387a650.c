// Function: sub_387A650
// Address: 0x387a650
//
_QWORD *__fastcall sub_387A650(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // r13
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  _QWORD *v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rax
  _DWORD *v28; // rdi
  _DWORD *v29; // rsi
  __int64 v30; // r10
  __int64 v31; // r11
  __int64 v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 ***v35; // r14
  double v36; // xmm4_8
  double v37; // xmm5_8
  __int64 *v38; // rax
  _QWORD *v39; // r15
  unsigned int v40; // r14d
  unsigned int v41; // eax
  bool v42; // cc
  _QWORD *v43; // r14
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r14
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 v52; // r14
  __int64 v53; // r15
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 v56; // rax
  __int64 v57; // r14
  double v58; // xmm4_8
  double v59; // xmm5_8
  __int64 v60; // rax
  _QWORD *v61; // rdx
  __int16 v62; // si
  _QWORD *v63; // r15
  _QWORD *v64; // rax
  _QWORD *v65; // r14
  unsigned __int64 v66; // r15
  _QWORD *v67; // r13
  _QWORD *v68; // rbx
  _QWORD *v69; // r12
  unsigned __int64 v70; // rdi
  _QWORD *v72; // rax
  __int64 v73; // rax
  __int64 *v74; // rax
  __int64 v75; // rax
  __int64 v76; // r15
  _QWORD *v77; // rdx
  unsigned __int8 v78; // al
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rdi
  unsigned __int64 *v82; // r14
  __int64 v83; // rax
  unsigned __int64 v84; // rcx
  _QWORD *v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdi
  __int64 v92; // rsi
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 *v95; // rbx
  __int64 v96; // rax
  __int64 v97; // rcx
  unsigned int v98; // esi
  int v99; // eax
  __int64 v100; // [rsp+18h] [rbp-1F8h]
  __int64 *v101; // [rsp+18h] [rbp-1F8h]
  _QWORD *v103; // [rsp+30h] [rbp-1E0h]
  __int64 v104; // [rsp+38h] [rbp-1D8h]
  __int64 ***v105; // [rsp+40h] [rbp-1D0h]
  unsigned int v106; // [rsp+48h] [rbp-1C8h]
  unsigned int v107; // [rsp+50h] [rbp-1C0h]
  __int64 **v108; // [rsp+58h] [rbp-1B8h]
  __int64 *v109; // [rsp+60h] [rbp-1B0h]
  _QWORD *v110; // [rsp+68h] [rbp-1A8h]
  __int64 v111; // [rsp+68h] [rbp-1A8h]
  __int64 ***v112; // [rsp+70h] [rbp-1A0h]
  __int64 v113; // [rsp+78h] [rbp-198h]
  __int64 **v114; // [rsp+78h] [rbp-198h]
  __int64 v115; // [rsp+80h] [rbp-190h]
  __int64 v116; // [rsp+80h] [rbp-190h]
  __int64 v117; // [rsp+80h] [rbp-190h]
  __int64 *v118; // [rsp+80h] [rbp-190h]
  int v119; // [rsp+80h] [rbp-190h]
  _QWORD *v121; // [rsp+88h] [rbp-188h]
  unsigned __int64 v122; // [rsp+90h] [rbp-180h] BYREF
  unsigned int v123; // [rsp+98h] [rbp-178h]
  __int64 v124[2]; // [rsp+A0h] [rbp-170h] BYREF
  __int16 v125; // [rsp+B0h] [rbp-160h]
  __int64 v126[2]; // [rsp+C0h] [rbp-150h] BYREF
  __int16 v127; // [rsp+D0h] [rbp-140h]
  char *v128; // [rsp+E0h] [rbp-130h] BYREF
  unsigned int v129; // [rsp+E8h] [rbp-128h]
  __int16 v130; // [rsp+F0h] [rbp-120h]
  _QWORD v131[5]; // [rsp+100h] [rbp-110h] BYREF
  char *v132; // [rsp+128h] [rbp-E8h]
  char v133; // [rsp+138h] [rbp-D8h] BYREF
  _QWORD *v134; // [rsp+1C0h] [rbp-50h]
  unsigned int v135; // [rsp+1D0h] [rbp-40h]

  sub_14585E0((__int64)v131);
  v115 = sub_14959A0((_QWORD *)*a1, *(_QWORD *)(a2 + 48), (__int64)v131, a5, a6);
  v13 = sub_13A5BC0((_QWORD *)a2, *a1);
  v14 = **(_QWORD **)(a2 + 32);
  v15 = sub_1456040(v14);
  v16 = *a1;
  v17 = v15;
  v113 = v115;
  v18 = sub_1456040(v115);
  v19 = v17;
  v116 = v17;
  v107 = sub_1456C90(v16, v18);
  v20 = a1 + 33;
  v106 = sub_1456C90(*a1, v19);
  v21 = (_QWORD *)sub_16498A0(a3);
  v108 = (__int64 **)sub_1644900(v21, v107);
  sub_17050D0(a1 + 33, a3);
  v109 = (__int64 *)sub_38767A0(a1, v113, v108, a3, (__m128)a5, *(double *)a6.m128i_i64, a7, a8, v22, v23, a11, a12);
  LODWORD(v16) = sub_1456C90(*a1, v116);
  v24 = (_QWORD *)sub_16498A0(a3);
  v114 = (__int64 **)sub_1644900(v24, v16);
  if ( *(_BYTE *)(v116 + 8) == 15 )
  {
    v27 = a1[1];
    v28 = *(_DWORD **)(v27 + 408);
    v29 = &v28[*(unsigned int *)(v27 + 416)];
    LODWORD(v128) = *(_DWORD *)(v116 + 8) >> 8;
    if ( v29 == sub_386EF90(v28, (__int64)v29, (int *)&v128) )
      v30 = v31;
    v117 = v30;
  }
  else
  {
    v117 = (__int64)v114;
  }
  v105 = sub_38767A0(a1, v13, v114, a3, (__m128)a5, *(double *)a6.m128i_i64, a7, a8, v25, v26, a11, a12);
  v32 = sub_1480620(*a1, v13, 0);
  v35 = sub_38767A0(a1, v32, v114, a3, (__m128)a5, *(double *)a6.m128i_i64, a7, a8, v33, v34, a11, a12);
  v112 = sub_38767A0(a1, v14, (__int64 **)v117, a3, (__m128)a5, *(double *)a6.m128i_i64, a7, a8, v36, v37, a11, a12);
  v129 = v106;
  if ( v106 > 0x40 )
    sub_16A4EF0((__int64)&v128, 0, 0);
  else
    v128 = 0;
  v38 = (__int64 *)sub_16498A0(a3);
  v104 = sub_159C0E0(v38, (__int64)&v128);
  if ( v129 > 0x40 && v128 )
    j_j___libc_free_0_0((unsigned __int64)v128);
  sub_17050D0(a1 + 33, a3);
  v130 = 257;
  v103 = sub_17B5310(a1 + 33, 40, (__int64)v105, v104, (__int64 *)&v128);
  v130 = 257;
  v39 = sub_3871A40(a1 + 33, (__int64)v103, (__int64)v35, (__int64)v105, (__int64 *)&v128, 0);
  v130 = 257;
  v40 = sub_16431D0(*v109);
  v41 = sub_16431D0((__int64)v114);
  v42 = v40 <= v41;
  if ( v40 < v41 )
  {
    v43 = sub_38723F0(a1 + 33, 37, (__int64)v109, v114, (__int64 *)&v128);
  }
  else
  {
    v43 = v109;
    if ( !v42 )
      v43 = sub_38723F0(a1 + 33, 36, (__int64)v109, v114, (__int64 *)&v128);
  }
  v128 = (char *)v114;
  v44 = (__int64 *)sub_15F2050(a3);
  v45 = sub_15E26F0(v44, 210, (__int64 *)&v128, 1);
  v126[1] = (__int64)v43;
  v128 = "mul";
  v130 = 259;
  v126[0] = (__int64)v39;
  v46 = sub_17B5EF0((__int64)(a1 + 33), *(_QWORD *)(v45 + 24), v45, v126, 2, (__int64 *)&v128, 0);
  v47 = v46;
  v127 = 259;
  v126[0] = (__int64)"mul.result";
  LODWORD(v124[0]) = 0;
  if ( *(_BYTE *)(v46 + 16) > 0x10u )
  {
    v130 = 257;
    v85 = sub_1648A60(88, 1u);
    v48 = (__int64)v85;
    if ( v85 )
    {
      v111 = (__int64)v85;
      v86 = sub_15FB2A0(*(_QWORD *)v47, (unsigned int *)v124, 1);
      sub_15F1EA0(v48, v86, 62, v48 - 24, 1, 0);
      sub_1593B40((_QWORD *)(v48 - 24), v47);
      *(_QWORD *)(v48 + 56) = v48 + 72;
      *(_QWORD *)(v48 + 64) = 0x400000000LL;
      sub_15FB110(v48, v124, 1, (__int64)&v128);
    }
    else
    {
      v111 = 0;
    }
    v87 = a1[34];
    if ( v87 )
    {
      v101 = (__int64 *)a1[35];
      sub_157E9D0(v87 + 40, v48);
      v88 = *v101;
      v89 = *(_QWORD *)(v48 + 24) & 7LL;
      *(_QWORD *)(v48 + 32) = v101;
      v88 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v48 + 24) = v88 | v89;
      *(_QWORD *)(v88 + 8) = v48 + 24;
      *v101 = *v101 & 7 | (v48 + 24);
    }
    sub_164B780(v111, v126);
    sub_12A86E0(a1 + 33, v48);
  }
  else
  {
    v48 = sub_15A3AE0((_QWORD *)v46, (unsigned int *)v124, 1, 0);
    v49 = sub_14DBA30(v48, a1[41], 0);
    if ( v49 )
      v48 = v49;
  }
  v126[0] = (__int64)"mul.overflow";
  v127 = 259;
  LODWORD(v124[0]) = 1;
  if ( *(_BYTE *)(v47 + 16) > 0x10u )
  {
    v130 = 257;
    v110 = sub_1648A60(88, 1u);
    if ( v110 )
    {
      v100 = (__int64)v110;
      v80 = sub_15FB2A0(*(_QWORD *)v47, (unsigned int *)v124, 1);
      sub_15F1EA0((__int64)v110, v80, 62, (__int64)(v110 - 3), 1, 0);
      sub_1593B40(v110 - 3, v47);
      v110[7] = v110 + 9;
      v110[8] = 0x400000000LL;
      sub_15FB110((__int64)v110, v124, 1, (__int64)&v128);
    }
    else
    {
      v100 = 0;
    }
    v81 = a1[34];
    if ( v81 )
    {
      v82 = (unsigned __int64 *)a1[35];
      sub_157E9D0(v81 + 40, (__int64)v110);
      v83 = v110[3];
      v84 = *v82 & 0xFFFFFFFFFFFFFFF8LL;
      v110[4] = v82;
      v110[3] = v84 | v83 & 7;
      *(_QWORD *)(v84 + 8) = v110 + 3;
      *v82 = *v82 & 7 | (unsigned __int64)(v110 + 3);
    }
    sub_164B780(v100, v126);
    sub_12A86E0(a1 + 33, (__int64)v110);
  }
  else
  {
    v50 = sub_15A3AE0((_QWORD *)v47, (unsigned int *)v124, 1, 0);
    v51 = sub_14DBA30(v50, a1[41], 0);
    if ( !v51 )
      v51 = v50;
    v110 = (_QWORD *)v51;
  }
  if ( *(_BYTE *)(v117 + 8) == 15 )
  {
    v52 = sub_146F1B0(*a1, v48);
    v53 = sub_1480620(*a1, v52, 0);
    v130 = 257;
    v56 = sub_3878B90(a1, v52, v117, v114, v112, a5, a6, a7, a8, v54, v55, a11, a12);
    v57 = (__int64)sub_38723F0(a1 + 33, 47, v56, (__int64 **)v117, (__int64 *)&v128);
    v130 = 257;
    v60 = sub_3878B90(a1, v53, v117, v114, v112, a5, a6, a7, a8, v58, v59, a11, a12);
    v61 = sub_38723F0(a1 + 33, 47, v60, (__int64 **)v117, (__int64 *)&v128);
  }
  else
  {
    v127 = 257;
    if ( *((_BYTE *)v112 + 16) > 0x10u || *(_BYTE *)(v48 + 16) > 0x10u )
    {
      v130 = 257;
      v90 = sub_15FB440(11, (__int64 *)v112, v48, (__int64)&v128, 0);
      v91 = a1[34];
      v57 = v90;
      if ( v91 )
      {
        v118 = (__int64 *)a1[35];
        sub_157E9D0(v91 + 40, v90);
        v92 = *v118;
        v93 = *(_QWORD *)(v57 + 24) & 7LL;
        *(_QWORD *)(v57 + 32) = v118;
        v92 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v57 + 24) = v92 | v93;
        *(_QWORD *)(v92 + 8) = v57 + 24;
        *v118 = *v118 & 7 | (v57 + 24);
      }
      sub_164B780(v57, v126);
      sub_12A86E0(a1 + 33, v57);
    }
    else
    {
      v57 = sub_15A2B30((__int64 *)v112, v48, 0, 0, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7);
      v73 = sub_14DBA30(v57, a1[41], 0);
      if ( v73 )
        v57 = v73;
    }
    v130 = 257;
    v61 = sub_38718D0(
            a1 + 33,
            (__int64)v112,
            v48,
            (__int64 *)&v128,
            0,
            0,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            a7);
  }
  v130 = 257;
  if ( a4 )
  {
    v62 = 40;
    v63 = sub_17B5310(a1 + 33, 38, (__int64)v61, (__int64)v112, (__int64 *)&v128);
    v130 = 257;
  }
  else
  {
    v72 = sub_17B5310(a1 + 33, 34, (__int64)v61, (__int64)v112, (__int64 *)&v128);
    v62 = 36;
    v130 = 257;
    v63 = v72;
  }
  v64 = sub_17B5310(a1 + 33, v62, v57, (__int64)v112, (__int64 *)&v128);
  v130 = 257;
  v65 = sub_3871A40(a1 + 33, (__int64)v103, (__int64)v63, (__int64)v64, (__int64 *)&v128, 0);
  v66 = sub_1456C90(*a1, (__int64)v108);
  if ( v66 > sub_1456C90(*a1, (__int64)v114) )
  {
    v129 = v106;
    if ( v106 <= 0x40 )
      v128 = (char *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v106);
    else
      sub_16A4EF0((__int64)&v128, -1, 1);
    sub_16A5C50((__int64)&v122, (const void **)&v128, v107);
    if ( v129 > 0x40 && v128 )
      j_j___libc_free_0_0((unsigned __int64)v128);
    v130 = 257;
    v74 = (__int64 *)sub_16498A0(a3);
    v75 = sub_159C0E0(v74, (__int64)&v122);
    v76 = (__int64)sub_17B5310(a1 + 33, 34, (__int64)v109, v75, (__int64 *)&v128);
    v127 = 257;
    v125 = 257;
    v77 = sub_17B5310(a1 + 33, 33, (__int64)v105, v104, v124);
    v78 = *((_BYTE *)v77 + 16);
    if ( v78 > 0x10u )
      goto LABEL_71;
    if ( v78 != 13 )
    {
LABEL_51:
      if ( *(_BYTE *)(v76 + 16) <= 0x10u )
      {
        v76 = sub_15A2CF0((__int64 *)v76, (__int64)v77, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7);
        v79 = sub_14DBA30(v76, a1[41], 0);
        if ( v79 )
          v76 = v79;
        goto LABEL_54;
      }
LABEL_71:
      v130 = 257;
      v76 = sub_15FB440(26, (__int64 *)v76, (__int64)v77, (__int64)&v128, 0);
      v94 = a1[34];
      if ( v94 )
      {
        v95 = (__int64 *)a1[35];
        sub_157E9D0(v94 + 40, v76);
        v96 = *(_QWORD *)(v76 + 24);
        v97 = *v95;
        *(_QWORD *)(v76 + 32) = v95;
        v97 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v76 + 24) = v97 | v96 & 7;
        *(_QWORD *)(v97 + 8) = v76 + 24;
        *v95 = *v95 & 7 | (v76 + 24);
      }
      sub_164B780(v76, v126);
      sub_12A86E0(v20, v76);
      goto LABEL_54;
    }
    v98 = *((_DWORD *)v77 + 8);
    if ( v98 <= 0x40 )
    {
      if ( v77[3] != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v98) )
        goto LABEL_51;
    }
    else
    {
      v119 = *((_DWORD *)v77 + 8);
      v121 = v77;
      v99 = sub_16A58F0((__int64)(v77 + 3));
      v77 = v121;
      if ( v119 != v99 )
        goto LABEL_51;
    }
LABEL_54:
    v130 = 257;
    v65 = sub_17B51C0(v20, (__int64)v65, v76, (__int64 *)&v128, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7);
    if ( v123 > 0x40 && v122 )
      j_j___libc_free_0_0(v122);
  }
  v130 = 257;
  v67 = sub_17B51C0(
          v20,
          (__int64)v65,
          (__int64)v110,
          (__int64 *)&v128,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64,
          a7);
  v131[0] = &unk_49EC708;
  if ( v135 )
  {
    v68 = v134;
    v69 = &v134[7 * v135];
    do
    {
      if ( *v68 != -8 && *v68 != -16 )
      {
        v70 = v68[1];
        if ( (_QWORD *)v70 != v68 + 3 )
          _libc_free(v70);
      }
      v68 += 7;
    }
    while ( v69 != v68 );
  }
  j___libc_free_0((unsigned __int64)v134);
  if ( v132 != &v133 )
    _libc_free((unsigned __int64)v132);
  return v67;
}
