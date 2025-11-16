// Function: sub_20B7F50
// Address: 0x20b7f50
//
__int64 __fastcall sub_20B7F50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  unsigned int v7; // r15d
  __int64 v8; // r10
  __int64 v10; // rax
  char v11; // dl
  char *v12; // rax
  __int64 v13; // rsi
  unsigned __int8 v14; // r13
  const void **v15; // rax
  unsigned int v16; // r12d
  unsigned int v18; // eax
  __int64 v19; // r10
  unsigned int v20; // esi
  unsigned int v21; // eax
  const void **v22; // rdx
  const void **v23; // r14
  __int128 v24; // rax
  __int128 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // r10
  __int64 v29; // r8
  __int128 v30; // rax
  __int64 v31; // r10
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned int v38; // eax
  const void **v39; // rdx
  __int128 v40; // rax
  __int64 *v41; // rax
  unsigned __int64 v42; // rdx
  __int64 *v43; // rax
  unsigned __int64 v44; // rdx
  __int128 v45; // rax
  unsigned int v46; // eax
  const void **v47; // rdx
  __int128 v48; // rax
  __int64 *v49; // rax
  unsigned __int64 v50; // rdx
  __int64 *v51; // rax
  __int64 v52; // rdx
  unsigned int v53; // edx
  __int128 v54; // rax
  __int64 *v55; // rax
  unsigned __int64 v56; // rdx
  __int64 *v57; // rax
  __int64 v58; // rdx
  unsigned int v59; // edx
  int v60; // eax
  const void **v61; // rdx
  __int64 *v62; // rax
  __int64 v63; // rdx
  __int128 v64; // rax
  __int128 v65; // rax
  unsigned int v66; // eax
  const void **v67; // rdx
  __int64 *v68; // rax
  __int64 v69; // rdx
  __int128 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int128 v75; // rax
  __int64 v76; // r9
  unsigned int v77; // edx
  __int64 *v78; // rax
  unsigned __int64 v79; // rdx
  __int128 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // r15
  __int64 v85; // rbx
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int128 v90; // rax
  __int64 *v91; // rdi
  __int64 v92; // r9
  int v93; // edx
  __int128 v94; // [rsp-30h] [rbp-190h]
  __int128 v95; // [rsp-20h] [rbp-180h]
  __int64 v96; // [rsp+18h] [rbp-148h]
  __int64 v97; // [rsp+20h] [rbp-140h]
  __int64 v98; // [rsp+28h] [rbp-138h]
  __int128 v99; // [rsp+30h] [rbp-130h]
  __int64 v100; // [rsp+30h] [rbp-130h]
  __int128 v101; // [rsp+40h] [rbp-120h]
  const void **v102; // [rsp+40h] [rbp-120h]
  const void **v103; // [rsp+40h] [rbp-120h]
  unsigned int v104; // [rsp+40h] [rbp-120h]
  __int128 v105; // [rsp+50h] [rbp-110h]
  __int128 v106; // [rsp+50h] [rbp-110h]
  __int128 v107; // [rsp+50h] [rbp-110h]
  __int128 v108; // [rsp+60h] [rbp-100h]
  __int128 v109; // [rsp+60h] [rbp-100h]
  __int128 v110; // [rsp+70h] [rbp-F0h]
  __int128 v111; // [rsp+70h] [rbp-F0h]
  __int128 v112; // [rsp+80h] [rbp-E0h]
  unsigned int v113; // [rsp+90h] [rbp-D0h]
  __int128 v114; // [rsp+90h] [rbp-D0h]
  __int64 v115; // [rsp+98h] [rbp-C8h]
  __int64 v116; // [rsp+A0h] [rbp-C0h]
  __int64 v117; // [rsp+A0h] [rbp-C0h]
  __int64 v118; // [rsp+A0h] [rbp-C0h]
  __int64 v119; // [rsp+A0h] [rbp-C0h]
  __int64 v120; // [rsp+A0h] [rbp-C0h]
  __int64 v121; // [rsp+A0h] [rbp-C0h]
  __int64 v122; // [rsp+A0h] [rbp-C0h]
  __int128 v123; // [rsp+A0h] [rbp-C0h]
  unsigned __int64 v124; // [rsp+A8h] [rbp-B8h]
  __int64 v125; // [rsp+A8h] [rbp-B8h]
  unsigned __int64 v126; // [rsp+A8h] [rbp-B8h]
  const void **v128; // [rsp+B8h] [rbp-A8h]
  __int64 v129; // [rsp+B8h] [rbp-A8h]
  __int64 *v130; // [rsp+D0h] [rbp-90h]
  _BYTE v131[8]; // [rsp+100h] [rbp-60h] BYREF
  __int64 v132; // [rsp+108h] [rbp-58h]
  __int64 v133; // [rsp+110h] [rbp-50h] BYREF
  int v134; // [rsp+118h] [rbp-48h]
  __int64 v135; // [rsp+120h] [rbp-40h] BYREF
  unsigned int v136; // [rsp+128h] [rbp-38h]

  v8 = a2;
  v10 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v11 = *(_BYTE *)v10;
  v132 = *(_QWORD *)(v10 + 8);
  v12 = *(char **)(a2 + 40);
  v131[0] = v11;
  v13 = *(_QWORD *)(a2 + 72);
  v14 = *v12;
  v15 = (const void **)*((_QWORD *)v12 + 1);
  v133 = v13;
  v128 = v15;
  if ( v13 )
  {
    v116 = v8;
    sub_1623A60((__int64)&v133, v13, 2);
    v11 = v131[0];
    v8 = v116;
  }
  v117 = v8;
  v134 = *(_DWORD *)(v8 + 64);
  if ( v14 != 6 || v11 != 9 )
  {
    v16 = 0;
    goto LABEL_6;
  }
  v18 = sub_1F3E310(v131);
  v19 = v117;
  v20 = v18;
  if ( v18 == 32 )
  {
    LOBYTE(v21) = 5;
  }
  else if ( v18 > 0x20 )
  {
    if ( v18 == 64 )
    {
      LOBYTE(v21) = 6;
    }
    else
    {
      if ( v18 != 128 )
      {
LABEL_14:
        v21 = sub_1F58CC0(*(_QWORD **)(a4 + 48), v20);
        v19 = v117;
        v7 = v21;
        v23 = v22;
        goto LABEL_15;
      }
      LOBYTE(v21) = 7;
    }
  }
  else if ( v18 == 8 )
  {
    LOBYTE(v21) = 3;
  }
  else
  {
    LOBYTE(v21) = 4;
    if ( v20 != 16 )
    {
      LOBYTE(v21) = 2;
      if ( v20 != 1 )
        goto LABEL_14;
    }
  }
  v23 = 0;
LABEL_15:
  LOBYTE(v7) = v21;
  v118 = v19;
  *(_QWORD *)&v24 = sub_1D38BB0(a4, 2139095040, (__int64)&v133, v7, v23, 0, a5, a6, a7, 0);
  v108 = v24;
  *(_QWORD *)&v25 = sub_1D38BB0(a4, 23, (__int64)&v133, v7, v23, 0, a5, a6, a7, 0);
  v112 = v25;
  *(_QWORD *)&v105 = sub_1D38BB0(a4, 127, (__int64)&v133, v7, v23, 0, a5, a6, a7, 0);
  *((_QWORD *)&v105 + 1) = v26;
  if ( v131[0] )
    v27 = sub_1F3E310(v131);
  else
    v27 = sub_1F58D40((__int64)v131);
  v28 = v118;
  v136 = v27;
  v29 = 1LL << ((unsigned __int8)v27 - 1);
  if ( v27 <= 0x40 )
  {
    v135 = 0;
LABEL_19:
    v135 |= v29;
    goto LABEL_20;
  }
  v100 = 1LL << ((unsigned __int8)v27 - 1);
  v104 = v27 - 1;
  sub_16A4EF0((__int64)&v135, 0, 0);
  v29 = v100;
  v28 = v118;
  if ( v136 <= 0x40 )
    goto LABEL_19;
  *(_QWORD *)(v135 + 8LL * (v104 >> 6)) |= v100;
LABEL_20:
  v119 = v28;
  *(_QWORD *)&v30 = sub_1D38970(a4, (__int64)&v135, (__int64)&v133, v7, v23, 0, a5, a6, a7, 0);
  v31 = v119;
  v101 = v30;
  if ( v136 > 0x40 && v135 )
  {
    j_j___libc_free_0_0(v135);
    v31 = v119;
  }
  v120 = v31;
  if ( v131[0] )
    v32 = sub_1F3E310(v131);
  else
    v32 = sub_1F58D40((__int64)v131);
  v33 = sub_1D38BB0(a4, (unsigned int)(v32 - 1), (__int64)&v133, v7, v23, 0, a5, a6, a7, 0);
  v97 = v34;
  v98 = v33;
  *(_QWORD *)&v99 = sub_1D38BB0(a4, 0x7FFFFF, (__int64)&v133, v7, v23, 0, a5, a6, a7, 0);
  *((_QWORD *)&v99 + 1) = v35;
  v36 = sub_1D309E0(
          (__int64 *)a4,
          158,
          (__int64)&v133,
          v7,
          v23,
          0,
          *(double *)a5.m128i_i64,
          a6,
          *(double *)a7.m128i_i64,
          *(_OWORD *)*(_QWORD *)(v120 + 32));
  v124 = v37;
  v121 = v36;
  v96 = sub_1E0A0C0(*(_QWORD *)(a4 + 32));
  v38 = sub_1F40B60(a1, v7, (__int64)v23, v96, 1);
  *(_QWORD *)&v40 = sub_1D323C0(
                      (__int64 *)a4,
                      v112,
                      *((__int64 *)&v112 + 1),
                      (__int64)&v133,
                      v38,
                      v39,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64);
  v110 = v40;
  v41 = sub_1D332F0((__int64 *)a4, 118, (__int64)&v133, v7, v23, 0, *(double *)a5.m128i_i64, a6, a7, v121, v124, v108);
  v43 = sub_1D332F0(
          (__int64 *)a4,
          124,
          (__int64)&v133,
          v7,
          v23,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          (__int64)v41,
          v42,
          v110);
  *(_QWORD *)&v45 = sub_1D332F0(
                      (__int64 *)a4,
                      53,
                      (__int64)&v133,
                      v7,
                      v23,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      (__int64)v43,
                      v44,
                      v105);
  v109 = v45;
  v46 = sub_1F40B60(a1, v7, (__int64)v23, v96, 1);
  *(_QWORD *)&v48 = sub_1D323C0(
                      (__int64 *)a4,
                      v98,
                      v97,
                      (__int64)&v133,
                      v46,
                      v47,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64);
  v111 = v48;
  v49 = sub_1D332F0((__int64 *)a4, 118, (__int64)&v133, v7, v23, 0, *(double *)a5.m128i_i64, a6, a7, v121, v124, v101);
  v51 = sub_1D332F0(
          (__int64 *)a4,
          123,
          (__int64)&v133,
          v7,
          v23,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          (__int64)v49,
          v50,
          v111);
  *((_QWORD *)&v111 + 1) = v52;
  *(_QWORD *)&v111 = sub_1D322C0(
                       (__int64 *)a4,
                       (__int64)v51,
                       v52,
                       (__int64)&v133,
                       v14,
                       v128,
                       *(double *)a5.m128i_i64,
                       a6,
                       *(double *)a7.m128i_i64);
  *((_QWORD *)&v111 + 1) = v53 | *((_QWORD *)&v111 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v54 = sub_1D38BB0(a4, 0x800000, (__int64)&v133, v7, v23, 0, a5, a6, a7, 0);
  v106 = v54;
  v55 = sub_1D332F0((__int64 *)a4, 118, (__int64)&v133, v7, v23, 0, *(double *)a5.m128i_i64, a6, a7, v121, v124, v99);
  v57 = sub_1D332F0(
          (__int64 *)a4,
          119,
          (__int64)&v133,
          v7,
          v23,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          (__int64)v55,
          v56,
          v106);
  v125 = v58;
  v122 = sub_1D323C0(
           (__int64 *)a4,
           (__int64)v57,
           v58,
           (__int64)&v133,
           v14,
           v128,
           *(double *)a5.m128i_i64,
           a6,
           *(double *)a7.m128i_i64);
  v126 = v59 | v125 & 0xFFFFFFFF00000000LL;
  v60 = sub_1F40B60(a1, v7, (__int64)v23, v96, 1);
  v102 = v61;
  LODWORD(v106) = v60;
  v62 = sub_1D332F0(
          (__int64 *)a4,
          53,
          (__int64)&v133,
          v7,
          v23,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v112,
          *((unsigned __int64 *)&v112 + 1),
          v109);
  *(_QWORD *)&v64 = sub_1D323C0(
                      (__int64 *)a4,
                      (__int64)v62,
                      v63,
                      (__int64)&v133,
                      (unsigned int)v106,
                      v102,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64);
  *(_QWORD *)&v65 = sub_1D332F0(
                      (__int64 *)a4,
                      124,
                      (__int64)&v133,
                      v14,
                      v128,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      v122,
                      v126,
                      v64);
  v107 = v65;
  v66 = sub_1F40B60(a1, v7, (__int64)v23, v96, 1);
  v103 = v67;
  v113 = v66;
  v68 = sub_1D332F0(
          (__int64 *)a4,
          53,
          (__int64)&v133,
          v7,
          v23,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v109,
          *((unsigned __int64 *)&v109 + 1),
          v112);
  *(_QWORD *)&v70 = sub_1D323C0(
                      (__int64 *)a4,
                      (__int64)v68,
                      v69,
                      (__int64)&v133,
                      v113,
                      v103,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64);
  *(_QWORD *)&v99 = sub_1D332F0(
                      (__int64 *)a4,
                      122,
                      (__int64)&v133,
                      v14,
                      v128,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      v122,
                      v126,
                      v70);
  LODWORD(v103) = v71;
  v115 = v71;
  *(_QWORD *)&v75 = sub_1D28D50((_QWORD *)a4, 0x12u, v71, v72, v73, v74);
  *((_QWORD *)&v95 + 1) = v115;
  *(_QWORD *)&v95 = v99;
  v130 = sub_1D36A20(
           (__int64 *)a4,
           136,
           (__int64)&v133,
           *(unsigned __int8 *)(*(_QWORD *)(v99 + 40) + 16LL * (unsigned int)v103),
           *(const void ***)(*(_QWORD *)(v99 + 40) + 16LL * (unsigned int)v103 + 8),
           v76,
           v109,
           v112,
           v95,
           v107,
           v75);
  v78 = sub_1D332F0(
          (__int64 *)a4,
          120,
          (__int64)&v133,
          v14,
          v128,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          (__int64)v130,
          v77 | v126 & 0xFFFFFFFF00000000LL,
          v111);
  *(_QWORD *)&v80 = sub_1D332F0(
                      (__int64 *)a4,
                      53,
                      (__int64)&v133,
                      v14,
                      v128,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      (__int64)v78,
                      v79,
                      v111);
  v123 = v80;
  v81 = sub_1D38BB0(a4, 0, (__int64)&v133, v14, v128, 0, a5, a6, a7, 0);
  v82 = v7;
  v84 = v83;
  v129 = v81;
  v85 = 16LL * (unsigned int)v83;
  *(_QWORD *)&v114 = sub_1D38BB0(a4, 0, (__int64)&v133, v82, v23, 0, a5, a6, a7, 0);
  *((_QWORD *)&v114 + 1) = v86;
  *(_QWORD *)&v90 = sub_1D28D50((_QWORD *)a4, 0x14u, v86, v87, v88, v89);
  v91 = (__int64 *)a4;
  v16 = 1;
  *((_QWORD *)&v94 + 1) = v84;
  *(_QWORD *)&v94 = v129;
  *(_QWORD *)a3 = sub_1D36A20(
                    v91,
                    136,
                    (__int64)&v133,
                    *(unsigned __int8 *)(*(_QWORD *)(v129 + 40) + v85),
                    *(const void ***)(*(_QWORD *)(v129 + 40) + v85 + 8),
                    v92,
                    v109,
                    v114,
                    v94,
                    v123,
                    v90);
  *(_DWORD *)(a3 + 8) = v93;
LABEL_6:
  if ( v133 )
    sub_161E7C0((__int64)&v133, v133);
  return v16;
}
