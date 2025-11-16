// Function: sub_379AF20
// Address: 0x379af20
//
void __fastcall sub_379AF20(__int64 *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v4; // r8
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // r14
  unsigned __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // r12
  unsigned __int16 v17; // dx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int16 v20; // cx
  __int64 v21; // rax
  unsigned __int16 v22; // r12
  __int64 v23; // rax
  unsigned int v24; // r13d
  int v25; // r8d
  unsigned int v26; // ecx
  __int64 v27; // rdi
  _QWORD *v28; // rax
  unsigned int v29; // r8d
  _QWORD *v30; // r14
  unsigned int v31; // esi
  bool v32; // al
  char v33; // dl
  char v34; // al
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  unsigned __int8 v39; // al
  __int64 v40; // r12
  __int64 v41; // rsi
  __int64 v42; // rdx
  char v43; // al
  _QWORD *v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // rdx
  _QWORD *v47; // rdi
  __m128i *v48; // r13
  unsigned __int64 v49; // rdx
  __int128 v50; // rax
  _QWORD *v51; // r12
  __m128i v52; // xmm2
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int8 v56; // al
  __m128i *v57; // rax
  __int16 v58; // cx
  __int64 v59; // r13
  __int64 v60; // r8
  unsigned int v61; // edx
  __int64 v62; // rax
  unsigned __int64 v63; // r12
  __int64 v64; // rdx
  __int64 *v65; // rdi
  __m128i *v66; // rax
  __int64 v67; // rcx
  int v68; // edx
  const __m128i *v69; // roff
  __int16 v70; // dx
  __int64 *v71; // rdi
  __int64 v72; // rcx
  __int64 v73; // rax
  __m128i *v74; // rax
  __int64 v75; // rsi
  unsigned int v76; // edx
  unsigned int v77; // eax
  int v78; // eax
  unsigned int v79; // eax
  __m128i v80; // rax
  __int64 v81; // r12
  __int64 v82; // rax
  __int64 v83; // rsi
  _QWORD *v84; // rbx
  int v85; // edx
  unsigned __int8 *v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rdi
  unsigned __int16 *v89; // rdx
  __int64 v90; // r9
  unsigned int v91; // edx
  __int128 v92; // [rsp+0h] [rbp-210h]
  char v93; // [rsp+1Bh] [rbp-1F5h]
  unsigned int v94; // [rsp+1Ch] [rbp-1F4h]
  unsigned int v95; // [rsp+1Ch] [rbp-1F4h]
  unsigned int v96; // [rsp+20h] [rbp-1F0h]
  unsigned __int16 v97; // [rsp+20h] [rbp-1F0h]
  int v98; // [rsp+20h] [rbp-1F0h]
  unsigned int v99; // [rsp+24h] [rbp-1ECh]
  unsigned int v100; // [rsp+24h] [rbp-1ECh]
  unsigned __int16 v101; // [rsp+24h] [rbp-1ECh]
  unsigned __int16 v102; // [rsp+24h] [rbp-1ECh]
  unsigned __int16 v103; // [rsp+24h] [rbp-1ECh]
  __int64 v104; // [rsp+28h] [rbp-1E8h]
  __int128 v105; // [rsp+30h] [rbp-1E0h]
  __int128 v106; // [rsp+30h] [rbp-1E0h]
  __int64 v107; // [rsp+50h] [rbp-1C0h]
  __int64 v108; // [rsp+50h] [rbp-1C0h]
  __int64 *v109; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v110; // [rsp+58h] [rbp-1B8h]
  unsigned __int64 v111; // [rsp+58h] [rbp-1B8h]
  __int64 v112; // [rsp+60h] [rbp-1B0h]
  unsigned __int64 v114; // [rsp+70h] [rbp-1A0h]
  unsigned __int64 v115; // [rsp+78h] [rbp-198h]
  unsigned __int8 v116; // [rsp+78h] [rbp-198h]
  __int64 v117; // [rsp+F0h] [rbp-120h] BYREF
  int v118; // [rsp+F8h] [rbp-118h]
  unsigned int v119; // [rsp+100h] [rbp-110h] BYREF
  unsigned __int64 v120; // [rsp+108h] [rbp-108h]
  unsigned int v121; // [rsp+110h] [rbp-100h] BYREF
  __int64 v122; // [rsp+118h] [rbp-F8h]
  __int64 v123; // [rsp+120h] [rbp-F0h] BYREF
  __int64 v124; // [rsp+128h] [rbp-E8h]
  unsigned __int64 v125; // [rsp+130h] [rbp-E0h] BYREF
  unsigned __int64 v126; // [rsp+138h] [rbp-D8h]
  unsigned __int64 v127; // [rsp+140h] [rbp-D0h]
  __int64 v128; // [rsp+148h] [rbp-C8h]
  __int64 v129; // [rsp+150h] [rbp-C0h]
  __int64 v130; // [rsp+158h] [rbp-B8h]
  __int128 v131; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v132; // [rsp+170h] [rbp-A0h]
  __m128i v133; // [rsp+180h] [rbp-90h] BYREF
  __int64 v134; // [rsp+190h] [rbp-80h]
  __int128 v135; // [rsp+1A0h] [rbp-70h] BYREF
  __int64 v136; // [rsp+1B0h] [rbp-60h]
  __int64 v137; // [rsp+1C0h] [rbp-50h] BYREF
  __int64 v138; // [rsp+1C8h] [rbp-48h]
  __int64 v139; // [rsp+1D0h] [rbp-40h]
  __int64 v140; // [rsp+1D8h] [rbp-38h]

  v4 = a2;
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v10 = *(unsigned int *)(v7 + 8);
  v11 = *(unsigned int *)(v7 + 48);
  v114 = *(_QWORD *)v7;
  v12 = *(_QWORD *)(v7 + 80);
  v105 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 80));
  v110 = *(_QWORD *)(v7 + 8);
  v13 = *(_QWORD *)v7;
  v117 = v8;
  v115 = v13;
  v112 = *(_QWORD *)(v7 + 40);
  if ( v8 )
  {
    v107 = v4;
    sub_B96E90((__int64)&v117, v8, 1);
    v4 = v107;
  }
  v118 = *(_DWORD *)(v4 + 72);
  sub_375E8D0((__int64)a1, v114, v110, a3, (__int64)a4);
  v15 = *(_QWORD *)a3;
  v104 = 16 * v11;
  v16 = *(_QWORD *)(v115 + 48) + 16 * v10;
  v17 = *(_WORD *)v16;
  v120 = *(_QWORD *)(v16 + 8);
  v18 = *(unsigned int *)(a3 + 8);
  LOWORD(v119) = v17;
  v19 = *(_QWORD *)(v15 + 48) + 16 * v18;
  v20 = *(_WORD *)v19;
  v122 = *(_QWORD *)(v19 + 8);
  v21 = *(_QWORD *)(v112 + 48) + 16 * v11;
  LOWORD(v121) = v20;
  v22 = *(_WORD *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  LOWORD(v123) = v22;
  v108 = v23;
  v124 = v23;
  if ( v17 )
  {
    v24 = word_4456340[v17 - 1];
    if ( v22 )
      goto LABEL_5;
LABEL_24:
    v97 = v17;
    v102 = v20;
    v78 = sub_3007240((__int64)&v123);
    v20 = v102;
    v17 = v97;
    v25 = v78;
    if ( v102 )
      goto LABEL_6;
    goto LABEL_25;
  }
  v101 = v20;
  v77 = sub_3007240((__int64)&v119);
  v17 = 0;
  v20 = v101;
  v24 = v77;
  if ( !v22 )
    goto LABEL_24;
LABEL_5:
  v25 = word_4456340[v22 - 1];
  if ( v20 )
  {
LABEL_6:
    v26 = word_4456340[v20 - 1];
    goto LABEL_7;
  }
LABEL_25:
  v98 = v25;
  v103 = v17;
  v79 = sub_3007240((__int64)&v121);
  v25 = v98;
  v17 = v103;
  v26 = v79;
LABEL_7:
  v27 = *(_QWORD *)(v12 + 96);
  v28 = *(_QWORD **)(v27 + 24);
  if ( *(_DWORD *)(v27 + 32) > 0x40u )
    v28 = (_QWORD *)*v28;
  v29 = (_DWORD)v28 + v25;
  v30 = (_QWORD *)a1[1];
  v31 = (unsigned int)v28;
  if ( v29 <= v26 )
  {
    *(_QWORD *)a3 = sub_340F900(
                      (_QWORD *)a1[1],
                      0xA0u,
                      (__int64)&v117,
                      v121,
                      v122,
                      v14,
                      *(_OWORD *)a3,
                      *(_OWORD *)&v9,
                      v105);
    *(_DWORD *)(a3 + 8) = v85;
    goto LABEL_38;
  }
  if ( v17 )
  {
    v33 = (unsigned __int16)(v17 - 176) <= 0x34u;
    if ( v22 )
      goto LABEL_12;
LABEL_22:
    v93 = v33;
    v95 = v26;
    v100 = v29;
    v34 = sub_3007100((__int64)&v123);
    v33 = v93;
    v26 = v95;
    v29 = v100;
    goto LABEL_13;
  }
  v94 = v26;
  v99 = v29;
  v96 = (unsigned int)v28;
  v32 = sub_3007100((__int64)&v119);
  v26 = v94;
  v31 = v96;
  v29 = v99;
  v33 = v32;
  if ( !v22 )
    goto LABEL_22;
LABEL_12:
  v34 = (unsigned __int16)(v22 - 176) <= 0x34u;
LABEL_13:
  if ( v31 >= v26 && v29 <= v24 && v34 == v33 )
  {
    v86 = sub_3400EE0((__int64)v30, v31 - v26, (__int64)&v117, 0, v9);
    v88 = v87;
    v89 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2]);
    *((_QWORD *)&v92 + 1) = v88;
    *(_QWORD *)&v92 = v86;
    *(_QWORD *)a4 = sub_340F900(
                      v30,
                      0xA0u,
                      (__int64)&v117,
                      *v89,
                      *((_QWORD *)v89 + 1),
                      v90,
                      *(_OWORD *)a4,
                      *(_OWORD *)&v9,
                      v92);
    a4[2] = v91;
LABEL_38:
    if ( v117 )
      sub_B91220((__int64)&v117, v117);
    return;
  }
  v35 = *a1;
  sub_2FE6CC0((__int64)&v137, *a1, v30[8], v22, v108);
  if ( (_BYTE)v137 == 7 && *(_DWORD *)(v115 + 24) == 51 && (unsigned __int16)sub_3281170(&v123, v35, v36, v37, v38) == 2 )
  {
    v80.m128i_i64[0] = sub_379AB60((__int64)a1, v9.m128i_u64[0], v9.m128i_i64[1]);
    v133 = v80;
    v81 = v80.m128i_i64[0];
    v82 = *(_QWORD *)(v80.m128i_i64[0] + 48) + 16LL * v80.m128i_u32[2];
    if ( (_WORD)v119 == *(_WORD *)v82 && (v120 == *(_QWORD *)(v82 + 8) || *(_WORD *)v82) )
    {
      v83 = *(_QWORD *)(v81 + 80);
      v84 = (_QWORD *)a1[1];
      *(_QWORD *)&v135 = v83;
      if ( v83 )
        sub_B96E90((__int64)&v135, v83, 1);
      DWORD2(v135) = *(_DWORD *)(v81 + 72);
      sub_3776C40((__int64)&v137, v84, (__int128 *)v133.m128i_i8, (__int64)&v135);
      *(_QWORD *)a3 = v137;
      *(_DWORD *)(a3 + 8) = v138;
      *(_QWORD *)a4 = v139;
      a4[2] = v140;
      sub_9C6650(&v135);
      goto LABEL_38;
    }
  }
  v39 = sub_33CD850(a1[1], v119, v120, 0);
  v40 = a1[1];
  v116 = v39;
  if ( (_WORD)v119 )
  {
    if ( (_WORD)v119 == 1 || (unsigned __int16)(v119 - 504) <= 7u )
      BUG();
    v41 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v119 - 16];
    v43 = byte_444C4A0[16 * (unsigned __int16)v119 - 8];
  }
  else
  {
    v129 = sub_3007260((__int64)&v119);
    v41 = v129;
    v130 = v42;
    v43 = v42;
  }
  LOBYTE(v128) = v43;
  v127 = (unsigned __int64)(v41 + 7) >> 3;
  v44 = sub_33EDE90(v40, v127, v128, v116);
  v126 = v45;
  v46 = a1[1];
  v125 = (unsigned __int64)v44;
  v109 = *(__int64 **)(v46 + 40);
  sub_2EAC300((__int64)&v131, (__int64)v109, *((_DWORD *)v44 + 24), 0);
  v47 = (_QWORD *)a1[1];
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v48 = sub_33F4560(
          v47,
          (unsigned __int64)(v47 + 36),
          0,
          (__int64)&v117,
          v114,
          v110,
          v125,
          v126,
          v131,
          v132,
          v116,
          0,
          (__int64)&v137);
  v111 = v49;
  *(_QWORD *)&v50 = sub_3465D80(v9, *a1, (_QWORD *)a1[1], v125, v126, v119, v120, v123, v124, v105);
  v51 = (_QWORD *)a1[1];
  v106 = v50;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  sub_2EAC3A0((__int64)&v133, v109);
  v52 = _mm_loadu_si128(&v133);
  v136 = v134;
  v135 = (__int128)v52;
  v56 = sub_33CC4A0(
          (__int64)v51,
          *(unsigned __int16 *)(*(_QWORD *)(v112 + 48) + v104),
          *(_QWORD *)(*(_QWORD *)(v112 + 48) + v104 + 8),
          v53,
          v54,
          v55);
  v57 = sub_33F4560(
          v51,
          (unsigned __int64)v48,
          v111,
          (__int64)&v117,
          v9.m128i_u64[0],
          v9.m128i_u64[1],
          v106,
          *((unsigned __int64 *)&v106 + 1),
          v135,
          v136,
          v56,
          0,
          (__int64)&v137);
  HIBYTE(v58) = 1;
  v59 = (__int64)v57;
  LOBYTE(v58) = v116;
  v60 = (__int64)v57;
  v62 = *(unsigned int *)(a3 + 8);
  v137 = 0;
  v63 = v61 | v111 & 0xFFFFFFFF00000000LL;
  v64 = *(_QWORD *)a3;
  v138 = 0;
  v139 = 0;
  v65 = (__int64 *)a1[1];
  v140 = 0;
  v66 = sub_33F1F00(
          v65,
          *(unsigned __int16 *)(*(_QWORD *)(v64 + 48) + 16 * v62),
          *(_QWORD *)(*(_QWORD *)(v64 + 48) + 16 * v62 + 8),
          (__int64)&v117,
          v60,
          v63,
          v125,
          v126,
          v131,
          v132,
          v58,
          0,
          (__int64)&v137,
          0);
  v67 = v122;
  *(_QWORD *)a3 = v66;
  *(_DWORD *)(a3 + 8) = v68;
  v69 = (const __m128i *)v66[7].m128i_i64[0];
  v135 = (__int128)_mm_loadu_si128(v69);
  v136 = v69[1].m128i_i64[0];
  sub_3777490((__int64)a1, (__int64)v66, v121, v67, (__int64)&v135, (unsigned int *)&v125, v9, 0);
  HIBYTE(v70) = 1;
  v137 = 0;
  v71 = (__int64 *)a1[1];
  v72 = *(_QWORD *)a4;
  v73 = a4[2];
  v138 = 0;
  LOBYTE(v70) = v116;
  v139 = 0;
  v140 = 0;
  v74 = sub_33F1F00(
          v71,
          *(unsigned __int16 *)(*(_QWORD *)(v72 + 48) + 16 * v73),
          *(_QWORD *)(*(_QWORD *)(v72 + 48) + 16 * v73 + 8),
          (__int64)&v117,
          v59,
          v63,
          v125,
          v126,
          v135,
          v136,
          v70,
          0,
          (__int64)&v137,
          0);
  v75 = v117;
  *(_QWORD *)a4 = v74;
  a4[2] = v76;
  if ( v75 )
    sub_B91220((__int64)&v117, v75);
}
