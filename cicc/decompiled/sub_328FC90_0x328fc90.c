// Function: sub_328FC90
// Address: 0x328fc90
//
__int64 __fastcall sub_328FC90(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r9
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  const __m128i *v18; // rax
  __int64 v19; // r12
  __int64 v20; // r13
  unsigned __int32 v21; // ebx
  __int16 *v22; // rax
  __int16 v23; // dx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // rsi
  unsigned __int16 *v36; // rax
  __int64 v37; // r10
  unsigned int v38; // ecx
  __int64 v39; // rax
  __int32 v40; // edx
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  int v47; // r14d
  __int64 v48; // r12
  int v49; // edx
  int v50; // r15d
  int v51; // eax
  __m128i v52; // xmm3
  __m128i v53; // xmm4
  int v54; // esi
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r14
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // rax
  __int64 v62; // r12
  int v63; // edx
  int v64; // r15d
  int v65; // eax
  __m128i v66; // xmm1
  __m128i v67; // xmm2
  __int64 v68; // rdx
  __int64 v69; // r15
  int v70; // edx
  int v71; // eax
  __m128i v72; // xmm5
  __m128i v73; // xmm6
  int v74; // r9d
  __int64 v75; // rax
  int v76; // edx
  __int64 v77; // r15
  __int64 v78; // rax
  int v79; // r12d
  __int64 v80; // rbx
  int v81; // edx
  int v82; // eax
  __m128i v83; // xmm7
  __m128i v84; // xmm5
  int v85; // r9d
  __int128 v86; // rax
  __int64 v87; // r9
  __int128 v88; // rax
  __int128 v89; // rax
  __int64 v90; // r9
  __int128 v91; // [rsp-20h] [rbp-180h]
  __int128 v92; // [rsp-10h] [rbp-170h]
  __int128 v93; // [rsp-10h] [rbp-170h]
  unsigned int v94; // [rsp+0h] [rbp-160h]
  __int64 v95; // [rsp+8h] [rbp-158h]
  __int128 v96; // [rsp+10h] [rbp-150h]
  __int64 v97; // [rsp+20h] [rbp-140h]
  __int32 v98; // [rsp+2Ch] [rbp-134h]
  __int64 v99; // [rsp+40h] [rbp-120h]
  int v100; // [rsp+40h] [rbp-120h]
  __int32 v101; // [rsp+48h] [rbp-118h]
  int v102; // [rsp+48h] [rbp-118h]
  __int64 v103; // [rsp+50h] [rbp-110h]
  __int64 v104; // [rsp+50h] [rbp-110h]
  int v105; // [rsp+50h] [rbp-110h]
  int v106; // [rsp+50h] [rbp-110h]
  int v107; // [rsp+58h] [rbp-108h]
  __int64 v108; // [rsp+58h] [rbp-108h]
  int v109; // [rsp+58h] [rbp-108h]
  __int64 v110; // [rsp+60h] [rbp-100h]
  __int64 v111; // [rsp+60h] [rbp-100h]
  int v112; // [rsp+60h] [rbp-100h]
  int v113; // [rsp+60h] [rbp-100h]
  __int128 v114; // [rsp+60h] [rbp-100h]
  __int64 v115; // [rsp+80h] [rbp-E0h] BYREF
  int v116; // [rsp+88h] [rbp-D8h]
  __int64 v117; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v118; // [rsp+98h] [rbp-C8h]
  __int64 v119; // [rsp+A0h] [rbp-C0h]
  __int64 v120; // [rsp+A8h] [rbp-B8h]
  __m128i v121; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v122; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v123; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v124; // [rsp+E0h] [rbp-80h]
  __int64 v125; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v126; // [rsp+F8h] [rbp-68h]
  __int64 v127; // [rsp+100h] [rbp-60h]
  int v128; // [rsp+108h] [rbp-58h]
  __m128i v129; // [rsp+110h] [rbp-50h]
  __m128i v130; // [rsp+120h] [rbp-40h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = v4[1];
  v115 = v5;
  v8 = v4[5];
  v9 = v4[6];
  v10 = v4[10];
  v11 = v4[11];
  if ( v5 )
  {
    v103 = v4[6];
    v107 = v4[5];
    v110 = v6;
    sub_B96E90((__int64)&v115, v5, 1);
    LODWORD(v9) = v103;
    LODWORD(v8) = v107;
    v6 = v110;
  }
  v12 = v6;
  v13 = *a1;
  v116 = *(_DWORD *)(a2 + 72);
  v14 = sub_33E28A0(v13, v6, v7, v8, v9, v6, v10, v11);
  v15 = v14;
  if ( v14 )
  {
    v16 = v14;
    goto LABEL_5;
  }
  v18 = *(const __m128i **)(a2 + 40);
  v19 = *a1;
  v20 = v18->m128i_i64[0];
  v21 = v18->m128i_u32[2];
  v111 = v18[2].m128i_i64[1];
  v96 = (__int128)_mm_loadu_si128(v18);
  v108 = v18[3].m128i_i64[0];
  v101 = v18[3].m128i_i32[0];
  v104 = v18[5].m128i_i64[0];
  v99 = v18[5].m128i_i64[1];
  v98 = v18[5].m128i_i32[2];
  v22 = *(__int16 **)(a2 + 48);
  v23 = *v22;
  v24 = *((_QWORD *)v22 + 1);
  v121.m128i_i64[0] = v19;
  v118 = v24;
  v25 = *(_QWORD *)(v19 + 16);
  LOWORD(v117) = v23;
  v121.m128i_i64[1] = v25;
  v122.m128i_i64[0] = 0;
  v122.m128i_i32[2] = 0;
  v123.m128i_i64[0] = 0;
  v26 = *(unsigned int *)(a2 + 24);
  v123.m128i_i32[2] = 0;
  v124 = a2;
  v119 = sub_33CB160(v26);
  if ( BYTE4(v119) )
  {
    v12 = *(_QWORD *)(v124 + 40);
    v27 = v12 + 40LL * (unsigned int)v119;
    v122.m128i_i64[0] = *(_QWORD *)v27;
    v122.m128i_i32[2] = *(_DWORD *)(v27 + 8);
    v28 = *(unsigned int *)(v124 + 24);
  }
  else
  {
    v34 = v124;
    v28 = *(unsigned int *)(v124 + 24);
    if ( (_DWORD)v28 == 488 )
    {
      v35 = *(_QWORD *)(v124 + 80);
      v36 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v124 + 40) + 48LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(v124 + 40) + 8LL));
      v37 = *((_QWORD *)v36 + 1);
      v38 = *v36;
      v125 = v35;
      if ( v35 )
      {
        v94 = v38;
        v95 = v37;
        v97 = v124;
        sub_B96E90((__int64)&v125, v35, 1);
        v38 = v94;
        v37 = v95;
        v34 = v97;
      }
      LODWORD(v126) = *(_DWORD *)(v34 + 72);
      v39 = sub_34015B0(v19, &v125, v38, v37, 0, 0);
      v12 = v125;
      v122.m128i_i64[0] = v39;
      v122.m128i_i32[2] = v40;
      if ( v125 )
        sub_B91220((__int64)&v125, v125);
      v28 = *(unsigned int *)(v124 + 24);
    }
  }
  v125 = sub_33CB1F0(v28);
  if ( BYTE4(v125) )
  {
    v31 = *(_QWORD *)(v124 + 40) + 40LL * (unsigned int)v125;
    v123.m128i_i64[0] = *(_QWORD *)v31;
    v123.m128i_i32[2] = *(_DWORD *)(v31 + 8);
  }
  v32 = (unsigned __int16)v117;
  v33 = *(_QWORD *)(v20 + 48) + 16LL * v21;
  if ( *(_WORD *)v33 != (_WORD)v117 )
    goto LABEL_13;
  if ( (_WORD)v117 )
  {
    if ( (unsigned __int16)(v117 - 17) > 0xD3u )
    {
      LOWORD(v125) = v117;
      v126 = v118;
      goto LABEL_23;
    }
    v32 = (unsigned __int16)word_4456580[(unsigned __int16)v117 - 1];
  }
  else
  {
    v57 = v118;
    if ( *(_QWORD *)(v33 + 8) != v118 )
      goto LABEL_13;
    if ( !sub_30070B0((__int64)&v117) )
    {
      v126 = v57;
      LOWORD(v125) = 0;
      goto LABEL_39;
    }
    v32 = (unsigned int)sub_3009970((__int64)&v117, v12, v58, v59, v60);
    v15 = v68;
  }
  LOWORD(v125) = v32;
  v126 = v15;
  if ( !(_WORD)v32 )
  {
LABEL_39:
    v42 = sub_3007260((__int64)&v125);
    v119 = v42;
    v120 = v56;
    goto LABEL_26;
  }
LABEL_23:
  v41 = (unsigned __int16)v32;
  if ( (_WORD)v32 == 1 || (LOWORD(v32) = v32 - 504, (unsigned __int16)v32 <= 7u) )
    BUG();
  v42 = *(_QWORD *)&byte_444C4A0[16 * v41 - 16];
LABEL_26:
  if ( v42 != 1 )
  {
LABEL_13:
    v16 = 0;
    goto LABEL_5;
  }
  if ( v21 == v101 && v20 == v111 || (unsigned __int8)sub_33E0780(v111, v108, 1, v32, v29, v30) )
  {
    v61 = sub_33FB960(v19, v104, v99);
    v47 = v117;
    v62 = v61;
    v64 = v63;
    v112 = v118;
    v65 = sub_33CB7C0(187);
    v66 = _mm_loadu_si128(&v122);
    v67 = _mm_loadu_si128(&v123);
    v125 = v20;
    LODWORD(v126) = v21;
    v54 = v65;
    v127 = v62;
    v128 = v64;
    v129 = v66;
    v130 = v67;
    goto LABEL_34;
  }
  if ( v20 == v104 && v21 == v98 || (unsigned __int8)sub_33E0720(v104, v99, 1) )
  {
    v46 = sub_33FB960(v19, v111, v108);
    v47 = v117;
    v48 = v46;
    v50 = v49;
    v112 = v118;
    v51 = sub_33CB7C0(186);
    v52 = _mm_loadu_si128(&v122);
    v53 = _mm_loadu_si128(&v123);
    v125 = v20;
    LODWORD(v126) = v21;
    v54 = v51;
    v127 = v48;
    v128 = v50;
    v129 = v52;
    v130 = v53;
LABEL_34:
    *((_QWORD *)&v92 + 1) = 4;
    *(_QWORD *)&v92 = &v125;
    v55 = sub_33FC220(v121.m128i_i32[0], v54, (unsigned int)&v115, v47, v112, (unsigned int)&v115, v92);
    goto LABEL_35;
  }
  if ( (unsigned __int8)sub_33E0780(v104, v99, 1, v43, v44, v45) )
  {
    v69 = sub_34015B0(v19, &v115, (unsigned int)v117, v118, 0, 0);
    v100 = v70;
    v102 = v118;
    v105 = v117;
    v71 = sub_33CB7C0(188);
    LODWORD(v126) = v21;
    v125 = v20;
    v72 = _mm_loadu_si128(&v122);
    v128 = v100;
    v73 = _mm_loadu_si128(&v123);
    *((_QWORD *)&v93 + 1) = 4;
    *(_QWORD *)&v93 = &v125;
    v129 = v72;
    v130 = v73;
    v127 = v69;
    v75 = sub_33FC220(v121.m128i_i32[0], v71, (unsigned int)&v115, v105, v102, v74, v93);
    v106 = v76;
    v77 = v75;
    v78 = sub_33FB960(v19, v111, v108);
    v79 = v117;
    v80 = v78;
    v109 = v81;
    v113 = v118;
    v82 = sub_33CB7C0(187);
    v125 = v77;
    v83 = _mm_loadu_si128(&v122);
    v127 = v80;
    v128 = v109;
    v84 = _mm_loadu_si128(&v123);
    *((_QWORD *)&v91 + 1) = 4;
    *(_QWORD *)&v91 = &v125;
    LODWORD(v126) = v106;
    v129 = v83;
    v130 = v84;
    v55 = sub_33FC220(v121.m128i_i32[0], v82, (unsigned int)&v115, v79, v113, v85, v91);
  }
  else
  {
    if ( !(unsigned __int8)sub_33E0720(v111, v108, 1) )
      goto LABEL_13;
    *(_QWORD *)&v86 = sub_34015B0(v19, &v115, (unsigned int)v117, v118, 0, 0);
    *(_QWORD *)&v88 = sub_328FC10(&v121, 0xBCu, (int)&v115, v117, v118, v87, v96, v86);
    v114 = v88;
    *(_QWORD *)&v89 = sub_33FB960(v19, v104, v99);
    v55 = sub_328FC10(&v121, 0xBAu, (int)&v115, v117, v118, v90, v114, v89);
  }
LABEL_35:
  if ( !v55 )
    goto LABEL_13;
  v16 = v55;
LABEL_5:
  if ( v115 )
    sub_B91220((__int64)&v115, v115);
  return v16;
}
