// Function: sub_303C710
// Address: 0x303c710
//
__int64 __fastcall sub_303C710(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v5; // rdi
  __int64 (*v6)(void); // rax
  _BYTE *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 *v17; // rdi
  __int64 v18; // rax
  int v19; // edx
  unsigned __int16 v20; // ax
  __int64 v21; // rdx
  __int64 v22; // r13
  unsigned int v23; // eax
  __int64 v24; // rbx
  __int128 v25; // rax
  __int64 v26; // r8
  __int16 v27; // cx
  int v28; // r9d
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // rax
  unsigned __int16 *v33; // rax
  char v34; // al
  __int64 v35; // rdx
  int v36; // eax
  __int64 v37; // rbx
  int v38; // ecx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // r8
  unsigned __int64 v47; // r10
  __int64 *v48; // rdx
  unsigned int *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rax
  unsigned __int16 v53; // dx
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  int v57; // eax
  __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // r8
  __int16 v61; // r15
  __int64 v62; // rdx
  int v63; // r9d
  __int64 v64; // rdx
  unsigned __int16 *v65; // rax
  __int64 *v66; // rsi
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 **v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdi
  unsigned __int8 v75; // cf
  int v76; // ecx
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  unsigned __int64 v83; // rdx
  __m128i *v84; // rax
  __int64 v85; // r12
  __int64 v87; // rbx
  __int128 v88; // rax
  int v89; // r9d
  unsigned int v90; // edx
  __int64 v91; // r10
  __int64 v92; // r8
  int v93; // ecx
  __int128 v94; // rax
  int v95; // r9d
  unsigned int v96; // edx
  __int128 v97; // [rsp-20h] [rbp-1E0h]
  __int64 v98; // [rsp+18h] [rbp-1A8h]
  __int64 v99; // [rsp+20h] [rbp-1A0h]
  __int64 v100; // [rsp+28h] [rbp-198h]
  __int64 v101; // [rsp+28h] [rbp-198h]
  __int64 v102; // [rsp+30h] [rbp-190h]
  unsigned __int8 v103; // [rsp+38h] [rbp-188h]
  __int64 v104; // [rsp+38h] [rbp-188h]
  __int64 v105; // [rsp+40h] [rbp-180h]
  unsigned __int8 v106; // [rsp+4Ch] [rbp-174h]
  unsigned __int8 v107; // [rsp+4Eh] [rbp-172h]
  __m128i v108; // [rsp+50h] [rbp-170h] BYREF
  __m128i v109; // [rsp+60h] [rbp-160h]
  __int64 v110; // [rsp+70h] [rbp-150h]
  __int64 v111; // [rsp+78h] [rbp-148h]
  unsigned __int64 v112; // [rsp+80h] [rbp-140h]
  __int64 v113; // [rsp+88h] [rbp-138h]
  __int64 v114; // [rsp+90h] [rbp-130h]
  unsigned __int64 v115; // [rsp+98h] [rbp-128h]
  __int128 v116; // [rsp+A0h] [rbp-120h]
  __int64 v117; // [rsp+B0h] [rbp-110h]
  __int64 v118; // [rsp+B8h] [rbp-108h]
  __int64 v119; // [rsp+C0h] [rbp-100h]
  __int64 v120; // [rsp+C8h] [rbp-F8h]
  __int64 v121; // [rsp+D0h] [rbp-F0h] BYREF
  int v122; // [rsp+D8h] [rbp-E8h]
  int v123; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v124; // [rsp+E8h] [rbp-D8h]
  __int64 v125; // [rsp+F0h] [rbp-D0h]
  __int64 v126; // [rsp+F8h] [rbp-C8h]
  __int128 v127; // [rsp+100h] [rbp-C0h]
  __int64 v128; // [rsp+110h] [rbp-B0h]
  __int128 v129; // [rsp+120h] [rbp-A0h] BYREF
  __int64 v130; // [rsp+130h] [rbp-90h]
  __int64 v131; // [rsp+138h] [rbp-88h]
  __int64 *v132; // [rsp+140h] [rbp-80h] BYREF
  __int64 v133; // [rsp+148h] [rbp-78h]
  __int64 v134; // [rsp+150h] [rbp-70h] BYREF
  __int64 v135; // [rsp+158h] [rbp-68h]

  v5 = *(_BYTE **)(a1 + 537016);
  v113 = a2;
  v6 = *(__int64 (**)(void))(*(_QWORD *)v5 + 144LL);
  if ( (char *)v6 == (char *)sub_3020010 )
    v7 = v5 + 960;
  else
    v7 = (_BYTE *)v6();
  v8 = *(_QWORD *)(v113 + 80);
  v121 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v121, v8, 1);
  v9 = *(_QWORD *)(v113 + 40);
  v122 = *(_DWORD *)(v113 + 72);
  v10 = *(_QWORD *)(*(_QWORD *)(v9 + 80) + 96LL);
  v108 = _mm_loadu_si128((const __m128i *)v9);
  v110 = v10;
  v11 = *(_QWORD *)(v9 + 120);
  v109 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v12 = *(_QWORD *)(v11 + 96);
  v13 = *(_QWORD *)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = *(_QWORD *)v13;
  LOBYTE(v114) = 0;
  if ( v13 )
  {
    _BitScanReverse64(&v13, v13);
    LOBYTE(v114) = 1;
    v103 = 63 - (v13 ^ 0x3F);
  }
  v14 = *(_QWORD *)(*(_QWORD *)(v9 + 160) + 96LL);
  if ( *(_DWORD *)(v14 + 32) <= 0x40u )
    v100 = *(_QWORD *)(v14 + 24);
  else
    v100 = **(_QWORD **)(v14 + 24);
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  BYTE4(v130) = 0;
  v112 = v110 & 0xFFFFFFFFFFFFFFFBLL;
  *(_QWORD *)&v129 = v110 & 0xFFFFFFFFFFFFFFFBLL;
  v15 = 0;
  *((_QWORD *)&v129 + 1) = 0;
  if ( v110 )
  {
    v16 = *(_QWORD *)(v110 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
    v15 = *(_DWORD *)(v16 + 8) >> 8;
  }
  LODWORD(v130) = v15;
  v17 = *(__int64 **)(a4 + 40);
  *(_QWORD *)&v116 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
  v18 = sub_2E79000(v17);
  if ( (__int64 (__fastcall *)(__int64, __int64, unsigned int))v116 == sub_2D42F30 )
  {
    v19 = sub_AE2980(v18, 0)[1];
    v20 = 2;
    if ( v19 != 1 )
    {
      v20 = 3;
      if ( v19 != 2 )
      {
        v20 = 4;
        if ( v19 != 4 )
        {
          v20 = 5;
          if ( v19 != 8 )
          {
            v20 = 6;
            if ( v19 != 16 )
            {
              v20 = 7;
              if ( v19 != 32 )
              {
                v20 = 8;
                if ( v19 != 64 )
                  v20 = 9 * (v19 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v20 = ((__int64 (__fastcall *)(_BYTE *, __int64, _QWORD))v116)(v7, v18, 0);
  }
  v22 = sub_33F1F00(
          a4,
          v20,
          0,
          (unsigned int)&v121,
          v108.m128i_i32[0],
          v108.m128i_i32[2],
          v109.m128i_i64[0],
          v109.m128i_i64[1],
          v129,
          v130,
          0,
          0,
          (__int64)&v132,
          0);
  *(_QWORD *)&v116 = v22;
  v23 = v21;
  *((_QWORD *)&v116 + 1) = v21;
  v105 = v22;
  if ( (_BYTE)v114 && v7[73] < v103 )
  {
    v87 = 16LL * (unsigned int)v21;
    *(_QWORD *)&v88 = sub_3400BD0(
                        a4,
                        (unsigned int)(1LL << v103) - 1,
                        (unsigned int)&v121,
                        *(unsigned __int16 *)(v87 + *(_QWORD *)(v22 + 48)),
                        *(_QWORD *)(v87 + *(_QWORD *)(v22 + 48) + 8),
                        0,
                        0);
    *(_QWORD *)&v116 = sub_3406EB0(
                         a4,
                         56,
                         (unsigned int)&v121,
                         *(_WORD *)(*(_QWORD *)(v22 + 48) + v87),
                         *(_QWORD *)(*(_QWORD *)(v22 + 48) + v87 + 8),
                         v89,
                         v116,
                         v88);
    v91 = 16LL * v90;
    *((_QWORD *)&v116 + 1) = v90 | *((_QWORD *)&v116 + 1) & 0xFFFFFFFF00000000LL;
    v92 = *(_QWORD *)(*(_QWORD *)(v116 + 48) + v91 + 8);
    v93 = *(unsigned __int16 *)(*(_QWORD *)(v116 + 48) + v91);
    v114 = v91;
    *(_QWORD *)&v94 = sub_3401400(a4, -(int)(1LL << v103), (unsigned int)&v121, v93, v92, 0, 0);
    v105 = sub_3406EB0(
             a4,
             186,
             (unsigned int)&v121,
             *(_WORD *)(*(_QWORD *)(v116 + 48) + v114),
             *(_QWORD *)(*(_QWORD *)(v116 + 48) + v114 + 8),
             v95,
             v116,
             v94);
    v23 = v96;
  }
  v24 = v23;
  v102 = v23;
  v104 = 16LL * v23;
  *(_QWORD *)&v25 = sub_3400BD0(
                      a4,
                      v100,
                      (unsigned int)&v121,
                      *(unsigned __int16 *)(*(_QWORD *)(v105 + 48) + v104),
                      *(_QWORD *)(*(_QWORD *)(v105 + 48) + v104 + 8),
                      0,
                      0);
  *(_QWORD *)&v116 = v105;
  v26 = *(_QWORD *)(*(_QWORD *)(v105 + 48) + v104 + 8);
  v27 = *(_WORD *)(*(_QWORD *)(v105 + 48) + v104);
  *((_QWORD *)&v116 + 1) = v24 | *((_QWORD *)&v116 + 1) & 0xFFFFFFFF00000000LL;
  v29 = sub_3406EB0(
          a4,
          56,
          (unsigned int)&v121,
          v27,
          v26,
          v28,
          __PAIR128__(*((unsigned __int64 *)&v116 + 1), v105),
          v25);
  v132 = 0;
  v114 = v29;
  v115 = v30;
  v31 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  if ( v110 )
  {
    v32 = *(_QWORD *)(v110 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
      v32 = **(_QWORD **)(v32 + 16);
    v31 = *(_DWORD *)(v32 + 8) >> 8;
  }
  LODWORD(v130) = v31;
  BYTE4(v130) = 0;
  v129 = v112;
  v33 = (unsigned __int16 *)(*(_QWORD *)(v114 + 48) + 16LL * (unsigned int)v115);
  v34 = sub_33CC4A0(a4, *v33, *((_QWORD *)v33 + 1));
  v119 = sub_33F4560(
           a4,
           v22,
           1,
           (unsigned int)&v121,
           v114,
           v115,
           v109.m128i_i64[0],
           v109.m128i_i64[1],
           v129,
           v130,
           v34,
           0,
           (__int64)&v132);
  v108.m128i_i64[0] = v119;
  v120 = v35;
  v108.m128i_i64[1] = (unsigned int)v35 | v108.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v132 = &v134;
  v133 = 0x400000000LL;
  v36 = *(_DWORD *)(v113 + 68);
  if ( v36 == 1 )
  {
    v84 = (__m128i *)&v134;
    goto LABEL_57;
  }
  v37 = 0;
  v101 = (unsigned int)(v36 - 1);
  do
  {
    v49 = (unsigned int *)(*(_QWORD *)(v113 + 40) + 40LL * (unsigned int)(v37 + 5));
    v50 = *(_QWORD *)v49;
    v51 = *(_QWORD *)(*(_QWORD *)v49 + 96LL);
    if ( *(_DWORD *)(v51 + 32) <= 0x40u )
      v110 = *(_QWORD *)(v51 + 24);
    else
      v110 = **(_QWORD **)(v51 + 24);
    v52 = *(_QWORD *)(v50 + 48) + 16LL * v49[2];
    v53 = *(_WORD *)v52;
    v54 = *(_QWORD *)(v52 + 8);
    LOWORD(v123) = v53;
    v124 = v54;
    if ( v53 )
    {
      if ( v53 == 1 || (unsigned __int16)(v53 - 504) <= 7u )
        BUG();
      v56 = 16LL * (v53 - 1);
      v55 = *(_QWORD *)&byte_444C4A0[v56];
      LOBYTE(v56) = byte_444C4A0[v56 + 8];
    }
    else
    {
      v55 = sub_3007260((__int64)&v123);
      v125 = v55;
      v126 = v56;
    }
    BYTE8(v129) = v56;
    *(_QWORD *)&v129 = v55;
    v57 = sub_CA1930(&v129);
    LODWORD(v112) = v57 - 1;
    v58 = sub_3400BD0(a4, (unsigned int)v110 & ~(-1 << (v57 - 1)), (unsigned int)&v121, v123, v124, 0, 0);
    v59 = *(_QWORD *)(v105 + 48) + v104;
    *(_QWORD *)&v116 = v105;
    v60 = *(_QWORD *)(v59 + 8);
    v61 = *(_WORD *)v59;
    *((_QWORD *)&v97 + 1) = v62;
    *(_QWORD *)&v97 = v58;
    *((_QWORD *)&v116 + 1) = v102 | *((_QWORD *)&v116 + 1) & 0xFFFFFFFF00000000LL;
    v117 = sub_3406EB0(
             a4,
             56,
             (unsigned int)&v121,
             v61,
             v60,
             v63,
             __PAIR128__(*((unsigned __int64 *)&v116 + 1), v105),
             v97);
    v114 = v117;
    v118 = v64;
    v115 = (unsigned int)v64 | v115 & 0xFFFFFFFF00000000LL;
    v65 = (unsigned __int16 *)(*(_QWORD *)(v113 + 48) + 16 * v37);
    v66 = *(__int64 **)(a4 + 64);
    v109.m128i_i64[0] = 16 * v37;
    v67 = *v65;
    v68 = *((_QWORD *)v65 + 1);
    LOWORD(v129) = v67;
    *((_QWORD *)&v129 + 1) = v68;
    v71 = (__int64 **)sub_3007410((__int64)&v129, v66, 16 * v37, v67, v69, v70);
    v72 = sub_BCE760(v71, 5);
    v73 = sub_AD6530(v72, 5);
    v74 = v110;
    v129 = 0u;
    v130 = 0;
    v75 = _bittest64(&v74, (unsigned int)v112);
    v131 = 0;
    if ( v75 )
    {
      BYTE4(v128) = 0;
      *((_QWORD *)&v127 + 1) = 0;
      *(_QWORD *)&v127 = v73 & 0xFFFFFFFFFFFFFFFBLL;
      v38 = 0;
      if ( v73 )
      {
        v39 = *(_QWORD *)(v73 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 <= 1 )
          v39 = **(_QWORD **)(v39 + 16);
        v38 = *(_DWORD *)(v39 + 8) >> 8;
      }
      LODWORD(v128) = v38;
      v40 = *(_QWORD *)(v113 + 48) + v109.m128i_i64[0];
      v41 = v98;
      LOWORD(v41) = *(_WORD *)v40;
      v98 = v41;
      v42 = sub_33F1DB0(
              a4,
              1,
              (unsigned int)&v121,
              v41,
              *(_QWORD *)(v40 + 8),
              v106,
              v108.m128i_i64[0],
              v108.m128i_i64[1],
              v114,
              v115,
              v127,
              v128,
              5,
              0,
              0,
              (__int64)&v129);
      v44 = v43;
      v45 = (unsigned int)v133;
      v46 = v42;
      v47 = (unsigned int)v133 + 1LL;
      if ( v47 > HIDWORD(v133) )
      {
LABEL_49:
        v110 = v46;
        v111 = v44;
        sub_C8D5F0((__int64)&v132, &v134, v47, 0x10u, v46, v44);
        v45 = (unsigned int)v133;
        v46 = v110;
        v44 = v111;
      }
    }
    else
    {
      BYTE4(v128) = 0;
      *((_QWORD *)&v127 + 1) = 0;
      *(_QWORD *)&v127 = v73 & 0xFFFFFFFFFFFFFFFBLL;
      v76 = 0;
      if ( v73 )
      {
        v77 = *(_QWORD *)(v73 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v77 + 8) - 17 <= 1 )
          v77 = **(_QWORD **)(v77 + 16);
        v76 = *(_DWORD *)(v77 + 8) >> 8;
      }
      LODWORD(v128) = v76;
      v78 = *(_QWORD *)(v113 + 48) + v109.m128i_i64[0];
      v79 = v99;
      LOWORD(v79) = *(_WORD *)v78;
      v99 = v79;
      v80 = sub_33F1F00(
              a4,
              v79,
              *(_QWORD *)(v78 + 8),
              (unsigned int)&v121,
              v108.m128i_i32[0],
              v108.m128i_i32[2],
              v114,
              v115,
              v127,
              v128,
              v107,
              0,
              (__int64)&v129,
              0);
      v44 = v81;
      v45 = (unsigned int)v133;
      v46 = v80;
      v47 = (unsigned int)v133 + 1LL;
      if ( v47 > HIDWORD(v133) )
        goto LABEL_49;
    }
    v48 = &v132[2 * v45];
    ++v37;
    *v48 = v46;
    v48[1] = v44;
    LODWORD(v133) = v133 + 1;
  }
  while ( v37 != v101 );
  v82 = (unsigned int)v133;
  v83 = (unsigned int)v133 + 1LL;
  if ( v83 > HIDWORD(v133) )
  {
    sub_C8D5F0((__int64)&v132, &v134, v83, 0x10u, v46, v44);
    v82 = (unsigned int)v133;
  }
  v84 = (__m128i *)&v132[2 * v82];
LABEL_57:
  *v84 = _mm_load_si128(&v108);
  LODWORD(v133) = v133 + 1;
  v85 = sub_3411660(a4, v132, (unsigned int)v133, &v121);
  if ( v132 != &v134 )
    _libc_free((unsigned __int64)v132);
  if ( v121 )
    sub_B91220((__int64)&v121, v121);
  return v85;
}
