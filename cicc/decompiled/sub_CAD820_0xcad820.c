// Function: sub_CAD820
// Address: 0xcad820
//
__int64 __fastcall sub_CAD820(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  int v7; // edx
  __m128i v8; // xmm0
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // r14
  __int64 v17; // r9
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // r15
  _QWORD *v22; // rsi
  __int64 v23; // r8
  _QWORD *v24; // rcx
  __m128i v25; // xmm3
  _BYTE *v26; // rdi
  size_t v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rsi
  __int64 v30; // r8
  _QWORD *v31; // rcx
  __m128i v32; // xmm1
  _BYTE *v33; // rdi
  size_t v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  const char *v40; // rax
  __int64 v41; // r8
  __int64 v42; // r14
  __int64 v43; // rdx
  __int128 v44; // kr00_16
  unsigned __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // r14
  __int64 v48; // rdx
  __int128 v49; // kr10_16
  unsigned __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // r14
  __int64 v53; // rdx
  __int128 v54; // kr20_16
  unsigned __int64 v55; // rax
  __int64 v56; // r8
  __int64 v57; // r14
  __int64 v58; // rdx
  __int128 v59; // kr30_16
  unsigned __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r14
  __int64 v63; // rdx
  __int128 v64; // kr40_16
  unsigned __int64 v65; // rax
  __int64 v66; // r8
  __int64 v67; // r14
  __int64 v68; // rdx
  __int128 v69; // kr50_16
  unsigned __int64 v70; // rax
  size_t v71; // r14
  void *v72; // r15
  size_t v73; // r10
  __int64 v74; // r8
  __int64 v75; // r11
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // r9
  __m128i v79; // kr60_16
  unsigned __int64 v80; // rax
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rax
  __int64 v84; // rdx
  unsigned __int64 v85; // rax
  __int64 v86; // r14
  __int128 v87; // kr70_16
  char *v88; // rcx
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int128 v97; // [rsp-10h] [rbp-190h]
  __int64 v98; // [rsp+8h] [rbp-178h]
  __int64 v99; // [rsp+10h] [rbp-170h]
  __int64 v100; // [rsp+18h] [rbp-168h]
  __int128 v101; // [rsp+20h] [rbp-160h]
  size_t v102; // [rsp+20h] [rbp-160h]
  __int64 v103; // [rsp+28h] [rbp-158h]
  size_t v104; // [rsp+28h] [rbp-158h]
  __int64 v105; // [rsp+30h] [rbp-150h]
  __int64 v106; // [rsp+30h] [rbp-150h]
  __int64 v107; // [rsp+30h] [rbp-150h]
  __int64 v108; // [rsp+30h] [rbp-150h]
  __int64 v109; // [rsp+30h] [rbp-150h]
  __int64 v110; // [rsp+30h] [rbp-150h]
  __int64 v111; // [rsp+30h] [rbp-150h]
  __int64 v112; // [rsp+38h] [rbp-148h]
  unsigned __int64 v113; // [rsp+38h] [rbp-148h]
  __int64 v114; // [rsp+38h] [rbp-148h]
  __int128 v115; // [rsp+38h] [rbp-148h]
  __int128 v116; // [rsp+38h] [rbp-148h]
  unsigned __int64 v117; // [rsp+38h] [rbp-148h]
  __int128 v118; // [rsp+38h] [rbp-148h]
  __int128 v119; // [rsp+38h] [rbp-148h]
  size_t v120; // [rsp+38h] [rbp-148h]
  __int64 v121; // [rsp+40h] [rbp-140h]
  __int64 v122; // [rsp+40h] [rbp-140h]
  void *v123; // [rsp+40h] [rbp-140h]
  __int64 v124; // [rsp+40h] [rbp-140h]
  unsigned __int64 v125; // [rsp+40h] [rbp-140h]
  unsigned __int64 v126; // [rsp+40h] [rbp-140h]
  int v127; // [rsp+50h] [rbp-130h] BYREF
  __m128i v128; // [rsp+58h] [rbp-128h]
  void *v129; // [rsp+68h] [rbp-118h] BYREF
  size_t v130; // [rsp+70h] [rbp-110h]
  _QWORD v131[3]; // [rsp+78h] [rbp-108h] BYREF
  int v132; // [rsp+90h] [rbp-F0h]
  __m128i v133; // [rsp+98h] [rbp-E8h]
  void *dest; // [rsp+A8h] [rbp-D8h]
  size_t v135; // [rsp+B0h] [rbp-D0h]
  _QWORD v136[3]; // [rsp+B8h] [rbp-C8h] BYREF
  int v137; // [rsp+D0h] [rbp-B0h]
  __int128 v138; // [rsp+D8h] [rbp-A8h]
  void *v139; // [rsp+E8h] [rbp-98h]
  size_t v140; // [rsp+F0h] [rbp-90h]
  _QWORD v141[3]; // [rsp+F8h] [rbp-88h] BYREF
  const char *v142; // [rsp+110h] [rbp-70h] BYREF
  __m128i v143; // [rsp+118h] [rbp-68h] BYREF
  _QWORD *v144; // [rsp+128h] [rbp-58h]
  size_t n; // [rsp+130h] [rbp-50h]
  _QWORD src[9]; // [rsp+138h] [rbp-48h] BYREF

  v6 = sub_CAD7A0((__int64 **)a1, a2, a3, a4, a5);
  v7 = *(_DWORD *)v6;
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 8));
  v129 = v131;
  v9 = *(_BYTE **)(v6 + 24);
  v127 = v7;
  v10 = *(_QWORD *)(v6 + 32);
  v128 = v8;
  sub_CA64F0((__int64 *)&v129, v9, (__int64)&v9[v10]);
  dest = v136;
  v132 = 0;
  v133 = 0u;
  v135 = 0;
  LOBYTE(v136[0]) = 0;
  v137 = 0;
  v138 = 0u;
  v139 = v141;
  v140 = 0;
  LOBYTE(v141[0]) = 0;
  while ( 1 )
  {
    v15 = v127;
    if ( v127 != 21 )
    {
      while ( 1 )
      {
        if ( v15 != 22 )
        {
          if ( v15 == 20 )
          {
            sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
            if ( v144 != src )
              j_j___libc_free_0(v144, src[0] + 1LL);
            v16 = v128.m128i_i64[1];
            v17 = v128.m128i_i64[0];
            if ( v128.m128i_i64[1] )
            {
              v16 = v128.m128i_i64[1] - 1;
              v17 = v128.m128i_i64[0] + 1;
            }
            v18 = *(_QWORD *)(a1 + 8);
            *(_QWORD *)(a1 + 88) += 88LL;
            v19 = (v18 + 15) & 0xFFFFFFFFFFFFFFF0LL;
            if ( *(_QWORD *)(a1 + 16) >= v19 + 88 && v18 )
            {
              *(_QWORD *)(a1 + 8) = v19 + 88;
              v20 = (v18 + 15) & 0xFFFFFFFFFFFFFFF0LL;
            }
            else
            {
              v122 = v17;
              v38 = sub_9D1E70(a1 + 8, 88, 88, 4);
              v17 = v122;
              v20 = v38;
            }
            v121 = v17;
            sub_CAD7C0(v20, 6u, *(_QWORD *)a1 + 8LL, 0, 0, v17, 0);
            *(_QWORD *)(v20 + 80) = v16;
            *(_QWORD *)(v20 + 72) = v121;
            *(_QWORD *)v20 = &unk_49DCD38;
          }
          else
          {
            switch ( v15 )
            {
              case 0:
                v20 = 0;
                break;
              case 7:
                v41 = v133.m128i_i64[1];
                v42 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v41 = v133.m128i_i64[1] - 1;
                  v42 = v133.m128i_i64[0] + 1;
                }
                v43 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v44 = v138;
                v45 = (v43 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v45 + 88 && v43 )
                {
                  *(_QWORD *)(a1 + 8) = v45 + 88;
                  v20 = (v43 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v105 = v41;
                  v113 = *((_QWORD *)&v138 + 1);
                  v125 = v138;
                  v89 = sub_9D1E70(a1 + 8, 88, 88, 4);
                  v41 = v105;
                  v20 = v89;
                  v44 = __PAIR128__(v113, v125);
                }
                sub_CAD7C0(v20, 5u, *(_QWORD *)a1 + 8LL, v42, v41, *((__int64 *)&v44 + 1), v44);
                *(_BYTE *)(v20 + 78) = 1;
                *(_DWORD *)(v20 + 72) = 2;
                *(_WORD *)(v20 + 76) = 1;
                *(_QWORD *)v20 = &unk_49DCD18;
                *(_QWORD *)(v20 + 80) = 0;
                break;
              case 9:
                sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v144 != src )
                  j_j___libc_free_0(v144, src[0] + 1LL);
                v46 = v133.m128i_i64[1];
                v47 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v46 = v133.m128i_i64[1] - 1;
                  v47 = v133.m128i_i64[0] + 1;
                }
                v48 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v49 = v138;
                v50 = (v48 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v50 + 88 && v48 )
                {
                  *(_QWORD *)(a1 + 8) = v50 + 88;
                  v20 = (v48 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v107 = v46;
                  v115 = v138;
                  v91 = sub_9D1E70(a1 + 8, 88, 88, 4);
                  v46 = v107;
                  v20 = v91;
                  v49 = v115;
                }
                sub_CAD7C0(v20, 5u, *(_QWORD *)a1 + 8LL, v47, v46, *((__int64 *)&v49 + 1), v49);
                *(_BYTE *)(v20 + 78) = 1;
                *(_DWORD *)(v20 + 72) = 0;
                *(_WORD *)(v20 + 76) = 1;
                *(_QWORD *)v20 = &unk_49DCD18;
                *(_QWORD *)(v20 + 80) = 0;
                break;
              case 10:
                sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v144 != src )
                  j_j___libc_free_0(v144, src[0] + 1LL);
                v56 = v133.m128i_i64[1];
                v57 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v56 = v133.m128i_i64[1] - 1;
                  v57 = v133.m128i_i64[0] + 1;
                }
                v58 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v59 = v138;
                v60 = (v58 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v60 + 88 && v58 )
                {
                  *(_QWORD *)(a1 + 8) = v60 + 88;
                  v20 = (v58 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v108 = v56;
                  v116 = v138;
                  v92 = sub_9D1E70(a1 + 8, 88, 88, 4);
                  v56 = v108;
                  v20 = v92;
                  v59 = v116;
                }
                sub_CAD7C0(v20, 4u, *(_QWORD *)a1 + 8LL, v57, v56, *((__int64 *)&v59 + 1), v59);
                *(_DWORD *)(v20 + 72) = 0;
                *(_WORD *)(v20 + 76) = 1;
                *(_QWORD *)(v20 + 80) = 0;
                *(_QWORD *)v20 = &unk_49DCCF8;
                break;
              case 11:
              case 13:
              case 15:
                v39 = *(_QWORD *)(a1 + 104);
                if ( v39 && (unsigned int)(*(_DWORD *)(v39 + 32) - 4) <= 1 )
                  goto LABEL_37;
                BYTE1(n) = 1;
                v40 = "Unexpected token";
                goto LABEL_57;
              case 12:
                sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v144 != src )
                  j_j___libc_free_0(v144, src[0] + 1LL);
                v51 = v133.m128i_i64[1];
                v52 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v51 = v133.m128i_i64[1] - 1;
                  v52 = v133.m128i_i64[0] + 1;
                }
                v53 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v54 = v138;
                v55 = (v53 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v55 + 88 && v53 )
                {
                  *(_QWORD *)(a1 + 8) = v55 + 88;
                  v20 = (v53 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v109 = v51;
                  v117 = *((_QWORD *)&v138 + 1);
                  v126 = v138;
                  v93 = sub_9D1E70(a1 + 8, 88, 88, 4);
                  v51 = v109;
                  v20 = v93;
                  v54 = __PAIR128__(v117, v126);
                }
                sub_CAD7C0(v20, 5u, *(_QWORD *)a1 + 8LL, v52, v51, *((__int64 *)&v54 + 1), v54);
                *(_BYTE *)(v20 + 78) = 1;
                *(_DWORD *)(v20 + 72) = 1;
                *(_WORD *)(v20 + 76) = 1;
                *(_QWORD *)v20 = &unk_49DCD18;
                *(_QWORD *)(v20 + 80) = 0;
                break;
              case 14:
                sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v144 != src )
                  j_j___libc_free_0(v144, src[0] + 1LL);
                v61 = v133.m128i_i64[1];
                v62 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v61 = v133.m128i_i64[1] - 1;
                  v62 = v133.m128i_i64[0] + 1;
                }
                v63 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v64 = v138;
                v65 = (v63 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v65 + 88 && v63 )
                {
                  *(_QWORD *)(a1 + 8) = v65 + 88;
                  v20 = (v63 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v111 = v61;
                  v119 = v138;
                  v95 = sub_9D1E70(a1 + 8, 88, 88, 4);
                  v61 = v111;
                  v20 = v95;
                  v64 = v119;
                }
                sub_CAD7C0(v20, 4u, *(_QWORD *)a1 + 8LL, v62, v61, *((__int64 *)&v64 + 1), v64);
                *(_DWORD *)(v20 + 72) = 1;
                *(_WORD *)(v20 + 76) = 1;
                *(_QWORD *)(v20 + 80) = 0;
                *(_QWORD *)v20 = &unk_49DCCF8;
                break;
              case 16:
                v66 = v133.m128i_i64[1];
                v67 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v66 = v133.m128i_i64[1] - 1;
                  v67 = v133.m128i_i64[0] + 1;
                }
                v68 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v69 = v138;
                v70 = (v68 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v70 + 88 && v68 )
                {
                  *(_QWORD *)(a1 + 8) = v70 + 88;
                  v20 = (v68 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v110 = v66;
                  v118 = v138;
                  v94 = sub_9D1E70(a1 + 8, 88, 88, 4);
                  v66 = v110;
                  v20 = v94;
                  v69 = v118;
                }
                sub_CAD7C0(v20, 4u, *(_QWORD *)a1 + 8LL, v67, v66, *((__int64 *)&v69 + 1), v69);
                *(_DWORD *)(v20 + 72) = 2;
                *(_WORD *)(v20 + 76) = 1;
                *(_QWORD *)(v20 + 80) = 0;
                *(_QWORD *)v20 = &unk_49DCCF8;
                break;
              case 18:
                sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v144 != src )
                  j_j___libc_free_0(v144, src[0] + 1LL);
                v81 = v133.m128i_i64[1];
                v82 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v81 = v133.m128i_i64[1] - 1;
                  v82 = v133.m128i_i64[0] + 1;
                }
                v83 = v128.m128i_i64[1];
                v84 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v124 = v83;
                v85 = (v84 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                v86 = v128.m128i_i64[0];
                v87 = v138;
                if ( *(_QWORD *)(a1 + 16) >= v85 + 88 && v84 )
                {
                  *(_QWORD *)(a1 + 8) = v85 + 88;
                  v20 = (v84 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v101 = v138;
                  v106 = v81;
                  v114 = v82;
                  v90 = sub_9D1E70(a1 + 8, 88, 88, 4);
                  v87 = v101;
                  v81 = v106;
                  v82 = v114;
                  v20 = v90;
                }
                sub_CAD7C0(v20, 1u, *(_QWORD *)a1 + 8LL, v82, v81, v82, v87);
                *(_QWORD *)(v20 + 72) = v86;
                *(_QWORD *)(v20 + 16) = v86;
                *(_QWORD *)(v20 + 80) = v124;
                *(_QWORD *)v20 = &unk_49DCC98;
                *(_QWORD *)(v20 + 24) = v86 + v124;
                break;
              case 19:
                sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
                if ( v144 != src )
                  j_j___libc_free_0(v144, src[0] + 1LL);
                v71 = v130;
                v123 = 0;
                v103 = a1 + 8;
                v72 = v129;
                v73 = v130 + 1;
                if ( v130 != -1 )
                {
                  v88 = *(char **)(a1 + 8);
                  *(_QWORD *)(a1 + 88) += v73;
                  v123 = v88;
                  if ( *(_QWORD *)(a1 + 16) >= (unsigned __int64)&v88[v73] && v88 )
                  {
                    *(_QWORD *)(a1 + 8) = &v88[v73];
                    memmove(v88, v72, v73);
                  }
                  else
                  {
                    v120 = v73;
                    v123 = (void *)sub_9D1E70(v103, v73, v73, 0);
                    memmove(v123, v72, v120);
                  }
                  v73 = v71;
                }
                v74 = v133.m128i_i64[1];
                v75 = v133.m128i_i64[0];
                if ( v133.m128i_i64[1] )
                {
                  v74 = v133.m128i_i64[1] - 1;
                  v75 = v133.m128i_i64[0] + 1;
                }
                v76 = v138;
                v77 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 88LL;
                v78 = *((_QWORD *)&v138 + 1);
                v112 = v76;
                v79 = v128;
                v80 = (v77 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v80 + 88 && v77 )
                {
                  *(_QWORD *)(a1 + 8) = v80 + 88;
                  v20 = (v77 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v98 = v74;
                  v99 = v75;
                  v100 = *((_QWORD *)&v138 + 1);
                  v102 = v73;
                  v96 = sub_9D1E70(v103, 88, 88, 4);
                  v74 = v98;
                  v75 = v99;
                  v78 = v100;
                  v73 = v102;
                  v20 = v96;
                }
                *((_QWORD *)&v97 + 1) = v78;
                v104 = v73;
                *(_QWORD *)&v97 = v112;
                sub_CAD7C0(v20, 2u, *(_QWORD *)a1 + 8LL, v75, v74, v78, v97);
                *(_QWORD *)(v20 + 16) = v79.m128i_i64[0];
                *(_QWORD *)(v20 + 72) = v123;
                *(_QWORD *)v20 = &unk_49DCCB8;
                *(_QWORD *)(v20 + 80) = v104;
                *(_QWORD *)(v20 + 24) = v79.m128i_i64[1] + v79.m128i_i64[0];
                break;
              default:
LABEL_37:
                v36 = *(_QWORD *)(a1 + 8);
                *(_QWORD *)(a1 + 88) += 72LL;
                v37 = (v36 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                if ( *(_QWORD *)(a1 + 16) >= v37 + 72 && v36 )
                {
                  *(_QWORD *)(a1 + 8) = v37 + 72;
                  v20 = (v36 + 15) & 0xFFFFFFFFFFFFFFF0LL;
                }
                else
                {
                  v20 = sub_9D1E70(a1 + 8, 72, 72, 4);
                }
                sub_CAD7C0(v20, 0, *(_QWORD *)a1 + 8LL, 0, 0, v14, 0);
                *(_QWORD *)v20 = &unk_49DCC78;
                break;
            }
          }
          goto LABEL_14;
        }
        if ( v137 == 22 )
        {
          BYTE1(n) = 1;
          v40 = "Already encountered a tag for this node!";
          goto LABEL_57;
        }
        v22 = (_QWORD *)a1;
        sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
        v24 = src;
        v25 = _mm_loadu_si128(&v143);
        v26 = v139;
        v137 = (int)v142;
        v138 = (__int128)v25;
        if ( v144 == src )
        {
          v27 = n;
          if ( n )
          {
            if ( n == 1 )
            {
              *(_BYTE *)v139 = src[0];
              v27 = n;
              v26 = v139;
            }
            else
            {
              v22 = src;
              memcpy(v139, src, n);
              v27 = n;
              v26 = v139;
              v24 = src;
            }
          }
          v140 = v27;
          v26[v27] = 0;
          v26 = v144;
          goto LABEL_26;
        }
        v22 = (_QWORD *)n;
        v27 = src[0];
        if ( v139 == v141 )
          break;
        v23 = v141[0];
        v139 = v144;
        v140 = n;
        v141[0] = src[0];
        if ( !v26 )
          goto LABEL_50;
        v144 = v26;
        src[0] = v23;
LABEL_26:
        n = 0;
        *v26 = 0;
        if ( v144 != src )
        {
          v22 = (_QWORD *)(src[0] + 1LL);
          j_j___libc_free_0(v144, src[0] + 1LL);
        }
        v28 = sub_CAD7A0((__int64 **)a1, (unsigned __int64)v22, v27, (__int64)v24, v23);
        v127 = *(_DWORD *)v28;
        v128 = _mm_loadu_si128((const __m128i *)(v28 + 8));
        sub_2240AE0(&v129, v28 + 24);
        v15 = v127;
        if ( v127 == 21 )
          goto LABEL_29;
      }
      v139 = v144;
      v140 = n;
      v141[0] = src[0];
LABEL_50:
      v144 = src;
      v24 = src;
      v26 = src;
      goto LABEL_26;
    }
LABEL_29:
    if ( v132 == 21 )
      break;
    v29 = (_QWORD *)a1;
    sub_CAD680((__int64)&v142, (unsigned __int64 **)a1, v11, v12, v13);
    v31 = src;
    v32 = _mm_loadu_si128(&v143);
    v33 = dest;
    v132 = (int)v142;
    v133 = v32;
    if ( v144 == src )
    {
      v34 = n;
      if ( n )
      {
        if ( n == 1 )
        {
          *(_BYTE *)dest = src[0];
          v34 = n;
          v33 = dest;
        }
        else
        {
          v29 = src;
          memcpy(dest, src, n);
          v34 = n;
          v33 = dest;
          v31 = src;
        }
      }
      v135 = v34;
      v33[v34] = 0;
      v33 = v144;
    }
    else
    {
      v29 = (_QWORD *)n;
      v34 = src[0];
      if ( dest == v136 )
      {
        dest = v144;
        v135 = n;
        v136[0] = src[0];
      }
      else
      {
        v30 = v136[0];
        dest = v144;
        v135 = n;
        v136[0] = src[0];
        if ( v33 )
        {
          v144 = v33;
          src[0] = v30;
          goto LABEL_34;
        }
      }
      v144 = src;
      v31 = src;
      v33 = src;
    }
LABEL_34:
    n = 0;
    *v33 = 0;
    if ( v144 != src )
    {
      v29 = (_QWORD *)(src[0] + 1LL);
      j_j___libc_free_0(v144, src[0] + 1LL);
    }
    v35 = sub_CAD7A0((__int64 **)a1, (unsigned __int64)v29, v34, (__int64)v31, v30);
    v127 = *(_DWORD *)v35;
    v128 = _mm_loadu_si128((const __m128i *)(v35 + 8));
    sub_2240AE0(&v129, v35 + 24);
  }
  BYTE1(n) = 1;
  v40 = "Already encountered an anchor for this node!";
LABEL_57:
  v142 = v40;
  v20 = 0;
  LOBYTE(n) = 3;
  sub_CA8C70((__int64 **)a1, (__int64)&v142, (__int64)&v127, v12, v13);
LABEL_14:
  if ( v139 != v141 )
    j_j___libc_free_0(v139, v141[0] + 1LL);
  if ( dest != v136 )
    j_j___libc_free_0(dest, v136[0] + 1LL);
  if ( v129 != v131 )
    j_j___libc_free_0(v129, v131[0] + 1LL);
  return v20;
}
