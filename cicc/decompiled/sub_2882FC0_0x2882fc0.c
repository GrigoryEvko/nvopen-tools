// Function: sub_2882FC0
// Address: 0x2882fc0
//
__int64 __fastcall sub_2882FC0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 v8; // r12
  __int64 v9; // rax
  __m128i si128; // xmm0
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  __int64 *v14; // rsi
  __int64 *v15; // rcx
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rdi
  __int64 v21; // rax
  bool v22; // si
  int v23; // ecx
  int v24; // eax
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __m128i v30; // xmm0
  __int64 v32; // rax
  __m128i v33; // xmm0
  __int64 v34; // r15
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // r14
  int v38; // eax
  int v39; // r13d
  __int64 v40; // r15
  unsigned int v41; // r12d
  __int64 v42; // rsi
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  unsigned __int64 v45; // rax
  __int64 v46; // r13
  unsigned int v47; // eax
  __int64 v48; // r14
  __int64 v49; // r13
  __int64 v50; // r15
  __int64 v51; // rax
  bool v52; // al
  unsigned int v53; // edx
  int v54; // r9d
  unsigned int v55; // r13d
  __int64 v56; // r8
  unsigned int v57; // edi
  __int64 v58; // rcx
  unsigned __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __m128i v62; // xmm0
  __int8 *v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // r12
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __m128i v73; // xmm2
  __m128i v74; // xmm3
  __m128i v75; // xmm4
  unsigned __int64 *v76; // r15
  unsigned __int64 *v77; // r12
  __int64 v78; // r8
  unsigned __int64 *v79; // r14
  unsigned __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __m128i v85; // xmm0
  unsigned int v86; // esi
  __int64 v87; // rax
  __m128i v88; // xmm0
  char v89; // al
  __int64 v90; // rax
  __m128i v91; // xmm0
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  unsigned __int64 *v95; // r12
  __int64 v96; // r8
  unsigned __int64 v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // [rsp+0h] [rbp-470h]
  unsigned __int64 v101; // [rsp+8h] [rbp-468h]
  __int64 v102; // [rsp+8h] [rbp-468h]
  __int64 v103; // [rsp+10h] [rbp-460h]
  unsigned __int64 v104; // [rsp+10h] [rbp-460h]
  __int64 *v105; // [rsp+18h] [rbp-458h]
  unsigned int v106; // [rsp+20h] [rbp-450h]
  __int64 v107; // [rsp+24h] [rbp-44Ch]
  int v108; // [rsp+2Ch] [rbp-444h]
  __int64 *v109; // [rsp+30h] [rbp-440h]
  __int64 v110; // [rsp+48h] [rbp-428h]
  __int64 *v111; // [rsp+48h] [rbp-428h]
  __int64 v112; // [rsp+50h] [rbp-420h]
  unsigned int v113; // [rsp+58h] [rbp-418h]
  bool v114; // [rsp+5Eh] [rbp-412h]
  char v115; // [rsp+5Fh] [rbp-411h]
  __int64 v117; // [rsp+68h] [rbp-408h]
  unsigned int v118; // [rsp+68h] [rbp-408h]
  unsigned __int8 *v121; // [rsp+90h] [rbp-3E0h]
  __int64 v122; // [rsp+90h] [rbp-3E0h]
  unsigned int v123; // [rsp+98h] [rbp-3D8h]
  __int64 v124; // [rsp+A0h] [rbp-3D0h] BYREF
  __int64 v125; // [rsp+A8h] [rbp-3C8h] BYREF
  __int64 v126; // [rsp+B0h] [rbp-3C0h] BYREF
  __int64 v127; // [rsp+B8h] [rbp-3B8h] BYREF
  __int8 *v128; // [rsp+C0h] [rbp-3B0h] BYREF
  size_t v129; // [rsp+C8h] [rbp-3A8h]
  _QWORD v130[2]; // [rsp+D0h] [rbp-3A0h] BYREF
  __m128i v131; // [rsp+E0h] [rbp-390h] BYREF
  __int64 v132; // [rsp+F0h] [rbp-380h] BYREF
  __m128i v133; // [rsp+F8h] [rbp-378h]
  __int64 v134; // [rsp+108h] [rbp-368h]
  _OWORD v135[2]; // [rsp+110h] [rbp-360h] BYREF
  unsigned __int64 *v136; // [rsp+130h] [rbp-340h] BYREF
  __int64 v137; // [rsp+138h] [rbp-338h]
  _BYTE v138[320]; // [rsp+140h] [rbp-330h] BYREF
  char v139; // [rsp+280h] [rbp-1F0h]
  int v140; // [rsp+284h] [rbp-1ECh]
  __int64 v141; // [rsp+288h] [rbp-1E8h]
  unsigned __int64 v142; // [rsp+290h] [rbp-1E0h] BYREF
  __int64 v143; // [rsp+298h] [rbp-1D8h]
  __int64 v144; // [rsp+2A0h] [rbp-1D0h] BYREF
  __m128i v145; // [rsp+2A8h] [rbp-1C8h] BYREF
  __int64 v146; // [rsp+2B8h] [rbp-1B8h]
  __m128i v147; // [rsp+2C0h] [rbp-1B0h] BYREF
  __m128i v148; // [rsp+2D0h] [rbp-1A0h] BYREF
  unsigned __int64 *v149; // [rsp+2E0h] [rbp-190h] BYREF
  unsigned int v150; // [rsp+2E8h] [rbp-188h]
  char v151; // [rsp+2F0h] [rbp-180h] BYREF
  char v152; // [rsp+430h] [rbp-40h]
  int v153; // [rsp+434h] [rbp-3Ch]
  __int64 v154; // [rsp+438h] [rbp-38h]

  v6 = (__int64 *)a6;
  v8 = a1;
  sub_D4BD20(&v124, a1, (__int64)a3, a4, (__int64)a5, a6);
  v117 = **(_QWORD **)(a1 + 32);
  v121 = (unsigned __int8 *)sub_D48970(a1);
  if ( (_BYTE)qword_50020C8 )
  {
    v131.m128i_i64[0] = 41;
    v142 = (unsigned __int64)&v144;
    v9 = sub_22409D0((__int64)&v142, (unsigned __int64 *)&v131, 0);
    v142 = v9;
    v144 = v131.m128i_i64[0];
    *(__m128i *)v9 = _mm_load_si128((const __m128i *)&xmmword_43970C0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_43970D0);
    *(_QWORD *)(v9 + 32) = 0x6E696C6C6F726E75LL;
    *(_BYTE *)(v9 + 40) = 103;
    *(__m128i *)(v9 + 16) = si128;
    v143 = v131.m128i_i64[0];
    *(_BYTE *)(v142 + v131.m128i_i64[0]) = 0;
    sub_2882C00(v6, a1, (__int64)&v142, 0x6E696C6C6F726E75LL, v11, v12);
    if ( (__int64 *)v142 != &v144 )
      j_j___libc_free_0(v142);
    sub_D48630(a1, &v126, &v125);
    if ( v121 )
    {
      v13 = *(_QWORD *)(v125 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v13 == v125 + 48 || !v13 || (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
        goto LABEL_152;
      if ( *(_BYTE *)(v13 - 24) == 31 )
      {
        v123 = sub_DCF980(a5, (char *)a1);
        if ( !v123 )
        {
          v147.m128i_i64[0] = (__int64)&v148;
          v142 = 6;
          v143 = 0;
          v144 = 0;
          v145.m128i_i32[0] = 0;
          v145.m128i_i64[1] = 0;
          v146 = 0;
          v147.m128i_i64[1] = 0x200000000LL;
          v115 = sub_D4B6F0(a1, (__int64)a5, (__int64)&v142);
          if ( v115 )
            v115 = sub_1023590((__int64)&v142) != 0;
          if ( (__m128i *)v147.m128i_i64[0] != &v148 )
            _libc_free(v147.m128i_u64[0]);
          if ( v144 != -4096 && v144 != 0 && v144 != -8192 )
            sub_BD60C0(&v142);
          if ( !v115 )
            goto LABEL_111;
          v112 = v125;
          if ( v117 == v125 )
          {
            HIDWORD(v107) = *(_DWORD *)(a2 + 32);
            LODWORD(v107) = *(_DWORD *)(a2 + 36);
            v118 = *(_DWORD *)(a2 + 44);
            if ( *(_DWORD *)(a2 + 8) )
            {
              v113 = 0;
              v115 = 0;
              v108 = *(_DWORD *)(a2 + 40);
              v106 = *(_DWORD *)(a2 + 48) + *(_DWORD *)(a2 + 52);
              v52 = v118 == 0;
              goto LABEL_69;
            }
            v22 = 0;
LABEL_123:
            v23 = *(_DWORD *)(a2 + 40);
            v24 = *(_DWORD *)(a2 + 48);
            v25 = *(_DWORD *)(a2 + 52);
LABEL_115:
            v110 = *(_QWORD *)a2;
            goto LABEL_116;
          }
          v14 = *(__int64 **)(a1 + 32);
          v105 = *(__int64 **)(a1 + 40);
          if ( v14 != v105 )
          {
            v15 = *(__int64 **)(a1 + 32);
            while ( 1 )
            {
              v16 = *v15;
              if ( v125 == *v15 )
                goto LABEL_28;
              v17 = *(_QWORD *)(v16 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v17 == v16 + 48 || !v17 || (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 > 0xA )
                goto LABEL_152;
              if ( *(_BYTE *)(v17 - 24) != 31 || (*(_DWORD *)(v17 - 20) & 0x7FFFFFF) != 1 )
                break;
              if ( v105 == ++v15 )
                goto LABEL_28;
            }
            v23 = *(_DWORD *)(a2 + 40);
            v25 = *(_DWORD *)(a2 + 52);
            HIDWORD(v107) = *(_DWORD *)(a2 + 32);
            v108 = v23;
            LODWORD(v107) = *(_DWORD *)(a2 + 36);
            v118 = *(_DWORD *)(a2 + 44);
            v24 = *(_DWORD *)(a2 + 48);
            if ( *(_DWORD *)(a2 + 8) )
            {
              v113 = 0;
              v114 = *(_DWORD *)(a2 + 24) != 0;
              v106 = v25 + v24;
LABEL_44:
              v109 = v14;
              v100 = v8 + 56;
              v111 = v6;
              while ( 1 )
              {
                v34 = *v109;
                if ( v112 == *v109 )
                {
LABEL_67:
                  v6 = v111;
                  goto LABEL_68;
                }
                v103 = v34 + 48;
                v35 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                v36 = v35;
                if ( v34 + 48 == v35 )
                  goto LABEL_152;
                if ( !v35 )
                  BUG();
                v37 = v35 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v35 - 24) - 30 <= 0xA )
                {
                  v101 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  v38 = sub_B46E30(v37);
                  v36 = v101;
                  v39 = v38;
                  if ( v38 )
                  {
                    v102 = v34;
                    v40 = v8;
                    v41 = 0;
                    do
                    {
                      v42 = sub_B46EC0(v37, v41);
                      if ( *(_BYTE *)(v40 + 84) )
                      {
                        v43 = *(_QWORD **)(v40 + 64);
                        v44 = &v43[*(unsigned int *)(v40 + 76)];
                        if ( v43 == v44 )
                          goto LABEL_119;
                        while ( v42 != *v43 )
                        {
                          if ( v44 == ++v43 )
                            goto LABEL_119;
                        }
                      }
                      else if ( !sub_C8CA60(v100, v42) )
                      {
LABEL_119:
                        v8 = v40;
                        goto LABEL_120;
                      }
                      ++v41;
                    }
                    while ( v39 != v41 );
                    v8 = v40;
                    v45 = *(_QWORD *)(v102 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                    v36 = v45;
                    if ( v45 == v103 || !v45 )
                      goto LABEL_152;
                  }
                }
                v104 = v36;
                v46 = v36 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v36 - 24) - 30 > 0xA )
                  goto LABEL_152;
                v47 = sub_B46E30(v36 - 24);
                if ( v47 > 2 || *(_BYTE *)(v104 - 24) == 40 )
                  break;
                if ( v47 != 1 )
                {
                  v48 = sub_B46EC0(v46, 0);
                  v49 = sub_B46EC0(v46, 1u);
                  v50 = sub_AA56F0(v48);
                  v51 = sub_AA56F0(v49);
                  if ( v49 != v50 && v48 != v51 && (!v50 || v50 != v51) )
                  {
LABEL_120:
                    v6 = v111;
                    v52 = v118 == 0;
                    goto LABEL_69;
                  }
                }
                if ( v105 == ++v109 )
                  goto LABEL_67;
              }
              v115 = v47 > 2 || *(_BYTE *)(v104 - 24) == 40;
              v6 = v111;
              v52 = v118 == 0;
LABEL_69:
              if ( v106 > 0x17 )
              {
                v53 = v113 - 15;
                if ( 2 * v106 + 1 < v113 - 15 )
                  v53 = v113;
                v113 = v53;
              }
              v54 = HIDWORD(v107) | v107;
              if ( v52 && v108 != 0 )
              {
                if ( v107 )
                {
                  if ( !a3 || (v89 = sub_10563D0(a3, v121), v54 = HIDWORD(v107) | v107, v89) )
                  {
                    v142 = (unsigned __int64)&v144;
                    v131.m128i_i64[0] = 37;
                    v90 = sub_22409D0((__int64)&v142, (unsigned __int64 *)&v131, 0);
                    v142 = v90;
                    v144 = v131.m128i_i64[0];
                    *(__m128i *)v90 = _mm_load_si128((const __m128i *)&xmmword_43970E0);
                    v91 = _mm_load_si128((const __m128i *)&xmmword_43970F0);
                    *(_DWORD *)(v90 + 32) = 1819045746;
                    *(__m128i *)(v90 + 16) = v91;
                    *(_BYTE *)(v90 + 36) = 46;
                    v143 = v131.m128i_i64[0];
                    *(_BYTE *)(v142 + v131.m128i_i64[0]) = 0;
                    sub_2882C00(v6, v8, (__int64)&v142, v92, v93, v94);
                    sub_2240A30(&v142);
                    goto LABEL_38;
                  }
                }
              }
              if ( v115 )
                goto LABEL_111;
              v55 = 0;
              v56 = (unsigned int)qword_5001E28 >> 2;
              v57 = 1 << SLOBYTE(qword_500A028[8]);
              v58 = v118;
              if ( v118 - 1 <= 1 && (unsigned int)qword_5001E28 / v118 > v113 )
                v55 = 4 / v118;
              if ( HIDWORD(v107) && (v58 = v113, (unsigned int)v56 > v113) )
              {
                v55 = qword_5001FE8;
                if ( (_DWORD)qword_5001FE8 )
                  goto LABEL_85;
                v56 = (unsigned int)qword_5001F08;
                v59 = (unsigned int)qword_5001E28 / v113;
                if ( !((unsigned int)qword_5001E28 / v113) )
                  goto LABEL_84;
              }
              else
              {
                v86 = (unsigned int)qword_5001E28 >> 1;
                if ( (int)(float)((float)(6 * v107 + 22) * (float)((float)(int)qword_5001E28 / 200.0)) <= (unsigned int)qword_5001E28 >> 1 )
                  v86 = (int)(float)((float)(6 * v107 + 22) * (float)((float)(int)qword_5001E28 / 200.0));
                if ( !(v54 | v108) || (v58 = v113, v86 <= v113) || (v55 = qword_5001FE8) != 0 )
                {
LABEL_85:
                  if ( v55 > 1 )
                  {
                    if ( sub_2880E70((__int64 *)a2, a4, v55, v58, v56) <= (unsigned __int64)(unsigned int)(4 * *(_DWORD *)(a4 + 12)) )
                    {
                      v128 = (__int8 *)v130;
                      v142 = 40;
                      v61 = sub_22409D0((__int64)&v128, &v142, 0);
                      v128 = (__int8 *)v61;
                      v130[0] = v142;
                      *(__m128i *)v61 = _mm_load_si128((const __m128i *)&xmmword_4397140);
                      v62 = _mm_load_si128((const __m128i *)&xmmword_4397150);
                      v63 = v128;
                      *(_QWORD *)(v61 + 32) = 0x203A737365636375LL;
                      *(__m128i *)(v61 + 16) = v62;
                      v129 = v142;
                      v63[v142] = 0;
                      sub_D4BD20(&v127, v8, (__int64)v63, v64, v65, v66);
                      if ( v6 )
                      {
                        v122 = *v6;
                        v67 = **(_QWORD **)(v8 + 32);
                        v68 = sub_B2BE50(*v6);
                        if ( sub_B6EA50(v68)
                          || (v98 = sub_B2BE50(v122),
                              v99 = sub_B6F970(v98),
                              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v99 + 48LL))(v99)) )
                        {
                          sub_B157E0((__int64)&v131, &v127);
                          sub_B17430(
                            (__int64)&v142,
                            (__int64)"loop-unroll",
                            (__int64)"computeRuntimeUnrollCount",
                            25,
                            &v131,
                            v67);
                          sub_B18290((__int64)&v142, v128, v129);
                          sub_B169E0(v131.m128i_i64, "RuntimeUnrollVariable", 21, v55);
                          sub_23FD640((__int64)&v142, (__int64)&v131);
                          if ( (_OWORD *)v133.m128i_i64[1] != v135 )
                            j_j___libc_free_0(v133.m128i_u64[1]);
                          if ( (__int64 *)v131.m128i_i64[0] != &v132 )
                            j_j___libc_free_0(v131.m128i_u64[0]);
                          v73 = _mm_loadu_si128(&v145);
                          v74 = _mm_loadu_si128(&v147);
                          v75 = _mm_loadu_si128(&v148);
                          v131.m128i_i32[2] = v143;
                          v133 = v73;
                          v131.m128i_i8[12] = BYTE4(v143);
                          v135[0] = v74;
                          v132 = v144;
                          v135[1] = v75;
                          v131.m128i_i64[0] = (__int64)&unk_49D9D40;
                          v134 = v146;
                          v136 = (unsigned __int64 *)v138;
                          v137 = 0x400000000LL;
                          if ( v150 )
                          {
                            sub_2882450((__int64)&v136, (__int64)&v149, v69, v70, v71, v72);
                            v142 = (unsigned __int64)&unk_49D9D40;
                            v95 = v149;
                            v139 = v152;
                            v140 = v153;
                            v141 = v154;
                            v131.m128i_i64[0] = (__int64)&unk_49D9D78;
                            v96 = 10LL * v150;
                            v76 = &v149[v96];
                            if ( v149 != &v149[v96] )
                            {
                              do
                              {
                                v76 -= 10;
                                v97 = v76[4];
                                if ( (unsigned __int64 *)v97 != v76 + 6 )
                                  j_j___libc_free_0(v97);
                                if ( (unsigned __int64 *)*v76 != v76 + 2 )
                                  j_j___libc_free_0(*v76);
                              }
                              while ( v95 != v76 );
                              v76 = v149;
                            }
                          }
                          else
                          {
                            v76 = v149;
                            v139 = v152;
                            v140 = v153;
                            v141 = v154;
                            v131.m128i_i64[0] = (__int64)&unk_49D9D78;
                          }
                          if ( v76 != (unsigned __int64 *)&v151 )
                            _libc_free((unsigned __int64)v76);
                          sub_1049740(v6, (__int64)&v131);
                          v77 = v136;
                          v131.m128i_i64[0] = (__int64)&unk_49D9D40;
                          v78 = 10LL * (unsigned int)v137;
                          v79 = &v136[v78];
                          if ( v136 != &v136[v78] )
                          {
                            do
                            {
                              v79 -= 10;
                              v80 = v79[4];
                              if ( (unsigned __int64 *)v80 != v79 + 6 )
                                j_j___libc_free_0(v80);
                              if ( (unsigned __int64 *)*v79 != v79 + 2 )
                                j_j___libc_free_0(*v79);
                            }
                            while ( v77 != v79 );
                            v79 = v136;
                          }
                          if ( v79 != (unsigned __int64 *)v138 )
                            _libc_free((unsigned __int64)v79);
                        }
                      }
                      if ( v127 )
                        sub_B91220((__int64)&v127, v127);
                      if ( v128 != (__int8 *)v130 )
                        j_j___libc_free_0((unsigned __int64)v128);
                      v123 = v55;
                      goto LABEL_38;
                    }
                    v142 = (unsigned __int64)&v144;
                    v131.m128i_i64[0] = 88;
                    v87 = sub_22409D0((__int64)&v142, (unsigned __int64 *)&v131, 0);
                    v82 = 0x2E656772616C206FLL;
                    v142 = v87;
                    v144 = v131.m128i_i64[0];
                    *(__m128i *)v87 = _mm_load_si128((const __m128i *)&xmmword_43970A0);
                    v88 = _mm_load_si128((const __m128i *)&xmmword_4397100);
                    *(_QWORD *)(v87 + 80) = 0x2E656772616C206FLL;
                    *(__m128i *)(v87 + 16) = v88;
                    *(__m128i *)(v87 + 32) = _mm_load_si128((const __m128i *)&xmmword_4397110);
                    *(__m128i *)(v87 + 48) = _mm_load_si128((const __m128i *)&xmmword_4397120);
                    *(__m128i *)(v87 + 64) = _mm_load_si128((const __m128i *)&xmmword_4397130);
                    goto LABEL_112;
                  }
LABEL_111:
                  v142 = (unsigned __int64)&v144;
                  v131.m128i_i64[0] = 37;
                  v81 = sub_22409D0((__int64)&v142, (unsigned __int64 *)&v131, 0);
                  v142 = v81;
                  v144 = v131.m128i_i64[0];
                  *(__m128i *)v81 = _mm_load_si128((const __m128i *)&xmmword_43970E0);
                  v85 = _mm_load_si128((const __m128i *)&xmmword_43970F0);
                  *(_DWORD *)(v81 + 32) = 1819045746;
                  *(_BYTE *)(v81 + 36) = 46;
                  *(__m128i *)(v81 + 16) = v85;
LABEL_112:
                  v143 = v131.m128i_i64[0];
                  *(_BYTE *)(v142 + v131.m128i_i64[0]) = 0;
                  sub_2882C00(v6, v8, (__int64)&v142, v82, v83, v84);
                  if ( (__int64 *)v142 != &v144 )
                    j_j___libc_free_0(v142);
                  goto LABEL_38;
                }
                v56 = (unsigned int)qword_5001F08;
                v59 = 4 * v86 / v113;
                if ( !(4 * v86 / v113) )
                {
LABEL_84:
                  v55 = v57;
                  goto LABEL_85;
                }
              }
              _BitScanReverse64(&v59, v59);
              v58 = 63 - ((unsigned int)v59 ^ 0x3F);
              v60 = 1LL << (63 - ((unsigned __int8)v59 ^ 0x3Fu));
              if ( v57 <= (unsigned int)v60 )
              {
                v57 = v60;
                if ( (unsigned int)v56 <= (unsigned int)v60 )
                  v57 = v56;
              }
              goto LABEL_84;
            }
            v22 = v115;
            goto LABEL_115;
          }
LABEL_28:
          v18 = *(_QWORD *)(v125 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v18 == v125 + 48 )
          {
            v20 = 0;
            goto LABEL_32;
          }
          if ( v18 )
          {
            v19 = *(unsigned __int8 *)(v18 - 24);
            v20 = 0;
            v21 = v18 - 24;
            if ( (unsigned int)(v19 - 30) < 0xB )
              v20 = v21;
LABEL_32:
            v22 = (unsigned int)sub_B46E30(v20) > 2;
            HIDWORD(v107) = *(_DWORD *)(a2 + 32);
            LODWORD(v107) = *(_DWORD *)(a2 + 36);
            v118 = *(_DWORD *)(a2 + 44);
            if ( *(_DWORD *)(a2 + 8) )
            {
              v23 = *(_DWORD *)(a2 + 40);
              v24 = *(_DWORD *)(a2 + 48);
              v25 = *(_DWORD *)(a2 + 52);
LABEL_116:
              v108 = v23;
              v106 = v25 + v24;
              v113 = v110;
              if ( !v22 )
              {
                v115 = 0;
                v52 = v118 == 0;
                goto LABEL_69;
              }
              v14 = *(__int64 **)(v8 + 32);
              v105 = *(__int64 **)(v8 + 40);
              v114 = *(_DWORD *)(a2 + 24) != 0;
              if ( v105 == v14 )
              {
LABEL_68:
                v52 = v118 == 0;
                v115 = v118 == 0 && v114;
                goto LABEL_69;
              }
              goto LABEL_44;
            }
            goto LABEL_123;
          }
LABEL_152:
          BUG();
        }
      }
    }
    v142 = (unsigned __int64)&v144;
    v131.m128i_i64[0] = 37;
    v26 = sub_22409D0((__int64)&v142, (unsigned __int64 *)&v131, 0);
    v142 = v26;
    v144 = v131.m128i_i64[0];
    *(__m128i *)v26 = _mm_load_si128((const __m128i *)&xmmword_43970E0);
    v30 = _mm_load_si128((const __m128i *)&xmmword_43970F0);
    *(_DWORD *)(v26 + 32) = 1819045746;
    *(_BYTE *)(v26 + 36) = 46;
    *(__m128i *)(v26 + 16) = v30;
    v143 = v131.m128i_i64[0];
    *(_BYTE *)(v142 + v131.m128i_i64[0]) = 0;
  }
  else
  {
    v131.m128i_i64[0] = 47;
    v142 = (unsigned __int64)&v144;
    v32 = sub_22409D0((__int64)&v142, (unsigned __int64 *)&v131, 0);
    v27 = 0x2064656C62617369LL;
    v142 = v32;
    v144 = v131.m128i_i64[0];
    *(__m128i *)v32 = _mm_load_si128((const __m128i *)&xmmword_43970A0);
    v33 = _mm_load_si128((const __m128i *)&xmmword_43970B0);
    qmemcpy((void *)(v32 + 32), "isabled by flag", 15);
    *(__m128i *)(v32 + 16) = v33;
    v143 = v131.m128i_i64[0];
    *(_BYTE *)(v142 + v131.m128i_i64[0]) = 0;
  }
  sub_2882C00(v6, a1, (__int64)&v142, v27, v28, v29);
  if ( (__int64 *)v142 != &v144 )
    j_j___libc_free_0(v142);
  v123 = 0;
LABEL_38:
  if ( v124 )
    sub_B91220((__int64)&v124, v124);
  return v123;
}
