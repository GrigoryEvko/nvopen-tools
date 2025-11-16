// Function: sub_3413BC0
// Address: 0x3413bc0
//
void __fastcall sub_3413BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r8
  unsigned int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 *v15; // rax
  __int64 *v16; // r14
  __int64 *v17; // r15
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int16 v23; // dx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  char v27; // al
  __int16 *v28; // rdx
  unsigned __int16 v29; // ax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  char v33; // al
  unsigned int v34; // eax
  __int64 v35; // rdx
  const void *v36; // r10
  __int64 v37; // r11
  size_t v38; // r9
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 *v41; // rcx
  __int64 v42; // r8
  _QWORD *v43; // rdi
  int v44; // r13d
  __int64 v45; // r14
  unsigned __int64 v46; // rbx
  __int64 *v47; // rax
  __int32 v48; // edx
  int v49; // ecx
  int v50; // ecx
  __m128i v51; // xmm0
  __int64 v52; // rax
  __int64 *v53; // rax
  __int64 *v54; // rbx
  __int64 v55; // rax
  __int64 **v56; // rdi
  __int64 **v57; // r12
  __int64 **v58; // rbx
  __int64 *v59; // rsi
  __int64 v60; // rax
  __int64 v61; // r13
  int v62; // edx
  __int64 v63; // rsi
  __int64 v64; // rdx
  int v65; // esi
  __int64 v66; // r10
  const void *v67; // r9
  __int64 v68; // rax
  size_t v69; // r8
  __int64 v70; // rax
  unsigned __int64 v71; // rbx
  __int64 *v72; // rcx
  int v73; // ebx
  __int64 v74; // r15
  _QWORD *v75; // r14
  __int64 *v76; // rdx
  __int32 v77; // esi
  int v78; // ecx
  __m128i v79; // xmm1
  __int64 v80; // rcx
  __int64 v81; // rax
  _BYTE *v82; // rdi
  __int64 v83; // r8
  char v84; // di
  __int64 *v85; // rdi
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rdx
  _QWORD *v90; // r14
  int v91; // r10d
  unsigned __int64 v92; // r9
  __int64 v93; // rsi
  const __m128i *v94; // rcx
  __m128i *v95; // rdx
  __int64 v96; // rax
  __int64 *v97; // rdi
  int i; // ecx
  __int64 v99; // [rsp-1F8h] [rbp-1F8h]
  __int64 *v100; // [rsp-1D8h] [rbp-1D8h]
  size_t v101; // [rsp-1D8h] [rbp-1D8h]
  __int64 *v102; // [rsp-1D0h] [rbp-1D0h]
  const void *v103; // [rsp-1D0h] [rbp-1D0h]
  __int64 v104; // [rsp-1C8h] [rbp-1C8h]
  int v105; // [rsp-1C0h] [rbp-1C0h]
  size_t v106; // [rsp-1C0h] [rbp-1C0h]
  int v107; // [rsp-1B8h] [rbp-1B8h]
  const void *v108; // [rsp-1B8h] [rbp-1B8h]
  int v109; // [rsp-1ACh] [rbp-1ACh]
  __int64 *v110; // [rsp-1A0h] [rbp-1A0h]
  __int64 v111; // [rsp-1A0h] [rbp-1A0h]
  int v112; // [rsp-1A0h] [rbp-1A0h]
  __int64 v113; // [rsp-1A0h] [rbp-1A0h]
  __int64 v114; // [rsp-1A0h] [rbp-1A0h]
  __int64 v115; // [rsp-190h] [rbp-190h]
  __int64 v116; // [rsp-190h] [rbp-190h]
  unsigned __int64 v117; // [rsp-190h] [rbp-190h]
  __int64 v118; // [rsp-190h] [rbp-190h]
  int v119; // [rsp-188h] [rbp-188h]
  int v120; // [rsp-184h] [rbp-184h]
  char v121; // [rsp-184h] [rbp-184h]
  __int64 v122; // [rsp-180h] [rbp-180h]
  unsigned int v123; // [rsp-168h] [rbp-168h]
  __int64 *v124; // [rsp-160h] [rbp-160h]
  __int64 v125; // [rsp-158h] [rbp-158h] BYREF
  char v126; // [rsp-150h] [rbp-150h]
  __int64 v127; // [rsp-148h] [rbp-148h]
  __int64 v128; // [rsp-140h] [rbp-140h]
  __int64 v129; // [rsp-138h] [rbp-138h]
  __int64 v130; // [rsp-130h] [rbp-130h]
  __int64 v131; // [rsp-128h] [rbp-128h] BYREF
  __int64 v132; // [rsp-120h] [rbp-120h]
  int v133; // [rsp-118h] [rbp-118h]
  __m128i v134; // [rsp-108h] [rbp-108h] BYREF
  __int64 v135; // [rsp-F8h] [rbp-F8h]
  __m128i v136; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v137; // [rsp-D8h] [rbp-D8h]
  __int64 **v138; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v139; // [rsp-C0h] [rbp-C0h]
  _BYTE v140[16]; // [rsp-B8h] [rbp-B8h] BYREF
  _BYTE *v141; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v142; // [rsp-A0h] [rbp-A0h]
  _QWORD v143[4]; // [rsp-98h] [rbp-98h] BYREF
  __int64 *v144; // [rsp-78h] [rbp-78h] BYREF
  __int64 v145; // [rsp-70h] [rbp-70h]
  _BYTE v146[104]; // [rsp-68h] [rbp-68h] BYREF

  if ( (*(_BYTE *)(a2 + 32) & 1) != 0 )
  {
    v138 = (__int64 **)v140;
    v139 = 0x200000000LL;
    v8 = *(_QWORD *)(a1 + 720);
    v9 = *(_QWORD *)(v8 + 696);
    v10 = *(unsigned int *)(v8 + 712);
    if ( (_DWORD)v10 )
    {
      v11 = (unsigned int)(v10 - 1);
      v12 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = v9 + 40LL * v12;
      v14 = *(_QWORD *)v13;
      if ( a2 != *(_QWORD *)v13 )
      {
        for ( i = 1; ; i = a6 )
        {
          if ( v14 == -4096 )
            return;
          a6 = (unsigned int)(i + 1);
          v12 = v11 & (i + v12);
          v13 = v9 + 40LL * v12;
          v14 = *(_QWORD *)v13;
          if ( a2 == *(_QWORD *)v13 )
            break;
        }
      }
      if ( v13 != v9 + 40 * v10 )
      {
        v15 = *(__int64 **)(v13 + 8);
        v124 = &v15[*(unsigned int *)(v13 + 16)];
        if ( v124 != v15 )
        {
          v16 = *(__int64 **)(v13 + 8);
          v122 = a1;
          while ( 1 )
          {
            v17 = (__int64 *)*v16;
            if ( *(_BYTE *)(*v16 + 62) )
              goto LABEL_31;
            v18 = *(_DWORD *)(a2 + 24);
            if ( v18 == 56 )
            {
              v60 = *(_QWORD *)(a2 + 40);
              v61 = *(_QWORD *)v60;
              v62 = *(_DWORD *)(*(_QWORD *)v60 + 24LL);
              if ( v62 == 11 || v62 == 35 )
                goto LABEL_31;
              v63 = *(_QWORD *)(v60 + 40);
              v116 = v63;
              LOBYTE(v13) = *(_DWORD *)(v63 + 24) == 11 || *(_DWORD *)(v63 + 24) == 35;
              v121 = v13;
              if ( (_BYTE)v13 )
              {
                v64 = *(_QWORD *)(v63 + 96);
                if ( *(_DWORD *)(v64 + 32) <= 0x40u )
                  v104 = *(_QWORD *)(v64 + 24);
                else
                  v104 = **(_QWORD **)(v64 + 24);
LABEL_45:
                v65 = *(_DWORD *)(v60 + 8);
                v66 = v17[5];
                v67 = (const void *)v17[1];
                v107 = *(_DWORD *)(v60 + 48);
                v68 = *v17;
                v105 = v65;
                v144 = (__int64 *)v146;
                v70 = 24 * v68;
                v69 = v70;
                v145 = 0x200000000LL;
                v71 = 0xAAAAAAAAAAAAAAABLL * (v70 >> 3);
                if ( (unsigned __int64)v70 > 0x30 )
                {
                  v101 = v70;
                  v103 = v67;
                  v113 = v66;
                  sub_C8D5F0((__int64)&v144, v146, 0xAAAAAAAAAAAAAAABLL * (v70 >> 3), 0x18u, v70, (__int64)v67);
                  v66 = v113;
                  v67 = v103;
                  v69 = v101;
                  v97 = &v144[3 * (unsigned int)v145];
                }
                else
                {
                  v72 = (__int64 *)v146;
                  if ( !v70 )
                    goto LABEL_47;
                  v97 = (__int64 *)v146;
                }
                v114 = v66;
                memcpy(v97, v67, v69);
                LODWORD(v70) = v145;
                v72 = v144;
                v66 = v114;
LABEL_47:
                v73 = v70 + v71;
                LODWORD(v145) = v73;
                if ( !v73 )
                {
                  v83 = 0;
LABEL_60:
                  v84 = 1;
                  if ( !*((_BYTE *)v17 + 61) )
                    v84 = v73 != v83;
                  v53 = sub_33E4BC0(
                          v122,
                          v17[4],
                          v66,
                          v72,
                          v83,
                          *((_BYTE *)v17 + 60),
                          (const void *)v17[3],
                          v17[2],
                          v17 + 6,
                          *((_DWORD *)v17 + 14),
                          v84);
                  goto LABEL_27;
                }
                v102 = v17;
                v100 = v16;
                v74 = 0;
                v75 = (_QWORD *)v66;
                while ( 2 )
                {
                  while ( 1 )
                  {
                    v76 = &v72[3 * v74];
                    v77 = *(_DWORD *)v76;
                    if ( !*(_DWORD *)v76 && a2 == v76[1] )
                      break;
                    if ( v73 == ++v74 )
                      goto LABEL_59;
                  }
                  v78 = *(_DWORD *)(v61 + 24);
                  if ( v78 == 15 || v78 == 39 )
                  {
                    v77 = 2;
                    v134.m128i_i32[2] = *(_DWORD *)(v61 + 96);
                  }
                  else
                  {
                    v134.m128i_i64[1] = v61;
                    v109 = v105;
                  }
                  v134.m128i_i32[0] = v77;
                  v79 = _mm_loadu_si128(&v134);
                  LODWORD(v135) = v109;
                  v80 = v135;
                  *(__m128i *)v76 = v79;
                  v76[2] = v80;
                  if ( v121 )
                  {
                    v141 = v143;
                    v142 = 0x300000000LL;
                    sub_AF6280((__int64)&v141, v104);
                    v81 = sub_B0DBA0(v75, v141, (unsigned int)v142, v74, 1);
                    v82 = v141;
                    v75 = (_QWORD *)v81;
                    if ( v141 != (_BYTE *)v143 )
                      goto LABEL_57;
                  }
                  else
                  {
                    v88 = sub_B0D320(v75);
                    v89 = (unsigned int)v145;
                    v90 = (_QWORD *)v88;
                    v141 = v143;
                    v91 = v74;
                    v142 = 0x300000003LL;
                    v92 = (unsigned int)v145 + 1LL;
                    v143[1] = (unsigned int)v145;
                    v132 = v116;
                    v143[2] = 34;
                    v143[0] = 4101;
                    LODWORD(v131) = 0;
                    v133 = v107;
                    if ( v92 > HIDWORD(v145) )
                    {
                      if ( v144 > &v131 || (v99 = (__int64)v144, &v131 >= &v144[3 * (unsigned int)v145]) )
                      {
                        sub_C8D5F0((__int64)&v144, v146, v92, 0x18u, (__int64)v144, v92);
                        v93 = (__int64)v144;
                        v89 = (unsigned int)v145;
                        v94 = (const __m128i *)&v131;
                        v91 = v74;
                      }
                      else
                      {
                        sub_C8D5F0((__int64)&v144, v146, v92, 0x18u, (__int64)v144, v92);
                        v93 = (__int64)v144;
                        v89 = (unsigned int)v145;
                        v91 = v74;
                        v94 = (const __m128i *)((char *)&v131 + (_QWORD)v144 - v99);
                      }
                    }
                    else
                    {
                      v93 = (__int64)v144;
                      v94 = (const __m128i *)&v131;
                    }
                    v95 = (__m128i *)(v93 + 24 * v89);
                    *v95 = _mm_loadu_si128(v94);
                    v95[1].m128i_i64[0] = v94[1].m128i_i64[0];
                    LODWORD(v145) = v145 + 1;
                    v96 = sub_B0DBA0(v90, v141, (unsigned int)v142, v91, 1);
                    v82 = v141;
                    v75 = (_QWORD *)v96;
                    if ( v141 != (_BYTE *)v143 )
LABEL_57:
                      _libc_free((unsigned __int64)v82);
                  }
                  ++v74;
                  v72 = v144;
                  if ( v73 == v74 )
                  {
LABEL_59:
                    v66 = (__int64)v75;
                    v17 = v102;
                    v83 = (unsigned int)v145;
                    v16 = v100;
                    goto LABEL_60;
                  }
                  continue;
                }
              }
              v121 = *((_BYTE *)v17 + 60);
              if ( !v121 )
                goto LABEL_45;
              if ( v124 == ++v16 )
              {
LABEL_32:
                v56 = v138;
                v57 = &v138[(unsigned int)v139];
                if ( v57 != v138 )
                {
                  v58 = v138;
                  do
                  {
                    v59 = *v58++;
                    sub_33F99B0(v122, v59, 0, v13, v11, a6);
                  }
                  while ( v57 != v58 );
                  v56 = v138;
                }
                if ( v56 != (__int64 **)v140 )
                  _libc_free((unsigned __int64)v56);
                return;
              }
            }
            else
            {
              if ( v18 == 216 )
              {
                v19 = *(_QWORD *)(a2 + 40);
                v20 = *(_QWORD *)v19;
                v21 = *(unsigned int *)(v19 + 8);
                v120 = v21;
                v22 = *(_QWORD *)(v20 + 48) + 16 * v21;
                v23 = *(_WORD *)v22;
                v24 = *(_QWORD *)(v22 + 8);
                LOWORD(v144) = v23;
                v145 = v24;
                if ( v23 )
                {
                  if ( v23 == 1 || (unsigned __int16)(v23 - 504) <= 7u )
LABEL_93:
                    BUG();
                  v87 = 16LL * (v23 - 1);
                  v26 = *(_QWORD *)&byte_444C4A0[v87];
                  v27 = byte_444C4A0[v87 + 8];
                }
                else
                {
                  v127 = sub_3007260((__int64)&v144);
                  v128 = v25;
                  v26 = v127;
                  v27 = v128;
                }
                v125 = v26;
                v28 = *(__int16 **)(a2 + 48);
                v126 = v27;
                v29 = *v28;
                v30 = *((_QWORD *)v28 + 1);
                LOWORD(v141) = v29;
                v142 = v30;
                if ( v29 )
                {
                  if ( v29 == 1 || (unsigned __int16)(v29 - 504) <= 7u )
                    goto LABEL_93;
                  v86 = 16LL * (v29 - 1);
                  v32 = *(_QWORD *)&byte_444C4A0[v86];
                  v33 = byte_444C4A0[v86 + 8];
                }
                else
                {
                  v129 = sub_3007260((__int64)&v141);
                  v130 = v31;
                  v32 = v129;
                  v33 = v130;
                }
                v131 = v32;
                LOBYTE(v132) = v33;
                v115 = v17[5];
                v123 = sub_CA1930(&v131);
                v34 = sub_CA1930(&v125);
                sub_AF4FD0(&v141, v34, v123, 0);
                v35 = *v17;
                v36 = (const void *)v17[1];
                v144 = (__int64 *)v146;
                v37 = v115;
                v145 = 0x200000000LL;
                v39 = 24 * v35;
                v38 = v39;
                v40 = 0xAAAAAAAAAAAAAAABLL * (v39 >> 3);
                if ( (unsigned __int64)v39 > 0x30 )
                {
                  v106 = v39;
                  v108 = v36;
                  v111 = v115;
                  v117 = 0xAAAAAAAAAAAAAAABLL * (v39 >> 3);
                  sub_C8D5F0((__int64)&v144, v146, v117, 0x18u, v40, v39);
                  LODWORD(v40) = v117;
                  v37 = v111;
                  v36 = v108;
                  v38 = v106;
                  v85 = &v144[3 * (unsigned int)v145];
                }
                else
                {
                  v41 = (__int64 *)v146;
                  if ( !v39 )
                  {
LABEL_16:
                    LODWORD(v145) = v39 + v40;
                    v42 = (unsigned int)(v39 + v40);
                    if ( (_DWORD)v42 )
                    {
                      v43 = (_QWORD *)v37;
                      v110 = v16;
                      v44 = v119;
                      v45 = v20;
                      v46 = 0;
                      do
                      {
                        while ( 1 )
                        {
                          v47 = &v41[3 * v46];
                          v48 = *(_DWORD *)v47;
                          if ( !*(_DWORD *)v47 && a2 == v47[1] )
                            break;
                          v42 = (unsigned int)v145;
                          if ( ++v46 >= (unsigned int)v145 )
                            goto LABEL_25;
                        }
                        v49 = *(_DWORD *)(v45 + 24);
                        if ( v49 == 15 || v49 == 39 )
                        {
                          v136.m128i_i32[2] = *(_DWORD *)(v45 + 96);
                          v48 = 2;
                        }
                        else
                        {
                          v136.m128i_i64[1] = v45;
                          v44 = v120;
                        }
                        v136.m128i_i32[0] = v48;
                        v50 = v46;
                        LODWORD(v137) = v44;
                        ++v46;
                        v51 = _mm_loadu_si128(&v136);
                        v47[2] = v137;
                        *(__m128i *)v47 = v51;
                        v52 = sub_B0DBA0(v43, &v141, 6, v50, 0);
                        v42 = (unsigned int)v145;
                        v41 = v144;
                        v43 = (_QWORD *)v52;
                      }
                      while ( v46 < (unsigned int)v145 );
LABEL_25:
                      v119 = v44;
                      v37 = (__int64)v43;
                      v16 = v110;
                    }
                    v53 = sub_33E4BC0(
                            v122,
                            v17[4],
                            v37,
                            v41,
                            v42,
                            *((_BYTE *)v17 + 60),
                            (const void *)v17[3],
                            v17[2],
                            v17 + 6,
                            *((_DWORD *)v17 + 14),
                            *((_BYTE *)v17 + 61));
LABEL_27:
                    v13 = HIDWORD(v139);
                    v54 = v53;
                    v55 = (unsigned int)v139;
                    if ( (unsigned __int64)(unsigned int)v139 + 1 > HIDWORD(v139) )
                    {
                      sub_C8D5F0((__int64)&v138, v140, (unsigned int)v139 + 1LL, 8u, v11, a6);
                      v55 = (unsigned int)v139;
                    }
                    v138[v55] = v54;
                    *((_WORD *)v17 + 31) = 257;
                    LODWORD(v139) = v139 + 1;
                    if ( v144 != (__int64 *)v146 )
                      _libc_free((unsigned __int64)v144);
                    goto LABEL_31;
                  }
                  v85 = (__int64 *)v146;
                }
                v112 = v40;
                v118 = v37;
                memcpy(v85, v36, v38);
                LODWORD(v39) = v145;
                v41 = v144;
                LODWORD(v40) = v112;
                v37 = v118;
                goto LABEL_16;
              }
LABEL_31:
              if ( v124 == ++v16 )
                goto LABEL_32;
            }
          }
        }
      }
    }
  }
}
