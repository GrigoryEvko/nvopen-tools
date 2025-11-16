// Function: sub_33FF780
// Address: 0x33ff780
//
unsigned __int8 *__fastcall sub_33FF780(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7,
        unsigned __int8 a8)
{
  __int16 v8; // r15
  unsigned __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rax
  __m128i *v14; // r13
  int v15; // r12d
  int v16; // edx
  __int64 v17; // r9
  __int64 v18; // r9
  __int64 v19; // rax
  int v20; // r8d
  unsigned __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // rax
  int v24; // r8d
  __int64 v25; // rax
  int v26; // r9d
  __int64 v27; // r15
  __int16 v28; // ax
  unsigned __int8 *v29; // r13
  void *v30; // rdi
  __int64 v32; // r9
  __int64 v33; // r10
  __int64 (__fastcall *v34)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  unsigned __int32 v41; // r12d
  unsigned __int64 v42; // rbx
  __int64 v43; // rsi
  unsigned int v44; // ebx
  unsigned int v45; // r12d
  unsigned int v46; // r14d
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // r8
  __int64 *v52; // rdx
  int v54; // r12d
  void *v55; // r9
  int v56; // r13d
  unsigned int v57; // eax
  __int64 v58; // rdi
  int v59; // ebx
  unsigned __int64 v60; // rdx
  __int64 v61; // r8
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 *v65; // rdi
  int v66; // r10d
  int v67; // r9d
  void *v68; // rsi
  __int64 v69; // r8
  int v70; // edx
  __int64 (__fastcall *v71)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v72; // rsi
  __int64 v73; // r9
  unsigned int v74; // eax
  __int64 v75; // rdx
  void *v76; // rax
  __int64 v77; // rdx
  unsigned int v78; // eax
  __int64 *v79; // rdi
  void *v80; // rax
  __int64 v81; // rdx
  unsigned int v82; // eax
  unsigned __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdx
  unsigned __int64 v86; // rax
  __int64 v87; // r9
  unsigned int v88; // r13d
  int v89; // ebx
  unsigned int v90; // r15d
  __int64 v91; // r8
  __int64 v92; // rax
  __int64 v93; // rdx
  unsigned __int64 v94; // rdx
  __int64 *v95; // rax
  _QWORD *v96; // rsi
  __int64 v97; // rcx
  unsigned __int8 *v98; // rax
  __int64 v99; // rdx
  int v100; // r9d
  __int64 v101; // rdx
  const __m128i *v102; // rdx
  __m128i *v103; // rax
  __int64 v104; // rdx
  __int128 v105; // [rsp-10h] [rbp-1B0h]
  __int128 v106; // [rsp-10h] [rbp-1B0h]
  _TBYTE v107; // [rsp+Eh] [rbp-192h]
  __int64 v108; // [rsp+18h] [rbp-188h]
  unsigned int v109; // [rsp+20h] [rbp-180h]
  __int64 v110; // [rsp+20h] [rbp-180h]
  __int64 v111; // [rsp+20h] [rbp-180h]
  __int64 v112; // [rsp+28h] [rbp-178h]
  __int64 v113; // [rsp+28h] [rbp-178h]
  __int64 *v114; // [rsp+30h] [rbp-170h]
  __int64 *v115; // [rsp+38h] [rbp-168h]
  int v116; // [rsp+38h] [rbp-168h]
  int v117; // [rsp+38h] [rbp-168h]
  int v119; // [rsp+40h] [rbp-160h]
  __int64 v120; // [rsp+40h] [rbp-160h]
  __int64 v121; // [rsp+40h] [rbp-160h]
  int v122; // [rsp+40h] [rbp-160h]
  unsigned int v123; // [rsp+48h] [rbp-158h]
  void *v124; // [rsp+48h] [rbp-158h]
  __int64 (*v125)(); // [rsp+50h] [rbp-150h]
  __m128i v127; // [rsp+70h] [rbp-130h] BYREF
  __int64 *v128; // [rsp+88h] [rbp-118h] BYREF
  __int64 v129; // [rsp+90h] [rbp-110h] BYREF
  __int64 v130; // [rsp+98h] [rbp-108h]
  __m128i v131; // [rsp+A0h] [rbp-100h] BYREF
  void *src; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v133; // [rsp+B8h] [rbp-E8h]
  _BYTE v134[32]; // [rsp+C0h] [rbp-E0h] BYREF
  _QWORD *v135; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v136; // [rsp+E8h] [rbp-B8h]
  _QWORD v137[22]; // [rsp+F0h] [rbp-B0h] BYREF

  v8 = a4;
  v11 = a2;
  v127.m128i_i64[0] = a4;
  v127.m128i_i64[1] = a5;
  if ( (_WORD)a4 )
  {
    if ( (unsigned __int16)(a4 - 17) > 0xD3u )
    {
LABEL_3:
      v12 = v127.m128i_i64[1];
      goto LABEL_4;
    }
    v12 = 0;
    v8 = word_4456580[(unsigned __int16)a4 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v127) )
      goto LABEL_3;
    v8 = sub_3009970((__int64)&v127, a2, v62, v63, v64);
  }
LABEL_4:
  v13 = *(_QWORD *)(a2 + 8);
  v130 = v12;
  LOWORD(v129) = v8;
  if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
    v11 = sub_ACCFD0(*(__int64 **)(a1 + 64), a2 + 24);
  if ( v127.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v127.m128i_i16[0] - 17) > 0xD3u )
      goto LABEL_8;
  }
  else if ( !sub_30070B0((__int64)&v127) )
  {
    goto LABEL_8;
  }
  sub_2FE6CC0((__int64)&v135, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 64), v129, v130);
  if ( (_BYTE)v135 == 1 )
  {
    v71 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**(_QWORD **)(a1 + 16) + 592LL);
    if ( v71 == sub_2D56A50 )
    {
      v72 = *(_QWORD *)(a1 + 16);
      sub_2FE6CC0((__int64)&v135, v72, *(_QWORD *)(a1 + 64), v129, v130);
      LOWORD(v129) = v136;
      v130 = v137[0];
    }
    else
    {
      v72 = *(_QWORD *)(a1 + 64);
      LODWORD(v129) = v71(*(_QWORD *)(a1 + 16), v72, v129, v130);
      v130 = v99;
    }
    v73 = *(_QWORD *)(a1 + 16);
    v131.m128i_i32[2] = 1;
    v131.m128i_i64[0] = 0;
    v121 = v73;
    v125 = *(__int64 (**)())(*(_QWORD *)v73 + 1456LL);
    LOWORD(v74) = sub_3281100((unsigned __int16 *)&v127, v72);
    if ( v125 == sub_2D56680
      || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v125)(
            v121,
            v74,
            v75,
            (unsigned int)v129,
            v130) )
    {
      v76 = (void *)sub_2D5B750((unsigned __int16 *)&v129);
      v133 = v77;
      src = v76;
      v78 = sub_CA1930(&src);
      sub_C44AB0((__int64)&v135, v11 + 24, v78);
      if ( v131.m128i_i32[2] > 0x40u )
        goto LABEL_85;
    }
    else
    {
      v80 = (void *)sub_2D5B750((unsigned __int16 *)&v129);
      v133 = v81;
      src = v80;
      v82 = sub_CA1930(&src);
      sub_C44B10((__int64)&v135, (char **)(v11 + 24), v82);
      if ( v131.m128i_i32[2] > 0x40u )
      {
LABEL_85:
        if ( v131.m128i_i64[0] )
          j_j___libc_free_0_0(v131.m128i_u64[0]);
      }
    }
    v79 = *(__int64 **)(a1 + 64);
    v131.m128i_i64[0] = (__int64)v135;
    v131.m128i_i32[2] = v136;
    v11 = sub_ACCFD0(v79, (__int64)&v131);
    if ( v131.m128i_i32[2] > 0x40u && v131.m128i_i64[0] )
      j_j___libc_free_0_0(v131.m128i_u64[0]);
    goto LABEL_8;
  }
  if ( !*(_BYTE *)(a1 + 762) )
    goto LABEL_8;
  if ( v127.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v127.m128i_i16[0] - 17) > 0xD3u )
      goto LABEL_8;
  }
  else if ( !sub_30070B0((__int64)&v127) )
  {
    goto LABEL_8;
  }
  sub_2FE6CC0((__int64)&v135, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 64), v129, v130);
  if ( (_BYTE)v135 == 2 )
  {
    v32 = *(_QWORD *)(a1 + 16);
    v33 = *(_QWORD *)(a1 + 64);
    v115 = (__int64 *)(v11 + 24);
    v34 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v32 + 592LL);
    if ( v34 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v135, v32, v33, v129, v130);
      v131.m128i_i16[0] = v136;
      v131.m128i_i64[1] = v137[0];
    }
    else
    {
      v131.m128i_i32[0] = v34(v32, v33, v129, v130);
      v131.m128i_i64[1] = v101;
    }
    v135 = (_QWORD *)sub_2D5B750((unsigned __int16 *)&v131);
    v136 = v35;
    v109 = sub_CA1930(&v135);
    if ( v127.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v127.m128i_i16[0] - 176) > 0x34u )
      {
        if ( (v36 = *(_QWORD *)(a1 + 16), v37 = 1, v127.m128i_i16[0] != 1)
          && (v37 = v127.m128i_u16[0], !*(_QWORD *)(v36 + 8LL * v127.m128i_u16[0] + 112))
          || *(_BYTE *)(v36 + 500 * v37 + 6582) )
        {
LABEL_33:
          v38 = sub_2D5B750((unsigned __int16 *)&v127);
          v136 = v39;
          v135 = (_QWORD *)v38;
          v40 = sub_CA1930(&v135);
          v41 = v131.m128i_i32[0];
          v123 = v40 / v109;
          v42 = v40 / v109;
          v43 = (unsigned int)v42;
          v114 = *(__int64 **)(a1 + 64);
          v108 = v131.m128i_i64[1];
          HIWORD(v107) = 0;
          *(_QWORD *)&v107 = (unsigned __int16)sub_2D43050(v131.m128i_i16[0], v42);
          if ( !LOWORD(v107) )
          {
            v43 = v41;
            LOWORD(v107) = sub_3009400(v114, v41, v108, (unsigned int)v42, 0);
            *(_QWORD *)((char *)&v107 + 2) = v104;
          }
          v44 = 0;
          v45 = 0;
          src = v134;
          v133 = 0x200000000LL;
          v46 = v109;
          while ( v123 / (unsigned int)sub_3281500(&v127, v43) > v45 )
          {
            sub_C440A0((__int64)&v135, v115, v46, v44);
            v43 = (__int64)&v135;
            v47 = sub_34007B0(a1, (unsigned int)&v135, a3, v131.m128i_i32[0], v131.m128i_i32[2], a6, a8);
            v49 = v48;
            v50 = (unsigned int)v133;
            v51 = v47;
            if ( (unsigned __int64)(unsigned int)v133 + 1 > HIDWORD(v133) )
            {
              v43 = (__int64)v134;
              v111 = v47;
              v113 = v49;
              sub_C8D5F0((__int64)&src, v134, (unsigned int)v133 + 1LL, 0x10u, v47, v49);
              v50 = (unsigned int)v133;
              v51 = v111;
              v49 = v113;
            }
            v52 = (__int64 *)((char *)src + 16 * v50);
            *v52 = v51;
            v52[1] = v49;
            LODWORD(v133) = v133 + 1;
            if ( (unsigned int)v136 > 0x40 && v135 )
              j_j___libc_free_0_0((unsigned __int64)v135);
            ++v45;
            v44 += v46;
          }
          if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1 + 40)) )
          {
            v102 = (const __m128i *)src;
            v103 = (__m128i *)((char *)src + 16 * (unsigned int)v133);
            if ( src != v103 )
            {
              while ( v102 < --v103 )
              {
                a7 = _mm_loadu_si128(v102++);
                v102[-1].m128i_i64[0] = v103->m128i_i64[0];
                v102[-1].m128i_i32[2] = v103->m128i_i32[2];
                v103->m128i_i64[0] = a7.m128i_i64[0];
                v103->m128i_i32[2] = a7.m128i_i32[2];
              }
            }
          }
          v54 = 0;
          v135 = v137;
          v136 = 0x800000000LL;
          v56 = sub_3281500(&v127, v43);
          v57 = v136;
          if ( v56 )
          {
            do
            {
              v59 = v133;
              v58 = v57;
              v55 = src;
              v60 = v57 + (unsigned __int64)(unsigned int)v133;
              v61 = 16LL * (unsigned int)v133;
              if ( v60 > HIDWORD(v136) )
              {
                v120 = 16LL * (unsigned int)v133;
                v124 = src;
                sub_C8D5F0((__int64)&v135, v137, v60, 0x10u, v61, (__int64)src);
                v58 = (unsigned int)v136;
                v61 = v120;
                v55 = v124;
              }
              if ( v61 )
              {
                memcpy(&v135[2 * v58], v55, v61);
                LODWORD(v58) = v136;
              }
              ++v54;
              LODWORD(v136) = v58 + v59;
              v57 = v58 + v59;
            }
            while ( v56 != v54 );
          }
          else
          {
            v57 = v136;
          }
          *((_QWORD *)&v106 + 1) = v57;
          *(_QWORD *)&v106 = v135;
          sub_33FC220((_QWORD *)a1, 156, a3, LOWORD(v107), *(__int64 *)((char *)&v107 + 2), (__int64)v55, v106);
          v29 = sub_33FAF80(a1, 234, a3, v127.m128i_u32[0], v127.m128i_i64[1], v100, a7);
          if ( v135 != v137 )
            _libc_free((unsigned __int64)v135);
          v30 = src;
          if ( src != v134 )
            goto LABEL_19;
          return v29;
        }
      }
    }
    else if ( !sub_3007100((__int64)&v127) )
    {
      goto LABEL_33;
    }
    v84 = sub_2D5B750((unsigned __int16 *)&v129);
    v136 = v85;
    v135 = (_QWORD *)v84;
    v86 = sub_CA1930(&v135);
    v135 = v137;
    v136 = 0x200000000LL;
    v122 = v86 / v109;
    if ( v122 )
    {
      v88 = 0;
      v89 = 0;
      v90 = v109;
      do
      {
        sub_C440A0((__int64)&src, v115, v90, v88);
        v91 = sub_34007B0(a1, (unsigned int)&src, a3, v131.m128i_i32[0], v131.m128i_i32[2], a6, a8);
        v92 = (unsigned int)v136;
        v87 = v93;
        v94 = (unsigned int)v136 + 1LL;
        if ( v94 > HIDWORD(v136) )
        {
          v110 = v91;
          v112 = v87;
          sub_C8D5F0((__int64)&v135, v137, v94, 0x10u, v91, v87);
          v92 = (unsigned int)v136;
          v91 = v110;
          v87 = v112;
        }
        v95 = &v135[2 * v92];
        *v95 = v91;
        v95[1] = v87;
        LODWORD(v136) = v136 + 1;
        if ( (unsigned int)v133 > 0x40 && src )
          j_j___libc_free_0_0((unsigned __int64)src);
        ++v89;
        v88 += v90;
      }
      while ( v122 != v89 );
      v96 = v135;
      v97 = (unsigned int)v136;
    }
    else
    {
      v96 = v137;
      v97 = 0;
    }
    *((_QWORD *)&v105 + 1) = v97;
    *(_QWORD *)&v105 = v96;
    v98 = sub_33FC220((_QWORD *)a1, 169, a3, v127.m128i_u32[0], v127.m128i_i64[1], v87, v105);
    v30 = v135;
    v29 = v98;
    if ( v135 != v137 )
      goto LABEL_19;
    return v29;
  }
LABEL_8:
  v14 = sub_33ED250(a1, (unsigned int)v129, v130);
  v15 = a6 == 0 ? 11 : 35;
  v119 = v16;
  v135 = v137;
  v136 = 0x2000000000LL;
  sub_33C9670((__int64)&v135, v15, (unsigned __int64)v14, 0, 0, v17);
  v19 = (unsigned int)v136;
  v20 = v11;
  v21 = (unsigned int)v136 + 1LL;
  if ( v21 > HIDWORD(v136) )
  {
    sub_C8D5F0((__int64)&v135, v137, v21, 4u, (unsigned int)v11, v18);
    v19 = (unsigned int)v136;
    v20 = v11;
  }
  *((_DWORD *)v135 + v19) = v20;
  v22 = HIDWORD(v11);
  LODWORD(v136) = v136 + 1;
  v23 = (unsigned int)v136;
  if ( (unsigned __int64)(unsigned int)v136 + 1 > HIDWORD(v136) )
  {
    sub_C8D5F0((__int64)&v135, v137, (unsigned int)v136 + 1LL, 4u, v22, v18);
    v23 = (unsigned int)v136;
    LODWORD(v22) = HIDWORD(v11);
  }
  *((_DWORD *)v135 + v23) = v22;
  v24 = a8;
  LODWORD(v136) = v136 + 1;
  v25 = (unsigned int)v136;
  if ( (unsigned __int64)(unsigned int)v136 + 1 > HIDWORD(v136) )
  {
    sub_C8D5F0((__int64)&v135, v137, (unsigned int)v136 + 1LL, 4u, a8, v18);
    v25 = (unsigned int)v136;
    v24 = a8;
  }
  *((_DWORD *)v135 + v25) = v24;
  LODWORD(v136) = v136 + 1;
  v128 = 0;
  v27 = (__int64)sub_33CCCF0(a1, (__int64)&v135, a3, (__int64 *)&v128);
  if ( v27 )
  {
    v28 = v127.m128i_i16[0];
    if ( v127.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v127.m128i_i16[0] - 17) > 0xD3u )
      {
LABEL_17:
        v29 = (unsigned __int8 *)v27;
        goto LABEL_18;
      }
LABEL_58:
      v131 = _mm_load_si128(&v127);
      if ( (unsigned __int16)(v28 - 176) <= 0x34u )
      {
LABEL_59:
        if ( *(_DWORD *)(v27 + 24) == 51 )
        {
          src = 0;
          LODWORD(v133) = 0;
          v27 = (__int64)sub_33F17F0((_QWORD *)a1, 51, (__int64)&src, v131.m128i_u32[0], v131.m128i_i64[1]);
          if ( src )
            sub_B91220((__int64)&src, (__int64)src);
        }
        else
        {
          v27 = (__int64)sub_33FAF80(a1, 168, a3, v131.m128i_i64[0], v131.m128i_i64[1], v26, a7);
        }
        goto LABEL_61;
      }
      goto LABEL_53;
    }
    if ( !sub_30070B0((__int64)&v127) )
      goto LABEL_17;
    goto LABEL_52;
  }
  v27 = *(_QWORD *)(a1 + 416);
  v65 = (__int64 *)(a1 + 424);
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL) + 544LL) - 42) > 1 )
  {
    if ( v27 )
    {
      *(_QWORD *)(a1 + 416) = *(_QWORD *)v27;
    }
    else
    {
      v27 = sub_33E48B0(v65);
      if ( !v27 )
        goto LABEL_73;
    }
    v67 = v119;
    v69 = (__int64)v14;
    v70 = 0;
    src = 0;
LABEL_70:
    sub_33CA300(v27, v15, v70, (unsigned __int8 **)&src, v69, v67);
    if ( src )
      sub_B91220((__int64)&src, (__int64)src);
    *(_QWORD *)(v27 + 96) = v11;
    *(_BYTE *)(v27 + 32) = *(_BYTE *)(v27 + 32) & 0xF7 | (8 * (a8 & 1));
    goto LABEL_73;
  }
  v66 = 0;
  if ( !*(_DWORD *)(a1 + 72) )
    v66 = *(_DWORD *)(a3 + 8);
  if ( v27 )
  {
    *(_QWORD *)(a1 + 416) = *(_QWORD *)v27;
LABEL_67:
    v67 = v119;
    v68 = *(void **)a3;
    src = v68;
    if ( v68 )
    {
      v116 = v66;
      sub_B96E90((__int64)&src, (__int64)v68, 1);
      v67 = v119;
      v66 = v116;
    }
    v69 = (__int64)v14;
    v70 = v66;
    goto LABEL_70;
  }
  v117 = v66;
  v83 = sub_33E48B0(v65);
  v66 = v117;
  v27 = v83;
  if ( v83 )
    goto LABEL_67;
LABEL_73:
  sub_C657C0((__int64 *)(a1 + 520), (__int64 *)v27, v128, (__int64)off_4A367D0);
  sub_33CC420(a1, v27);
  v28 = v127.m128i_i16[0];
  if ( v127.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v127.m128i_i16[0] - 17) > 0xD3u )
      goto LABEL_61;
    goto LABEL_58;
  }
  if ( !sub_30070B0((__int64)&v127) )
    goto LABEL_61;
LABEL_52:
  v131 = _mm_load_si128(&v127);
  if ( sub_3007100((__int64)&v131) )
    goto LABEL_59;
LABEL_53:
  v27 = sub_32886A0(a1, v131.m128i_u32[0], v131.m128i_i64[1], a3, v27, 0);
LABEL_61:
  v29 = (unsigned __int8 *)v27;
LABEL_18:
  v30 = v135;
  if ( v135 != v137 )
LABEL_19:
    _libc_free((unsigned __int64)v30);
  return v29;
}
