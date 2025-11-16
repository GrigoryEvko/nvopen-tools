// Function: sub_3288BD0
// Address: 0x3288bd0
//
__int64 __fastcall sub_3288BD0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned __int8 a8)
{
  unsigned int v8; // r11d
  __int64 v13; // r14
  __int64 *v15; // rax
  unsigned __int64 v16; // r13
  __int64 v17; // rsi
  char v18; // al
  __int64 v19; // r9
  unsigned int v20; // r11d
  char v21; // r13
  __int16 v22; // dx
  __int64 v23; // r14
  unsigned __int16 i; // r13
  __int32 v25; // r8d
  unsigned int v26; // eax
  unsigned __int16 v27; // bx
  unsigned int v28; // r13d
  unsigned __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rdx
  int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int32 v38; // edx
  bool v39; // al
  int v40; // eax
  __int64 v41; // rax
  int v42; // edx
  int v43; // eax
  _BYTE *v44; // rbx
  unsigned __int64 v45; // r12
  bool v46; // al
  __int64 v47; // rcx
  __int64 v48; // r8
  __int16 v49; // ax
  int v50; // edx
  int v51; // r13d
  _QWORD *v52; // rsi
  __int64 v53; // rcx
  __int64 *v54; // rax
  __int64 v55; // rdi
  unsigned int v56; // eax
  int v57; // esi
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  int v60; // edx
  __int16 v61; // bx
  __int64 v62; // rdi
  unsigned int v63; // eax
  int v64; // esi
  unsigned __int64 v65; // rax
  __int64 v66; // rbx
  int v67; // r9d
  __int64 v68; // rdx
  __int64 v69; // r13
  __int64 v70; // rbx
  int v71; // eax
  __int128 v72; // rax
  int v73; // r9d
  __int64 v74; // rax
  __int64 v75; // r14
  unsigned __int64 v76; // rbx
  unsigned int *v77; // rax
  unsigned int v78; // r8d
  __int64 v79; // r13
  __int64 v80; // rax
  unsigned __int16 v81; // dx
  __int64 v82; // rax
  unsigned __int64 v83; // rdx
  __int64 v84; // rax
  __int128 v85; // rax
  __int128 v86; // rax
  __int64 v87; // rdx
  __int128 v88; // rax
  int v89; // r9d
  char v90; // al
  __int64 v91; // rdx
  __m128i v92; // rax
  __int128 v93; // [rsp-10h] [rbp-150h]
  __int128 v94; // [rsp-10h] [rbp-150h]
  __int128 v95; // [rsp-10h] [rbp-150h]
  __int128 v96; // [rsp+0h] [rbp-140h]
  unsigned int v97; // [rsp+20h] [rbp-120h]
  unsigned int v98; // [rsp+20h] [rbp-120h]
  unsigned int v99; // [rsp+20h] [rbp-120h]
  __int32 v100; // [rsp+20h] [rbp-120h]
  unsigned int v101; // [rsp+20h] [rbp-120h]
  unsigned int v102; // [rsp+20h] [rbp-120h]
  unsigned int v103; // [rsp+20h] [rbp-120h]
  __int64 v104; // [rsp+20h] [rbp-120h]
  __int64 v105; // [rsp+28h] [rbp-118h]
  _BYTE *v106; // [rsp+30h] [rbp-110h]
  __int128 v107; // [rsp+30h] [rbp-110h]
  __int128 v108; // [rsp+30h] [rbp-110h]
  unsigned __int128 v109; // [rsp+40h] [rbp-100h] BYREF
  __m128i si128; // [rsp+50h] [rbp-F0h] BYREF
  unsigned __int16 v111; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v112; // [rsp+68h] [rbp-D8h]
  __m128i v113; // [rsp+70h] [rbp-D0h] BYREF
  __int64 (__fastcall *v114)(__m128i *, __m128i *, int); // [rsp+80h] [rbp-C0h]
  __int64 (__fastcall *v115)(__int64 *, __int64, __int64, __int64, __int64, __int64); // [rsp+88h] [rbp-B8h]
  _BYTE *v116; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v117; // [rsp+98h] [rbp-A8h]
  _BYTE v118[48]; // [rsp+A0h] [rbp-A0h] BYREF
  _QWORD *v119; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v120; // [rsp+D8h] [rbp-68h]
  _QWORD v121[12]; // [rsp+E0h] [rbp-60h] BYREF

  v8 = a6;
  v109 = __PAIR128__(a4, a3);
  if ( (_WORD)a3 )
  {
    if ( (unsigned __int16)(a3 - 176) <= 0x34u )
      return 0;
  }
  else
  {
    v99 = a6;
    v39 = sub_3007100((__int64)&v109);
    v8 = v99;
    if ( v39 )
      return 0;
  }
  while ( ((*(_DWORD *)(a5 + 24) - 214) & 0xFFFFFFFD) == 0 )
  {
    v15 = *(__int64 **)(a5 + 40);
    a5 = *v15;
    v8 = *((_DWORD *)v15 + 2);
  }
  v113.m128i_i64[0] = (__int64)&v116;
  v115 = sub_326FEC0;
  v116 = v118;
  v114 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_325DB30;
  v97 = v8;
  v117 = 0x300000000LL;
  v16 = v8 | a6 & 0xFFFFFFFF00000000LL;
  v121[0] = 0;
  sub_325DB30(&v119, &v113, 2);
  v17 = v16;
  v121[1] = v115;
  v121[0] = v114;
  v18 = sub_33CA8D0(a5, v16, &v119);
  v20 = v97;
  v21 = v18;
  if ( v121[0] )
  {
    v17 = (__int64)&v119;
    ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v121[0])(&v119, &v119, 3);
    v20 = v97;
  }
  if ( v114 )
  {
    v17 = (__int64)&v113;
    v98 = v20;
    v114(&v113, &v113, 3);
    v20 = v98;
  }
  if ( !v21 )
  {
    if ( a7 == 6 )
      goto LABEL_46;
    v40 = *(_DWORD *)(a5 + 24);
    if ( v40 == 190 )
    {
      if ( a8
        || (v71 = *(_DWORD *)(a5 + 28), (v71 & 1) != 0)
        || (v71 & 2) != 0
        || (v103 = v20,
            v90 = sub_33CF4D0(**(_QWORD **)(a5 + 40), *(_QWORD *)(*(_QWORD *)(a5 + 40) + 8LL)),
            v20 = v103,
            v90) )
      {
        v101 = v20;
        *(_QWORD *)&v72 = sub_3288BD0(
                            a1,
                            a2,
                            v109,
                            DWORD2(v109),
                            **(_QWORD **)(a5 + 40),
                            *(_QWORD *)(*(_QWORD *)(a5 + 40) + 8LL),
                            a7 + 1,
                            a8);
        v20 = v101;
        v96 = v72;
        if ( (_QWORD)v72 )
        {
          v74 = *(_QWORD *)(a5 + 40);
          v75 = *(_QWORD *)(v74 + 40);
          v76 = *(_QWORD *)(v74 + 48);
          si128 = _mm_load_si128((const __m128i *)&v109);
          if ( *(_DWORD *)(v75 + 24) == 214 )
          {
            do
            {
              v77 = *(unsigned int **)(v75 + 40);
              v75 = *(_QWORD *)v77;
              v76 = v77[2] | v76 & 0xFFFFFFFF00000000LL;
            }
            while ( *(_DWORD *)(*(_QWORD *)v77 + 24LL) == 214 );
            v78 = v77[2];
          }
          else
          {
            v78 = *(_DWORD *)(v74 + 48);
          }
          v79 = v78;
          v80 = *(_QWORD *)(v75 + 48) + 16LL * v78;
          v81 = *(_WORD *)v80;
          v82 = *(_QWORD *)(v80 + 8);
          v111 = v81;
          v112 = v82;
          if ( v81 == si128.m128i_i16[0] && (v81 || v82 == si128.m128i_i64[1]) )
          {
            v83 = v78 | v76 & 0xFFFFFFFF00000000LL;
          }
          else
          {
            v119 = (_QWORD *)sub_2D5B750(&v111);
            v120 = v91;
            v92.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&si128);
            v113 = v92;
            if ( (_QWORD *)v92.m128i_i64[0] == v119 && v113.m128i_i8[8] == (_BYTE)v120 )
            {
              v75 = sub_33FB890(a1, si128.m128i_u32[0], si128.m128i_i64[1], v75, v79 | v76 & 0xFFFFFFFF00000000LL);
              v79 = (unsigned int)v83;
            }
            else
            {
              v75 = sub_33FB310(a1, v75, v79 | v76 & 0xFFFFFFFF00000000LL, a2, si128.m128i_u32[0], si128.m128i_i64[1]);
              v79 = (unsigned int)v83;
            }
          }
          *((_QWORD *)&v95 + 1) = v79 | v83 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v95 = v75;
          v84 = sub_3406EB0(a1, 56, a2, v109, DWORD2(v109), v73, v96, v95);
          goto LABEL_99;
        }
      }
      v40 = *(_DWORD *)(a5 + 24);
    }
    if ( (unsigned int)(v40 - 205) <= 1 )
    {
      v41 = *(_QWORD *)(a5 + 56);
      if ( !v41 )
        goto LABEL_46;
      v42 = 1;
      do
      {
        if ( *(_DWORD *)(v41 + 8) == v20 )
        {
          if ( !v42 )
            goto LABEL_46;
          v41 = *(_QWORD *)(v41 + 32);
          if ( !v41 )
            goto LABEL_101;
          if ( v20 == *(_DWORD *)(v41 + 8) )
            goto LABEL_46;
          v42 = 0;
        }
        v41 = *(_QWORD *)(v41 + 32);
      }
      while ( v41 );
      if ( v42 == 1 )
        goto LABEL_46;
LABEL_101:
      v102 = v20;
      *(_QWORD *)&v85 = sub_3288BD0(
                          a1,
                          a2,
                          v109,
                          DWORD2(v109),
                          *(_QWORD *)(*(_QWORD *)(a5 + 40) + 40LL),
                          *(_QWORD *)(*(_QWORD *)(a5 + 40) + 48LL),
                          a7 + 1,
                          a8);
      v20 = v102;
      v107 = v85;
      if ( (_QWORD)v85 )
      {
        *(_QWORD *)&v86 = sub_3288BD0(
                            a1,
                            a2,
                            v109,
                            DWORD2(v109),
                            *(_QWORD *)(*(_QWORD *)(a5 + 40) + 80LL),
                            *(_QWORD *)(*(_QWORD *)(a5 + 40) + 88LL),
                            a7 + 1,
                            a8);
        v20 = v102;
        if ( (_QWORD)v86 )
        {
          v13 = sub_3288B20(
                  a1,
                  a2,
                  v109,
                  SDWORD2(v109),
                  **(_QWORD **)(a5 + 40),
                  *(_QWORD *)(*(_QWORD *)(a5 + 40) + 8LL),
                  v107,
                  v86,
                  0);
          goto LABEL_47;
        }
      }
      v40 = *(_DWORD *)(a5 + 24);
    }
    if ( (unsigned int)(v40 - 182) <= 1 )
    {
      v59 = *(_QWORD *)(a5 + 56);
      if ( v59 )
      {
        v60 = 1;
        do
        {
          if ( *(_DWORD *)(v59 + 8) == v20 )
          {
            if ( !v60 )
              goto LABEL_46;
            v59 = *(_QWORD *)(v59 + 32);
            if ( !v59 )
              goto LABEL_105;
            if ( v20 == *(_DWORD *)(v59 + 8) )
              goto LABEL_46;
            v60 = 0;
          }
          v59 = *(_QWORD *)(v59 + 32);
        }
        while ( v59 );
        if ( v60 != 1 )
        {
LABEL_105:
          *(_QWORD *)&v108 = sub_3288BD0(
                               a1,
                               a2,
                               v109,
                               DWORD2(v109),
                               **(_QWORD **)(a5 + 40),
                               *(_QWORD *)(*(_QWORD *)(a5 + 40) + 8LL),
                               a7 + 1,
                               0);
          *((_QWORD *)&v108 + 1) = v87;
          if ( (_QWORD)v108 )
          {
            *(_QWORD *)&v88 = sub_3288BD0(
                                a1,
                                a2,
                                v109,
                                DWORD2(v109),
                                *(_QWORD *)(*(_QWORD *)(a5 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a5 + 40) + 48LL),
                                a7 + 1,
                                0);
            if ( (_QWORD)v88 )
            {
              v84 = sub_3406EB0(a1, *(_DWORD *)(a5 + 24), a2, v109, DWORD2(v109), v89, v108, v88);
LABEL_99:
              v13 = v84;
              goto LABEL_47;
            }
          }
        }
      }
    }
LABEL_46:
    v13 = 0;
    goto LABEL_47;
  }
  v22 = v109;
  if ( !(_WORD)v109 )
  {
    v46 = sub_30070B0((__int64)&v109);
    v22 = 0;
    if ( v46 )
    {
      if ( *(_DWORD *)(a5 + 24) != 168 )
        goto LABEL_15;
      v49 = sub_3009970((__int64)&v109, v17, 0, v47, v48);
      v51 = v50;
LABEL_80:
      v61 = v49;
      v62 = (__int64)&v116[16 * (unsigned int)v117 - 16];
      v63 = *(_DWORD *)(v62 + 8);
      if ( v63 > 0x40 )
      {
        v64 = v63 - 1 - sub_C444A0(v62);
      }
      else
      {
        v64 = -1;
        if ( *(_QWORD *)v62 )
        {
          _BitScanReverse64(&v65, *(_QWORD *)v62);
          v64 = 63 - (v65 ^ 0x3F);
        }
      }
      v66 = sub_3400BD0(a1, v64, a2, v61, v51, 0, 0);
      v69 = v68;
      v113 = _mm_load_si128((const __m128i *)&v109);
      if ( (_WORD)v109 )
      {
        if ( (unsigned __int16)(v109 - 176) <= 0x34u )
        {
LABEL_85:
          if ( *(_DWORD *)(v66 + 24) == 51 )
          {
            v119 = 0;
            LODWORD(v120) = 0;
            v70 = sub_33F17F0(a1, 51, &v119, v113.m128i_i64[0], v113.m128i_i64[1]);
            if ( v119 )
              sub_B91220((__int64)&v119, (__int64)v119);
          }
          else
          {
            *((_QWORD *)&v94 + 1) = v69;
            *(_QWORD *)&v94 = v66;
            v70 = sub_33FAF80(a1, 168, a2, v113.m128i_i32[0], v113.m128i_i32[2], v67, v94);
          }
          v13 = v70;
          goto LABEL_47;
        }
      }
      else if ( sub_3007100((__int64)&v113) )
      {
        goto LABEL_85;
      }
      v13 = sub_32886A0(a1, v113.m128i_u32[0], v113.m128i_i64[1], a2, v66, v69);
      goto LABEL_47;
    }
LABEL_64:
    v55 = (__int64)&v116[16 * (unsigned int)v117 - 16];
    v56 = *(_DWORD *)(v55 + 8);
    if ( v56 > 0x40 )
    {
      v57 = v56 - 1 - sub_C444A0(v55);
    }
    else
    {
      v57 = -1;
      if ( *(_QWORD *)v55 )
      {
        _BitScanReverse64(&v58, *(_QWORD *)v55);
        v57 = 63 - (v58 ^ 0x3F);
      }
    }
    v13 = sub_3400BD0(a1, v57, a2, (_WORD)v109, DWORD2(v109), 0, 0);
    goto LABEL_47;
  }
  if ( (unsigned __int16)(v109 - 17) > 0xD3u )
    goto LABEL_64;
  if ( *(_DWORD *)(a5 + 24) == 168 )
  {
    v51 = 0;
    v49 = word_4456580[(unsigned __int16)v109 - 1];
    goto LABEL_80;
  }
LABEL_15:
  v23 = (__int64)v116;
  v119 = v121;
  v120 = 0x300000000LL;
  if ( v116 == &v116[16 * (unsigned int)v117] )
  {
    v52 = v121;
    v53 = 0;
    goto LABEL_59;
  }
  v106 = &v116[16 * (unsigned int)v117];
  for ( i = v22; ; i = v109 )
  {
    if ( i )
    {
      if ( (unsigned __int16)(i - 17) > 0xD3u )
        goto LABEL_18;
      v25 = 0;
      i = word_4456580[i - 1];
    }
    else
    {
      if ( !sub_30070B0((__int64)&v109) )
      {
LABEL_18:
        v25 = DWORD2(v109);
        goto LABEL_19;
      }
      i = sub_3009970((__int64)&v109, v17, v35, v36, v37);
      v25 = v38;
    }
LABEL_19:
    v26 = *(_DWORD *)(v23 + 8);
    v27 = i;
    v28 = v26 - 1;
    if ( v26 > 0x40 )
    {
      v100 = v25;
      v43 = sub_C444A0(v23);
      v25 = v100;
      v17 = v28 - v43;
    }
    else
    {
      v17 = 0xFFFFFFFFLL;
      if ( *(_QWORD *)v23 )
      {
        _BitScanReverse64(&v29, *(_QWORD *)v23);
        v17 = 63 - ((unsigned int)v29 ^ 0x3F);
      }
    }
    v30 = sub_3400BD0(a1, v17, a2, v27, v25, 0, 0);
    v31 = (unsigned int)v120;
    v19 = v32;
    v33 = v120;
    if ( (unsigned int)v120 >= (unsigned __int64)HIDWORD(v120) )
    {
      if ( HIDWORD(v120) < (unsigned __int64)(unsigned int)v120 + 1 )
      {
        v17 = (__int64)v121;
        v104 = v30;
        v105 = v19;
        sub_C8D5F0((__int64)&v119, v121, (unsigned int)v120 + 1LL, 0x10u, v30, v19);
        v31 = (unsigned int)v120;
        v30 = v104;
        v19 = v105;
      }
      v54 = &v119[2 * v31];
      *v54 = v30;
      v54[1] = v19;
      LODWORD(v120) = v120 + 1;
    }
    else
    {
      v34 = &v119[2 * (unsigned int)v120];
      if ( v34 )
      {
        *v34 = v30;
        v34[1] = v19;
        v33 = v120;
      }
      LODWORD(v120) = v33 + 1;
    }
    v23 += 16;
    if ( (_BYTE *)v23 == v106 )
      break;
  }
  v52 = v119;
  v53 = (unsigned int)v120;
LABEL_59:
  *((_QWORD *)&v93 + 1) = v53;
  *(_QWORD *)&v93 = v52;
  v13 = sub_33FC220(a1, 156, a2, v109, DWORD2(v109), v19, v93);
  if ( v119 != v121 )
    _libc_free((unsigned __int64)v119);
LABEL_47:
  v44 = v116;
  v45 = (unsigned __int64)&v116[16 * (unsigned int)v117];
  if ( v116 != (_BYTE *)v45 )
  {
    do
    {
      v45 -= 16LL;
      if ( *(_DWORD *)(v45 + 8) > 0x40u && *(_QWORD *)v45 )
        j_j___libc_free_0_0(*(_QWORD *)v45);
    }
    while ( v44 != (_BYTE *)v45 );
    v45 = (unsigned __int64)v116;
  }
  if ( (_BYTE *)v45 != v118 )
    _libc_free(v45);
  return v13;
}
