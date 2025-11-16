// Function: sub_15083D0
// Address: 0x15083d0
//
__int64 __fastcall sub_15083D0(__int64 a1, __m128i *a2, __int64 a3, char a4, char a5, unsigned __int8 a6)
{
  __m128i v7; // rdi
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  unsigned int v10; // eax
  char v11; // dl
  size_t v12; // rax
  __int64 v13; // r8
  _BYTE *v14; // r13
  _BYTE *v15; // r14
  __int64 v16; // rdi
  __int64 v17; // r13
  __int64 v18; // rbx
  volatile signed __int32 *v19; // r12
  signed __int32 v20; // eax
  signed __int32 v21; // eax
  __int64 v22; // r15
  __int64 v23; // r12
  volatile signed __int32 *v24; // r13
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  __int64 v28; // rdx
  char v29; // si
  unsigned __int64 v30; // rax
  unsigned int v31; // esi
  __m128i v32; // xmm1
  __m128i v33; // xmm0
  __int64 v34; // r14
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // r15
  __m128i v39; // xmm1
  __m128i v40; // xmm0
  __int64 v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // rdx
  int v44; // edi
  int v45; // eax
  __int64 v46; // r8
  __m128i v47; // xmm1
  __m128i v48; // xmm0
  _BYTE *v49; // r15
  __int64 v50; // r8
  __int64 v51; // rax
  __int64 v52; // r8
  __int64 v53; // r13
  __int64 v54; // rdi
  __int64 v55; // r14
  __int64 v56; // rbx
  volatile signed __int32 *v57; // r12
  signed __int32 v58; // eax
  signed __int32 v59; // eax
  __int64 v60; // r15
  __int64 v61; // r12
  volatile signed __int32 *v62; // r14
  signed __int32 v63; // eax
  signed __int32 v64; // eax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 *v67; // r12
  __int64 v68; // rax
  __m128i *v69; // rdi
  __m128i *p_src; // rax
  size_t v71; // rsi
  __int64 v72; // rcx
  __int64 v73; // rdi
  __int64 v74; // r8
  _BYTE *v75; // r15
  _BYTE *v76; // r13
  __int64 v77; // rdi
  __int64 v78; // r14
  __int64 v79; // rbx
  volatile signed __int32 *v80; // r12
  signed __int32 v81; // eax
  signed __int32 v82; // eax
  __int64 v83; // r15
  __int64 v84; // r12
  volatile signed __int32 *v85; // r14
  signed __int32 v86; // eax
  signed __int32 v87; // eax
  __int64 v88; // r15
  __int64 v89; // rbx
  __int64 v90; // rax
  __int64 v91; // r12
  __int64 v92; // rdi
  size_t v93; // rax
  char v94; // al
  size_t v95; // rdx
  unsigned __int64 v96; // r11
  unsigned int v97; // r10d
  unsigned __int64 *v98; // r9
  unsigned __int64 v99; // r8
  unsigned int v100; // r11d
  unsigned int v101; // esi
  __int64 v102; // r9
  __int64 v103; // rdx
  __int64 v104; // r8
  char v105; // cl
  int v106; // edx
  __int64 v107; // r10
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rdi
  char v111; // cl
  __m128i *v112; // [rsp+0h] [rbp-510h]
  __int64 v113; // [rsp+8h] [rbp-508h]
  __int64 v114; // [rsp+10h] [rbp-500h]
  _BYTE *v118; // [rsp+40h] [rbp-4D0h]
  __int64 v119; // [rsp+40h] [rbp-4D0h]
  __int64 v123; // [rsp+68h] [rbp-4A8h]
  __int64 v124; // [rsp+68h] [rbp-4A8h]
  __int64 v125; // [rsp+68h] [rbp-4A8h]
  _QWORD *v126; // [rsp+A0h] [rbp-470h] BYREF
  __int64 v127; // [rsp+A8h] [rbp-468h]
  _QWORD v128[2]; // [rsp+B0h] [rbp-460h] BYREF
  __m128i v129; // [rsp+C0h] [rbp-450h] BYREF
  __m128i v130; // [rsp+D0h] [rbp-440h] BYREF
  __int64 v131; // [rsp+E0h] [rbp-430h]
  __int64 v132; // [rsp+E8h] [rbp-428h]
  __int64 v133; // [rsp+F0h] [rbp-420h]
  __int64 v134; // [rsp+F8h] [rbp-418h]
  _BYTE *v135; // [rsp+100h] [rbp-410h] BYREF
  __int64 v136; // [rsp+108h] [rbp-408h]
  _BYTE v137[256]; // [rsp+110h] [rbp-400h] BYREF
  __int64 v138; // [rsp+210h] [rbp-300h]
  __m128i v139; // [rsp+220h] [rbp-2F0h] BYREF
  __m128i v140; // [rsp+230h] [rbp-2E0h] BYREF
  __int64 v141; // [rsp+240h] [rbp-2D0h]
  __int64 v142; // [rsp+248h] [rbp-2C8h]
  __int64 v143; // [rsp+250h] [rbp-2C0h]
  __int64 v144; // [rsp+258h] [rbp-2B8h]
  _BYTE *v145; // [rsp+260h] [rbp-2B0h] BYREF
  __int64 v146; // [rsp+268h] [rbp-2A8h]
  _BYTE v147[256]; // [rsp+270h] [rbp-2A0h] BYREF
  __int64 v148; // [rsp+370h] [rbp-1A0h]
  size_t n[2]; // [rsp+380h] [rbp-190h] BYREF
  __m128i src; // [rsp+390h] [rbp-180h] BYREF
  __int64 v151; // [rsp+3A0h] [rbp-170h]
  __int64 v152; // [rsp+3A8h] [rbp-168h]
  __int64 v153; // [rsp+3B0h] [rbp-160h]
  __int64 v154; // [rsp+3B8h] [rbp-158h]
  _BYTE *v155; // [rsp+3C0h] [rbp-150h] BYREF
  __int64 v156; // [rsp+3C8h] [rbp-148h]
  _BYTE v157[256]; // [rsp+3D0h] [rbp-140h] BYREF
  __int64 v158; // [rsp+4D0h] [rbp-40h]

  v131 = 0x200000000LL;
  v135 = v137;
  v136 = 0x800000000LL;
  v7 = *a2;
  v126 = v128;
  v8 = a2[3].m128i_u64[0];
  v129 = v7;
  v130.m128i_i64[1] = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v138 = 0;
  v127 = 0;
  LOBYTE(v128[0]) = 0;
  if ( v8 != -1 )
  {
    v9 = (v8 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
    v130.m128i_i64[0] = v9;
    v10 = v8 & 0x3F;
    if ( (v8 & 0x3F) != 0 )
    {
      if ( v7.m128i_i64[1] <= v9 )
        goto LABEL_181;
      v7.m128i_i64[0] += v9;
      if ( v7.m128i_i64[1] < v9 + 8 )
      {
        v101 = v7.m128i_i32[2] - v9;
        v102 = v101;
        v97 = 8 * v101;
        v103 = v101 + v9;
        if ( !v101 )
        {
          v130.m128i_i64[0] = v103;
          LODWORD(v131) = 0;
          goto LABEL_181;
        }
        v7.m128i_i64[1] = 0;
        v96 = 0;
        do
        {
          v104 = *(unsigned __int8 *)(v7.m128i_i64[0] + v7.m128i_i64[1]);
          v105 = 8 * v7.m128i_i8[8];
          ++v7.m128i_i64[1];
          v96 |= v104 << v105;
          v130.m128i_i64[1] = v96;
        }
        while ( v102 != v7.m128i_i64[1] );
        v130.m128i_i64[0] = v103;
        LODWORD(v131) = v97;
        if ( v10 > v97 )
LABEL_181:
          sub_16BD130("Unexpected end of file", 1);
      }
      else
      {
        v96 = *(_QWORD *)v7.m128i_i64[0];
        v130.m128i_i64[0] = v9 + 8;
        v97 = 64;
      }
      LODWORD(v131) = v97 - v10;
      v130.m128i_i64[1] = v96 >> v10;
    }
    sub_14EE150((__m128i *)n, (__int64)&v129);
    v11 = v151 & 1;
    LOBYTE(v151) = (2 * (v151 & 1)) | v151 & 0xFD;
    if ( v11 )
    {
      v12 = n[0];
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_5;
    }
    sub_2240AE0(&v126, n);
    if ( (v151 & 2) != 0 )
      sub_14F2AF0(n, (__int64)n, v28);
    if ( (v151 & 1) != 0 )
    {
      if ( n[0] )
        (*(void (**)(void))(*(_QWORD *)n[0] + 8LL))();
    }
    else if ( (__m128i *)n[0] != &src )
    {
      j_j___libc_free_0(n[0], src.m128i_i64[0] + 1);
    }
  }
  LODWORD(v131) = 0;
  v29 = a2[3].m128i_i64[1];
  v30 = ((unsigned __int64)a2[3].m128i_i64[1] >> 3) & 0xFFFFFFFFFFFFFFF8LL;
  v130.m128i_i64[0] = v30;
  v31 = v29 & 0x3F;
  if ( !v31 )
    goto LABEL_47;
  if ( v30 >= v129.m128i_i64[1] )
    goto LABEL_181;
  v98 = (unsigned __int64 *)(v30 + v129.m128i_i64[0]);
  if ( v129.m128i_i64[1] < v30 + 8 )
  {
    v130.m128i_i64[1] = 0;
    v106 = v129.m128i_i32[2] - v30;
    v107 = (unsigned int)(v129.m128i_i32[2] - v30);
    v100 = 8 * (v129.m128i_i32[2] - v30);
    v108 = v107 + v30;
    if ( v106 )
    {
      v109 = 0;
      v99 = 0;
      do
      {
        v110 = *((unsigned __int8 *)v98 + v109);
        v111 = 8 * v109++;
        v99 |= v110 << v111;
        v130.m128i_i64[1] = v99;
      }
      while ( v109 != v107 );
      v130.m128i_i64[0] = v108;
      LODWORD(v131) = v100;
      if ( v100 >= v31 )
        goto LABEL_166;
    }
    else
    {
      v130.m128i_i64[0] = v108;
    }
    goto LABEL_181;
  }
  v99 = *v98;
  v130.m128i_i64[0] = v30 + 8;
  v100 = 64;
LABEL_166:
  LODWORD(v131) = v100 - v31;
  v130.m128i_i64[1] = v99 >> v31;
LABEL_47:
  v32 = _mm_loadu_si128(&v129);
  v33 = _mm_loadu_si128(&v130);
  v141 = v131;
  v142 = v132;
  v132 = 0;
  v143 = v133;
  v133 = 0;
  v144 = v134;
  v145 = v147;
  v134 = 0;
  v146 = 0x800000000LL;
  v139 = v32;
  v140 = v33;
  if ( (_DWORD)v136 )
    sub_14F2DD0((__int64)&v145, (__int64 *)&v135);
  v148 = v138;
  v34 = a2[2].m128i_i64[0];
  v35 = a2[2].m128i_i64[1];
  v118 = v126;
  v113 = v127;
  v36 = sub_22077B0(1808);
  v37 = v36;
  if ( v36 )
  {
    v38 = v36 + 8;
    v39 = _mm_loadu_si128(&v139);
    v40 = _mm_loadu_si128(&v140);
    v41 = v142;
    v142 = 0;
    v42 = v143;
    v151 = v141;
    v43 = v144;
    v44 = HIDWORD(v141);
    v155 = v157;
    v156 = 0x800000000LL;
    v45 = v146;
    v152 = v41;
    v153 = v143;
    v154 = v144;
    v144 = 0;
    v143 = 0;
    *(__m128i *)n = v39;
    src = v40;
    if ( (_DWORD)v146 )
    {
      sub_14F2DD0((__int64)&v155, (__int64 *)&v145);
      v44 = HIDWORD(v151);
      v41 = v152;
      v42 = v153;
      v43 = v154;
      v45 = v156;
    }
    v46 = v148;
    v47 = _mm_loadu_si128((const __m128i *)n);
    *(_QWORD *)(v37 + 8) = 0;
    v48 = _mm_loadu_si128(&src);
    *(_QWORD *)(v37 + 16) = 0;
    v158 = v46;
    LODWORD(v46) = v151;
    *(_QWORD *)(v37 + 24) = 0;
    *(_QWORD *)(v37 + 80) = v42;
    *(_QWORD *)(v37 + 88) = v43;
    *(_DWORD *)(v37 + 64) = v46;
    *(_DWORD *)(v37 + 68) = v44;
    *(_QWORD *)(v37 + 72) = v41;
    v154 = 0;
    v153 = 0;
    v152 = 0;
    *(_QWORD *)(v37 + 96) = v37 + 112;
    *(_QWORD *)(v37 + 104) = 0x800000000LL;
    *(__m128i *)(v37 + 32) = v47;
    *(__m128i *)(v37 + 48) = v48;
    if ( v45 )
    {
      sub_14F2DD0(v37 + 96, (__int64 *)&v155);
      v50 = (unsigned int)v156;
      *(_QWORD *)(v37 + 368) = v38;
      v112 = (__m128i *)(v37 + 416);
      *(_QWORD *)(v37 + 400) = v37 + 416;
      v51 = (__int64)v155;
      v52 = 32 * v50;
      *(_QWORD *)(v37 + 376) = v34;
      v49 = (_BYTE *)(v51 + v52);
      *(_QWORD *)(v37 + 384) = v35;
      *(_BYTE *)(v37 + 392) = 0;
      *(_QWORD *)(v37 + 408) = 0;
      *(_BYTE *)(v37 + 416) = 0;
      v124 = v51;
      if ( v51 != v51 + v52 )
      {
        v114 = v37;
        v53 = v51 + v52;
        do
        {
          v54 = *(_QWORD *)(v53 - 24);
          v55 = *(_QWORD *)(v53 - 16);
          v53 -= 32;
          v56 = v54;
          if ( v55 != v54 )
          {
            do
            {
              while ( 1 )
              {
                v57 = *(volatile signed __int32 **)(v56 + 8);
                if ( v57 )
                {
                  if ( &_pthread_key_create )
                  {
                    v58 = _InterlockedExchangeAdd(v57 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v58 = *((_DWORD *)v57 + 2);
                    *((_DWORD *)v57 + 2) = v58 - 1;
                  }
                  if ( v58 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v57 + 16LL))(v57);
                    if ( &_pthread_key_create )
                    {
                      v59 = _InterlockedExchangeAdd(v57 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v59 = *((_DWORD *)v57 + 3);
                      *((_DWORD *)v57 + 3) = v59 - 1;
                    }
                    if ( v59 == 1 )
                      break;
                  }
                }
                v56 += 16;
                if ( v55 == v56 )
                  goto LABEL_67;
              }
              v56 += 16;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v57 + 24LL))(v57);
            }
            while ( v55 != v56 );
LABEL_67:
            v54 = *(_QWORD *)(v53 + 8);
          }
          if ( v54 )
            j_j___libc_free_0(v54, *(_QWORD *)(v53 + 24) - v54);
        }
        while ( v124 != v53 );
        v37 = v114;
        v49 = v155;
      }
    }
    else
    {
      *(_QWORD *)(v37 + 368) = v38;
      v49 = v155;
      *(_QWORD *)(v37 + 376) = v34;
      *(_QWORD *)(v37 + 384) = v35;
      *(_BYTE *)(v37 + 392) = 0;
      v112 = (__m128i *)(v37 + 416);
      *(_QWORD *)(v37 + 400) = v37 + 416;
      *(_QWORD *)(v37 + 408) = 0;
      *(_BYTE *)(v37 + 416) = 0;
    }
    if ( v49 != v157 )
      _libc_free((unsigned __int64)v49);
    v60 = v153;
    v61 = v152;
    if ( v153 != v152 )
    {
      do
      {
        while ( 1 )
        {
          v62 = *(volatile signed __int32 **)(v61 + 8);
          if ( v62 )
          {
            if ( &_pthread_key_create )
            {
              v63 = _InterlockedExchangeAdd(v62 + 2, 0xFFFFFFFF);
            }
            else
            {
              v63 = *((_DWORD *)v62 + 2);
              *((_DWORD *)v62 + 2) = v63 - 1;
            }
            if ( v63 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v62 + 16LL))(v62);
              if ( &_pthread_key_create )
              {
                v64 = _InterlockedExchangeAdd(v62 + 3, 0xFFFFFFFF);
              }
              else
              {
                v64 = *((_DWORD *)v62 + 3);
                *((_DWORD *)v62 + 3) = v64 - 1;
              }
              if ( v64 == 1 )
                break;
            }
          }
          v61 += 16;
          if ( v60 == v61 )
            goto LABEL_85;
        }
        v61 += 16;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v62 + 24LL))(v62);
      }
      while ( v60 != v61 );
LABEL_85:
      v61 = v152;
    }
    if ( v61 )
      j_j___libc_free_0(v61, v154 - v61);
    *(_QWORD *)(v37 + 440) = 0;
    *(_QWORD *)v37 = off_4984BB0;
    *(_QWORD *)(v37 + 448) = 0;
    *(_QWORD *)(v37 + 432) = a3;
    *(_QWORD *)(v37 + 600) = a3;
    *(_QWORD *)(v37 + 648) = v37 + 664;
    *(_QWORD *)(v37 + 656) = 0x4000000000LL;
    *(_QWORD *)(v37 + 456) = 0;
    *(_BYTE *)(v37 + 464) = 0;
    *(_QWORD *)(v37 + 472) = 0;
    *(_QWORD *)(v37 + 480) = 0;
    *(_QWORD *)(v37 + 488) = 0;
    *(_QWORD *)(v37 + 496) = 0;
    *(_QWORD *)(v37 + 504) = 0;
    *(_QWORD *)(v37 + 512) = 0;
    *(_QWORD *)(v37 + 520) = 0;
    *(_QWORD *)(v37 + 528) = 0;
    *(_QWORD *)(v37 + 536) = 0;
    *(_QWORD *)(v37 + 544) = 0;
    *(_QWORD *)(v37 + 552) = 0;
    *(_QWORD *)(v37 + 560) = 0;
    *(_QWORD *)(v37 + 568) = 0;
    *(_QWORD *)(v37 + 576) = 0;
    *(_QWORD *)(v37 + 584) = 0;
    *(_QWORD *)(v37 + 592) = 0;
    *(_BYTE *)(v37 + 616) = 0;
    *(_QWORD *)(v37 + 624) = 0;
    *(_QWORD *)(v37 + 632) = 0;
    *(_QWORD *)(v37 + 640) = 0;
    *(_QWORD *)(v37 + 1176) = 0;
    *(_QWORD *)(v37 + 1184) = 0;
    *(_QWORD *)(v37 + 1192) = 0;
    *(_QWORD *)(v37 + 1344) = v37 + 1328;
    *(_QWORD *)(v37 + 1352) = v37 + 1328;
    *(_QWORD *)(v37 + 1200) = 0;
    *(_QWORD *)(v37 + 1208) = 0;
    *(_QWORD *)(v37 + 1216) = 0;
    *(_QWORD *)(v37 + 1224) = 0;
    *(_QWORD *)(v37 + 1232) = 0;
    *(_QWORD *)(v37 + 1240) = 0;
    *(_QWORD *)(v37 + 1248) = 0;
    *(_QWORD *)(v37 + 1256) = 0;
    *(_QWORD *)(v37 + 1264) = 0;
    *(_QWORD *)(v37 + 1272) = 0;
    *(_QWORD *)(v37 + 1280) = 0;
    *(_QWORD *)(v37 + 1288) = 0;
    *(_QWORD *)(v37 + 1296) = 0;
    *(_QWORD *)(v37 + 1304) = 0;
    *(_QWORD *)(v37 + 1312) = 0;
    *(_DWORD *)(v37 + 1328) = 0;
    *(_QWORD *)(v37 + 1336) = 0;
    *(_QWORD *)(v37 + 1360) = 0;
    *(_QWORD *)(v37 + 1368) = 0;
    *(_QWORD *)(v37 + 1376) = 0;
    *(_QWORD *)(v37 + 1384) = 0;
    *(_QWORD *)(v37 + 1392) = 0;
    *(_QWORD *)(v37 + 1400) = 0;
    *(_QWORD *)(v37 + 1408) = 0;
    *(_QWORD *)(v37 + 1416) = 0;
    *(_QWORD *)(v37 + 1424) = 0;
    *(_QWORD *)(v37 + 1432) = 0;
    *(_DWORD *)(v37 + 1440) = 0;
    *(_QWORD *)(v37 + 1448) = 0;
    *(_QWORD *)(v37 + 1456) = 0;
    *(_QWORD *)(v37 + 1464) = 0;
    *(_DWORD *)(v37 + 1472) = 0;
    *(_BYTE *)(v37 + 1480) = 0;
    *(_QWORD *)(v37 + 1488) = 0;
    *(_QWORD *)(v37 + 1496) = 0;
    *(_QWORD *)(v37 + 1504) = 0;
    *(_DWORD *)(v37 + 1512) = 0;
    *(_QWORD *)(v37 + 1520) = 0;
    *(_QWORD *)(v37 + 1528) = 0;
    *(_QWORD *)(v37 + 1536) = 0;
    *(_QWORD *)(v37 + 1544) = 0;
    *(_QWORD *)(v37 + 1552) = 0;
    *(_QWORD *)(v37 + 1560) = 0;
    *(_DWORD *)(v37 + 1568) = 0;
    *(_QWORD *)(v37 + 1576) = 0;
    *(_QWORD *)(v37 + 1592) = 0;
    *(_QWORD *)(v37 + 1600) = 0;
    *(_QWORD *)(v37 + 1608) = 0;
    *(_QWORD *)(v37 + 1616) = 0;
    *(_QWORD *)(v37 + 1624) = 0;
    *(_QWORD *)(v37 + 1632) = 0;
    *(_QWORD *)(v37 + 1640) = 0;
    *(_QWORD *)(v37 + 1648) = 0;
    *(_QWORD *)(v37 + 1584) = 8;
    v65 = sub_22077B0(64);
    v66 = *(_QWORD *)(v37 + 1584);
    *(_QWORD *)(v37 + 1576) = v65;
    v67 = (__int64 *)(v65 + ((4 * v66 - 4) & 0xFFFFFFFFFFFFFFF8LL));
    v68 = sub_22077B0(512);
    *(_QWORD *)(v37 + 1616) = v67;
    *v67 = v68;
    *(_QWORD *)(v37 + 1600) = v68;
    *(_QWORD *)(v37 + 1632) = v68;
    *(_QWORD *)(v37 + 1592) = v68;
    *(_QWORD *)(v37 + 1624) = v68;
    *(_WORD *)(v37 + 1656) = 0;
    *(_QWORD *)(v37 + 1760) = v37 + 1776;
    *(_QWORD *)(v37 + 1608) = v68 + 512;
    *(_QWORD *)(v37 + 1648) = v67;
    *(_QWORD *)(v37 + 1640) = v68 + 512;
    *(_BYTE *)(v37 + 1658) = 0;
    *(_QWORD *)(v37 + 1664) = 0;
    *(_QWORD *)(v37 + 1672) = 0;
    *(_QWORD *)(v37 + 1680) = 0;
    *(_QWORD *)(v37 + 1688) = 0;
    *(_DWORD *)(v37 + 1696) = 0;
    *(_QWORD *)(v37 + 1704) = 0;
    *(_QWORD *)(v37 + 1712) = 0;
    *(_QWORD *)(v37 + 1720) = 0;
    *(_DWORD *)(v37 + 1728) = 0;
    *(_QWORD *)(v37 + 1736) = 0;
    *(_QWORD *)(v37 + 1744) = 0;
    *(_QWORD *)(v37 + 1752) = 0;
    *(_QWORD *)(v37 + 1768) = 0x800000000LL;
    *(_QWORD *)(v37 + 1784) = 0;
    *(_QWORD *)(v37 + 1792) = 0;
    *(_QWORD *)(v37 + 1800) = 0;
    if ( v118 )
    {
      n[0] = (size_t)&src;
      sub_14E9CA0((__int64 *)n, v118, (__int64)&v118[v113]);
      v69 = *(__m128i **)(v37 + 400);
      p_src = v69;
      if ( (__m128i *)n[0] != &src )
      {
        v71 = n[1];
        v72 = src.m128i_i64[0];
        if ( v69 == v112 )
        {
          *(_QWORD *)(v37 + 400) = n[0];
          *(_QWORD *)(v37 + 408) = v71;
          *(_QWORD *)(v37 + 416) = v72;
        }
        else
        {
          v73 = *(_QWORD *)(v37 + 416);
          *(_QWORD *)(v37 + 400) = n[0];
          *(_QWORD *)(v37 + 408) = v71;
          *(_QWORD *)(v37 + 416) = v72;
          if ( p_src )
          {
            n[0] = (size_t)p_src;
            src.m128i_i64[0] = v73;
            goto LABEL_93;
          }
        }
        n[0] = (size_t)&src;
        p_src = &src;
LABEL_93:
        n[1] = 0;
        p_src->m128i_i8[0] = 0;
        if ( (__m128i *)n[0] != &src )
          j_j___libc_free_0(n[0], src.m128i_i64[0] + 1);
        goto LABEL_95;
      }
      v95 = n[1];
      if ( n[1] )
      {
        if ( n[1] == 1 )
          v69->m128i_i8[0] = src.m128i_i8[0];
        else
          memcpy(v69, &src, n[1]);
        v95 = n[1];
        v69 = *(__m128i **)(v37 + 400);
      }
    }
    else
    {
      src.m128i_i8[0] = 0;
      v95 = 0;
      n[0] = (size_t)&src;
      v69 = *(__m128i **)(v37 + 400);
    }
    *(_QWORD *)(v37 + 408) = v95;
    v69->m128i_i8[v95] = 0;
    p_src = (__m128i *)n[0];
    goto LABEL_93;
  }
LABEL_95:
  v74 = 32LL * (unsigned int)v146;
  v125 = (__int64)v145;
  v75 = &v145[v74];
  if ( v145 != &v145[v74] )
  {
    v119 = v37;
    v76 = &v145[v74];
    do
    {
      v77 = *((_QWORD *)v76 - 3);
      v78 = *((_QWORD *)v76 - 2);
      v76 -= 32;
      v79 = v77;
      if ( v78 != v77 )
      {
        do
        {
          while ( 1 )
          {
            v80 = *(volatile signed __int32 **)(v79 + 8);
            if ( v80 )
            {
              if ( &_pthread_key_create )
              {
                v81 = _InterlockedExchangeAdd(v80 + 2, 0xFFFFFFFF);
              }
              else
              {
                v81 = *((_DWORD *)v80 + 2);
                *((_DWORD *)v80 + 2) = v81 - 1;
              }
              if ( v81 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v80 + 16LL))(v80);
                if ( &_pthread_key_create )
                {
                  v82 = _InterlockedExchangeAdd(v80 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v82 = *((_DWORD *)v80 + 3);
                  *((_DWORD *)v80 + 3) = v82 - 1;
                }
                if ( v82 == 1 )
                  break;
              }
            }
            v79 += 16;
            if ( v78 == v79 )
              goto LABEL_108;
          }
          v79 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v80 + 24LL))(v80);
        }
        while ( v78 != v79 );
LABEL_108:
        v77 = *((_QWORD *)v76 + 1);
      }
      if ( v77 )
        j_j___libc_free_0(v77, *((_QWORD *)v76 + 3) - v77);
    }
    while ( (_BYTE *)v125 != v76 );
    v37 = v119;
    v75 = v145;
  }
  if ( v75 != v147 )
    _libc_free((unsigned __int64)v75);
  v83 = v143;
  v84 = v142;
  if ( v143 != v142 )
  {
    do
    {
      while ( 1 )
      {
        v85 = *(volatile signed __int32 **)(v84 + 8);
        if ( v85 )
        {
          if ( &_pthread_key_create )
          {
            v86 = _InterlockedExchangeAdd(v85 + 2, 0xFFFFFFFF);
          }
          else
          {
            v86 = *((_DWORD *)v85 + 2);
            *((_DWORD *)v85 + 2) = v86 - 1;
          }
          if ( v86 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v85 + 16LL))(v85);
            if ( &_pthread_key_create )
            {
              v87 = _InterlockedExchangeAdd(v85 + 3, 0xFFFFFFFF);
            }
            else
            {
              v87 = *((_DWORD *)v85 + 3);
              *((_DWORD *)v85 + 3) = v87 - 1;
            }
            if ( v87 == 1 )
              break;
          }
        }
        v84 += 16;
        if ( v83 == v84 )
          goto LABEL_126;
      }
      v84 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v85 + 24LL))(v85);
    }
    while ( v83 != v84 );
LABEL_126:
    v84 = v142;
  }
  if ( v84 )
    j_j___libc_free_0(v84, v144 - v84);
  v88 = a2[1].m128i_i64[1];
  v89 = a2[1].m128i_i64[0];
  v90 = sub_22077B0(736);
  v91 = v90;
  if ( v90 )
    sub_1631D60(v90, v89, v88, a3);
  sub_1633030(v91, v37);
  *(_QWORD *)(v37 + 440) = v91;
  src.m128i_i64[1] = (__int64)sub_14F00D0;
  n[0] = v37;
  src.m128i_i64[0] = (__int64)sub_14E9C70;
  sub_1516610(&v139, v37 + 32, v91, v37 + 552, a6, n);
  v92 = v37 + 608;
  if ( *(_BYTE *)(v37 + 616) )
  {
    sub_1517310(v92, &v139);
  }
  else
  {
    sub_15160A0(v92, &v139);
    *(_BYTE *)(v37 + 616) = 1;
  }
  sub_1517350(&v139);
  if ( src.m128i_i64[0] )
    ((void (__fastcall *)(size_t *, size_t *, __int64))src.m128i_i64[0])(n, n, 3);
  sub_1505110((__int64 *)n, v37, 0, a5);
  v93 = n[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    if ( a4 )
    {
      sub_16330A0(n, v91);
      v93 = n[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
        goto LABEL_138;
    }
    else
    {
      sub_15046F0((__int64 *)n, v37);
      v93 = n[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
      {
LABEL_138:
        v94 = *(_BYTE *)(a1 + 8);
        *(_QWORD *)a1 = v91;
        *(_BYTE *)(a1 + 8) = v94 & 0xFC | 2;
        goto LABEL_5;
      }
    }
  }
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v93;
  if ( v91 )
  {
    sub_1633490(v91);
    j_j___libc_free_0(v91, 736);
  }
LABEL_5:
  if ( v126 != v128 )
    j_j___libc_free_0(v126, v128[0] + 1LL);
  v13 = 32LL * (unsigned int)v136;
  v123 = (__int64)v135;
  v14 = &v135[v13];
  if ( v135 != &v135[v13] )
  {
    v15 = &v135[v13];
    do
    {
      v16 = *((_QWORD *)v15 - 3);
      v17 = *((_QWORD *)v15 - 2);
      v15 -= 32;
      v18 = v16;
      if ( v17 != v16 )
      {
        do
        {
          while ( 1 )
          {
            v19 = *(volatile signed __int32 **)(v18 + 8);
            if ( v19 )
            {
              if ( &_pthread_key_create )
              {
                v20 = _InterlockedExchangeAdd(v19 + 2, 0xFFFFFFFF);
              }
              else
              {
                v20 = *((_DWORD *)v19 + 2);
                *((_DWORD *)v19 + 2) = v20 - 1;
              }
              if ( v20 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 16LL))(v19);
                if ( &_pthread_key_create )
                {
                  v21 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v21 = *((_DWORD *)v19 + 3);
                  *((_DWORD *)v19 + 3) = v21 - 1;
                }
                if ( v21 == 1 )
                  break;
              }
            }
            v18 += 16;
            if ( v17 == v18 )
              goto LABEL_20;
          }
          v18 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
        }
        while ( v17 != v18 );
LABEL_20:
        v16 = *((_QWORD *)v15 + 1);
      }
      if ( v16 )
        j_j___libc_free_0(v16, *((_QWORD *)v15 + 3) - v16);
    }
    while ( (_BYTE *)v123 != v15 );
    v14 = v135;
  }
  if ( v14 != v137 )
    _libc_free((unsigned __int64)v14);
  v22 = v133;
  v23 = v132;
  if ( v133 != v132 )
  {
    do
    {
      while ( 1 )
      {
        v24 = *(volatile signed __int32 **)(v23 + 8);
        if ( v24 )
        {
          if ( &_pthread_key_create )
          {
            v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
          }
          else
          {
            v25 = *((_DWORD *)v24 + 2);
            *((_DWORD *)v24 + 2) = v25 - 1;
          }
          if ( v25 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 16LL))(v24);
            if ( &_pthread_key_create )
            {
              v26 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF);
            }
            else
            {
              v26 = *((_DWORD *)v24 + 3);
              *((_DWORD *)v24 + 3) = v26 - 1;
            }
            if ( v26 == 1 )
              break;
          }
        }
        v23 += 16;
        if ( v22 == v23 )
          goto LABEL_38;
      }
      v23 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 24LL))(v24);
    }
    while ( v22 != v23 );
LABEL_38:
    v23 = v132;
  }
  if ( v23 )
    j_j___libc_free_0(v23, v134 - v23);
  return a1;
}
