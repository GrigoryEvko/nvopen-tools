// Function: sub_37AC930
// Address: 0x37ac930
//
__int64 __fastcall sub_37AC930(__int64 *a1, __int64 a2, __int64 a3)
{
  const __m128i *v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int16 v8; // cx
  __int128 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  unsigned __int16 v12; // ax
  unsigned __int64 v13; // r14
  __int64 v14; // rdx
  char v15; // r15
  __int64 v16; // rax
  unsigned __int16 v17; // dx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rdx
  const __m128i *v25; // roff
  char v27; // r13
  unsigned __int8 v28; // r11
  __int64 v29; // rax
  __m128i v30; // xmm0
  __m128i *v31; // rdx
  unsigned __int64 v32; // r8
  const __m128i *v33; // rbx
  __m128i *v34; // rax
  __int64 v35; // rbx
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  char v38; // cl
  const __m128i *v39; // rdi
  __int64 v40; // r8
  __int64 v41; // r9
  int v42; // edx
  __int64 v43; // r13
  __int64 v44; // r15
  __m128i v45; // rax
  int v46; // r14d
  unsigned __int8 v47; // r12
  __int128 v48; // rax
  __int64 v49; // r9
  unsigned __int8 *v50; // rax
  unsigned __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // r9
  __m128i *v54; // r10
  __int64 v55; // rax
  __int64 v56; // r8
  __m128i **v57; // rax
  __int64 v58; // r12
  __int64 v59; // rdi
  char v60; // al
  char v61; // r9
  unsigned __int64 v62; // rdx
  char *v63; // rbx
  unsigned __int32 v65; // r12d
  __int64 *v66; // r14
  __int64 v67; // rax
  int v68; // r9d
  __int64 v69; // r8
  __int64 v70; // rbx
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // r15
  __int64 v74; // r14
  __int128 v75; // rax
  __int64 v76; // r9
  unsigned __int8 *v77; // rax
  unsigned __int64 v78; // rdx
  __m128i *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r8
  __int64 v82; // rdx
  __m128i **v83; // rdx
  int v84; // edx
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // [rsp-10h] [rbp-2B0h]
  unsigned __int64 v88; // [rsp+10h] [rbp-290h]
  __int64 v89; // [rsp+18h] [rbp-288h]
  __int64 v90; // [rsp+20h] [rbp-280h]
  __int64 v91; // [rsp+20h] [rbp-280h]
  __int64 v92; // [rsp+28h] [rbp-278h]
  unsigned __int64 v93; // [rsp+30h] [rbp-270h]
  unsigned __int8 v94; // [rsp+3Dh] [rbp-263h]
  __int16 v95; // [rsp+3Eh] [rbp-262h]
  const __m128i *v96; // [rsp+40h] [rbp-260h]
  unsigned __int64 v97; // [rsp+48h] [rbp-258h]
  unsigned __int64 v98; // [rsp+48h] [rbp-258h]
  const __m128i *v99; // [rsp+50h] [rbp-250h]
  unsigned __int64 v100; // [rsp+58h] [rbp-248h]
  __int128 v101; // [rsp+70h] [rbp-230h]
  __int128 v102; // [rsp+80h] [rbp-220h]
  __int32 v103; // [rsp+90h] [rbp-210h]
  __int32 v104; // [rsp+98h] [rbp-208h]
  int v105; // [rsp+9Ch] [rbp-204h]
  __int64 v107; // [rsp+A8h] [rbp-1F8h]
  _QWORD *v108; // [rsp+A8h] [rbp-1F8h]
  __m128i *v109; // [rsp+A8h] [rbp-1F8h]
  _QWORD *v110; // [rsp+A8h] [rbp-1F8h]
  __m128i *v111; // [rsp+A8h] [rbp-1F8h]
  unsigned __int64 v112; // [rsp+B0h] [rbp-1F0h]
  unsigned __int8 v113; // [rsp+C0h] [rbp-1E0h]
  unsigned __int8 v114; // [rsp+C0h] [rbp-1E0h]
  __int64 v115; // [rsp+C0h] [rbp-1E0h]
  __int64 v116; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v117; // [rsp+D8h] [rbp-1C8h]
  __m128i v118; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v119; // [rsp+F0h] [rbp-1B0h] BYREF
  int v120; // [rsp+F8h] [rbp-1A8h]
  unsigned __int16 v121; // [rsp+100h] [rbp-1A0h] BYREF
  __int64 v122; // [rsp+108h] [rbp-198h]
  unsigned int v123; // [rsp+110h] [rbp-190h] BYREF
  __int64 v124; // [rsp+118h] [rbp-188h]
  unsigned __int16 v125; // [rsp+120h] [rbp-180h] BYREF
  __int64 v126; // [rsp+128h] [rbp-178h]
  __m128i v127; // [rsp+130h] [rbp-170h] BYREF
  __int64 v128; // [rsp+140h] [rbp-160h]
  __int64 v129; // [rsp+148h] [rbp-158h]
  __int64 v130; // [rsp+150h] [rbp-150h]
  __int64 v131; // [rsp+158h] [rbp-148h]
  __int64 v132; // [rsp+160h] [rbp-140h]
  __int64 v133; // [rsp+168h] [rbp-138h]
  unsigned __int64 v134; // [rsp+170h] [rbp-130h]
  __int64 v135; // [rsp+178h] [rbp-128h]
  __int128 v136; // [rsp+180h] [rbp-120h] BYREF
  __int64 v137; // [rsp+190h] [rbp-110h]
  __m128i v138; // [rsp+1A0h] [rbp-100h] BYREF
  unsigned __int8 v139; // [rsp+1B0h] [rbp-F0h]
  __m128i v140; // [rsp+1C0h] [rbp-E0h] BYREF
  int v141; // [rsp+1D0h] [rbp-D0h]
  _OWORD v142[2]; // [rsp+1E0h] [rbp-C0h] BYREF
  __m128i *v143; // [rsp+200h] [rbp-A0h] BYREF
  __int64 v144; // [rsp+208h] [rbp-98h]
  _BYTE v145[144]; // [rsp+210h] [rbp-90h] BYREF

  v5 = *(const __m128i **)(a3 + 40);
  v6 = *(_QWORD *)(a3 + 112);
  v7 = v5->m128i_u64[1];
  v93 = v5->m128i_i64[0];
  v142[0] = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v8 = *(_WORD *)(v6 + 32);
  v118 = _mm_loadu_si128(v5 + 5);
  v142[1] = _mm_loadu_si128((const __m128i *)(v6 + 56));
  v112 = v7;
  v95 = v8;
  *(_QWORD *)&v9 = sub_379AB60((__int64)a1, v5[2].m128i_u64[1], v5[3].m128i_i64[0]);
  v10 = *(_QWORD *)(a3 + 80);
  v102 = v9;
  v119 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v119, v10, 1);
  v11 = *(_QWORD *)(a3 + 104);
  v120 = *(_DWORD *)(a3 + 72);
  v12 = *(_WORD *)(a3 + 96);
  v122 = v11;
  v121 = v12;
  if ( v12 )
  {
    if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      goto LABEL_79;
    v13 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
    v15 = byte_444C4A0[16 * v12 - 8];
  }
  else
  {
    v132 = sub_3007260((__int64)&v121);
    v13 = v132;
    v133 = v14;
    v15 = v14;
  }
  v16 = *(_QWORD *)(v102 + 48) + 16LL * DWORD2(v102);
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOWORD(v123) = v17;
  v124 = v18;
  if ( v17 )
  {
    if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      goto LABEL_79;
    v86 = v17 - 1;
    v88 = *(_QWORD *)&byte_444C4A0[16 * v86];
    v22 = word_4456580[v86];
    v23 = 0;
  }
  else
  {
    v130 = sub_3007260((__int64)&v123);
    v131 = v19;
    v88 = v130;
    v22 = sub_3009970((__int64)&v123, *((__int64 *)&v102 + 1), v19, v20, v21);
  }
  v125 = v22;
  v126 = v23;
  if ( v22 )
  {
    if ( v22 != 1 && (unsigned __int16)(v22 - 504) > 7u )
    {
      v89 = *(_QWORD *)&byte_444C4A0[16 * v22 - 16];
      goto LABEL_9;
    }
LABEL_79:
    BUG();
  }
  v128 = sub_3007260((__int64)&v125);
  v129 = v24;
  LODWORD(v89) = v128;
LABEL_9:
  v25 = *(const __m128i **)(a3 + 112);
  v116 = 0;
  v136 = (__int128)_mm_loadu_si128(v25);
  v137 = v25[1].m128i_i64[0];
  v143 = (__m128i *)v145;
  v144 = 0x400000000LL;
  if ( !v13 )
  {
    v94 = 1;
    goto LABEL_50;
  }
  v107 = a2;
  v27 = v15;
  while ( 1 )
  {
    sub_3776670(&v138, a1[1], *a1, (unsigned int)v13, v123, v124, 0, 0);
    v28 = v139;
    if ( !v139 )
    {
      v94 = 0;
      v96 = v143;
      goto LABEL_48;
    }
    v29 = (unsigned int)v144;
    v141 = 0;
    v30 = _mm_load_si128(&v138);
    v31 = v143;
    v32 = (unsigned int)v144 + 1LL;
    v33 = &v140;
    v140 = v30;
    if ( v32 > HIDWORD(v144) )
    {
      if ( v143 > &v140 )
      {
        v114 = v139;
      }
      else
      {
        v114 = v139;
        if ( &v140 < (__m128i *)((char *)v143 + 24 * (unsigned int)v144) )
        {
          v63 = (char *)((char *)&v140 - (char *)v143);
          sub_C8D5F0((__int64)&v143, v145, (unsigned int)v144 + 1LL, 0x18u, v32, v87);
          v31 = v143;
          v29 = (unsigned int)v144;
          v28 = v114;
          v33 = (const __m128i *)&v63[(_QWORD)v143];
          goto LABEL_15;
        }
      }
      sub_C8D5F0((__int64)&v143, v145, (unsigned int)v144 + 1LL, 0x18u, v32, v87);
      v31 = v143;
      v29 = (unsigned int)v144;
      v28 = v114;
    }
LABEL_15:
    v34 = (__m128i *)((char *)v31 + 24 * v29);
    *v34 = _mm_loadu_si128(v33);
    v34[1].m128i_i64[0] = v33[1].m128i_i64[0];
    v35 = (unsigned int)(v144 + 1);
    LODWORD(v144) = v144 + 1;
    if ( v138.m128i_i16[0] )
    {
      if ( v138.m128i_i16[0] == 1 || (unsigned __int16)(v138.m128i_i16[0] - 504) <= 7u )
        goto LABEL_79;
      v36 = *(_QWORD *)&byte_444C4A0[16 * v138.m128i_u16[0] - 16];
      v38 = byte_444C4A0[16 * v138.m128i_u16[0] - 8];
    }
    else
    {
      v113 = v28;
      v36 = sub_3007260((__int64)&v138);
      v28 = v113;
      v134 = v36;
      v135 = v37;
      v38 = v37;
    }
    v39 = v143;
    v40 = 24 * v35;
    v41 = (__int64)&v143[-1] + 24 * v35 - 8;
    if ( v36 )
      v27 = v38;
    v42 = *(_DWORD *)(v41 + 16) + 1;
    v13 -= v36;
    if ( !v13 )
      break;
    while ( (v27 || !v38) && v13 >= v36 )
    {
      ++v42;
      if ( v36 )
        v27 = v38;
      v13 -= v36;
      if ( !v13 )
        goto LABEL_25;
    }
    *(_DWORD *)(v41 + 16) = v42;
  }
LABEL_25:
  v43 = (__int64)a1;
  v94 = v28;
  *(_DWORD *)(v41 + 16) = v42;
  v96 = (const __m128i *)((char *)v39 + v40);
  if ( v39 != (const __m128i *)&v39->m128i_i8[v40] )
  {
    v99 = v39;
    v44 = v107;
    LODWORD(v97) = 0;
    while ( 1 )
    {
      v127 = _mm_loadu_si128(v99);
      v104 = v99[1].m128i_i32[0];
      if ( v127.m128i_i16[0] )
      {
        if ( v127.m128i_i16[0] == 1 || (unsigned __int16)(v127.m128i_i16[0] - 504) <= 7u )
          goto LABEL_79;
        v84 = v127.m128i_u16[0] - 1;
        if ( (unsigned __int16)(v127.m128i_i16[0] - 17) > 0xD3u )
        {
          v100 = *(_QWORD *)&byte_444C4A0[16 * v84];
          goto LABEL_57;
        }
        v105 = word_4456340[v84];
      }
      else
      {
        v45.m128i_i64[0] = sub_3007260((__int64)&v127);
        v140 = v45;
        if ( !sub_30070B0((__int64)&v127) )
        {
          v100 = v140.m128i_i64[0];
LABEL_57:
          v65 = v127.m128i_i32[0];
          v66 = *(__int64 **)(*(_QWORD *)(v43 + 8) + 64LL);
          v115 = v127.m128i_i64[1];
          LOWORD(v67) = sub_2D43050(v127.m128i_i16[0], v88 / v100);
          v69 = 0;
          if ( !(_WORD)v67 )
          {
            v67 = sub_3009400(v66, v65, v115, (unsigned int)(v88 / v100), 0);
            v92 = v67;
            v69 = v85;
          }
          v70 = v92;
          LOWORD(v70) = v67;
          v92 = v70;
          *(_QWORD *)&v101 = sub_33FAF80(*(_QWORD *)(v43 + 8), 234, (__int64)&v119, (unsigned int)v70, v69, v68, v30);
          *((_QWORD *)&v101 + 1) = v71;
          v98 = (unsigned int)(v89 * v97) / v100;
          v72 = v44;
          v73 = (int)v98;
          v74 = v72;
          do
          {
            v110 = *(_QWORD **)(v43 + 8);
            *(_QWORD *)&v75 = sub_3400EE0((__int64)v110, v73, (__int64)&v119, 0, v30);
            v77 = sub_3406EB0(v110, 0x9Eu, (__int64)&v119, v127.m128i_u32[0], v127.m128i_i64[1], v76, v101, v75);
            v79 = sub_33F4560(
                    *(_QWORD **)(v43 + 8),
                    v93,
                    v112,
                    (__int64)&v119,
                    (unsigned __int64)v77,
                    v78,
                    v118.m128i_u64[0],
                    v118.m128i_u64[1],
                    v136,
                    v137,
                    *(_BYTE *)(*(_QWORD *)(a3 + 112) + 34LL),
                    v95,
                    (__int64)v142);
            v81 = v80;
            v82 = *(unsigned int *)(v74 + 8);
            if ( v82 + 1 > (unsigned __int64)*(unsigned int *)(v74 + 12) )
            {
              v91 = v81;
              v111 = v79;
              sub_C8D5F0(v74, (const void *)(v74 + 16), v82 + 1, 0x10u, v81, v82 + 1);
              v82 = *(unsigned int *)(v74 + 8);
              v81 = v91;
              v79 = v111;
            }
            v83 = (__m128i **)(*(_QWORD *)v74 + 16 * v82);
            *v83 = v79;
            ++v73;
            v83[1] = (__m128i *)v81;
            ++*(_DWORD *)(v74 + 8);
            sub_3777490(
              v43,
              (__int64)v79,
              v127.m128i_u32[0],
              v127.m128i_i64[1],
              (__int64)&v136,
              (unsigned int *)&v118,
              v30,
              0);
          }
          while ( v73 != (int)v98 + (unsigned __int64)(unsigned int)(v104 - 1) + 1 );
          v44 = v74;
          v97 = v100 * ((int)v98 + v104) / (unsigned int)v89;
          goto LABEL_54;
        }
        v117 = sub_3007240((__int64)&v127);
        v105 = v117;
      }
      v46 = v97;
      v103 = v104;
      do
      {
        v58 = v116;
        v59 = *(_QWORD *)(a3 + 112);
        if ( v116 )
        {
          v60 = sub_2EAC4F0(v59);
          v61 = -1;
          v62 = -(v58 | (1LL << v60)) & (v58 | (1LL << v60));
          if ( v62 )
          {
            _BitScanReverse64(&v62, v62);
            v61 = 63 - (v62 ^ 0x3F);
          }
          v47 = v61;
        }
        else
        {
          v47 = *(_BYTE *)(v59 + 34);
        }
        v108 = *(_QWORD **)(v43 + 8);
        *(_QWORD *)&v48 = sub_3400EE0((__int64)v108, v46, (__int64)&v119, 0, v30);
        v50 = sub_3406EB0(v108, 0xA1u, (__int64)&v119, v127.m128i_u32[0], v127.m128i_i64[1], v49, v102, v48);
        v54 = sub_33F4560(
                *(_QWORD **)(v43 + 8),
                v93,
                v112,
                (__int64)&v119,
                (unsigned __int64)v50,
                v51,
                v118.m128i_u64[0],
                v118.m128i_u64[1],
                v136,
                v137,
                v47,
                v95,
                (__int64)v142);
        v55 = *(unsigned int *)(v44 + 8);
        v56 = v52;
        if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(v44 + 12) )
        {
          v90 = v52;
          v109 = v54;
          sub_C8D5F0(v44, (const void *)(v44 + 16), v55 + 1, 0x10u, v52, v53);
          v55 = *(unsigned int *)(v44 + 8);
          v56 = v90;
          v54 = v109;
        }
        v57 = (__m128i **)(*(_QWORD *)v44 + 16 * v55);
        v57[1] = (__m128i *)v56;
        *v57 = v54;
        ++*(_DWORD *)(v44 + 8);
        v46 += v105;
        sub_3777490(
          v43,
          (__int64)v54,
          v127.m128i_u32[0],
          v127.m128i_i64[1],
          (__int64)&v136,
          (unsigned int *)&v118,
          v30,
          &v116);
        --v103;
      }
      while ( v103 );
      LODWORD(v97) = v105 * (v104 - 1) + v105 + v97;
LABEL_54:
      v99 = (const __m128i *)((char *)v99 + 24);
      if ( v96 == v99 )
      {
        v96 = v143;
        break;
      }
    }
  }
LABEL_48:
  if ( v96 != (const __m128i *)v145 )
    _libc_free((unsigned __int64)v96);
LABEL_50:
  if ( v119 )
    sub_B91220((__int64)&v119, v119);
  return v94;
}
