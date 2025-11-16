// Function: sub_328E980
// Address: 0x328e980
//
__int64 __fastcall sub_328E980(__int64 *a1, __int64 a2)
{
  __int64 v2; // roff
  __int64 v3; // r13
  int v4; // r8d
  __int64 v5; // rax
  unsigned __int16 v6; // cx
  __int64 v7; // rax
  __int64 v8; // r13
  int v12; // r9d
  __int64 *v13; // roff
  __int64 v14; // rdx
  __int64 v15; // rax
  const __m128i *v16; // roff
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned __int16 v20; // r10
  __int64 v21; // r8
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdi
  bool (__fastcall *v29)(__int64, __int64, unsigned __int16); // rax
  __int128 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // edx
  __int64 v36; // rax
  int v37; // esi
  bool v38; // al
  __int64 v39; // rax
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  const void *v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rbx
  const void *v49; // rsi
  int v50; // r9d
  __int64 v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // r11
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rbx
  __int64 v57; // rbx
  int v58; // r11d
  __int128 v59; // rax
  __int64 v60; // r14
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // r12
  bool v66; // al
  __int64 (*v67)(); // rax
  __int128 v68; // rax
  int v69; // r9d
  __int64 v70; // r12
  __int64 v71; // rbx
  char v72; // al
  int v73; // r9d
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r13
  __int64 v77; // r12
  int v78; // r9d
  __int128 v79; // rax
  __int64 v80; // rax
  int v81; // ecx
  int v82; // r9d
  __int128 v83; // rax
  __int64 v84; // rax
  int v85; // ecx
  __int64 v86; // rax
  bool v87; // al
  int v88; // r9d
  __int64 v89; // rbx
  bool v90; // al
  __int128 v91; // rax
  char v92; // al
  __int128 v93; // rax
  int v94; // r9d
  char v95; // al
  __int64 (*v96)(); // rax
  char v97; // al
  int v98; // edx
  unsigned int v99; // edx
  __int128 v100; // rax
  __int64 v101; // r14
  __int64 v102; // rax
  __int64 v103; // rdx
  __int128 v104; // [rsp-30h] [rbp-140h]
  __int64 v105; // [rsp+8h] [rbp-108h]
  unsigned __int16 v106; // [rsp+8h] [rbp-108h]
  unsigned __int16 v107; // [rsp+10h] [rbp-100h]
  unsigned __int16 v108; // [rsp+10h] [rbp-100h]
  __int128 v109; // [rsp+10h] [rbp-100h]
  __int64 v110; // [rsp+20h] [rbp-F0h]
  unsigned __int16 v111; // [rsp+20h] [rbp-F0h]
  __int64 v112; // [rsp+20h] [rbp-F0h]
  __int128 v113; // [rsp+20h] [rbp-F0h]
  __int128 v114; // [rsp+30h] [rbp-E0h]
  __int128 v115; // [rsp+40h] [rbp-D0h]
  int v116; // [rsp+48h] [rbp-C8h]
  __int64 v117; // [rsp+48h] [rbp-C8h]
  int v118; // [rsp+50h] [rbp-C0h]
  __int64 v119; // [rsp+50h] [rbp-C0h]
  int v120; // [rsp+50h] [rbp-C0h]
  unsigned __int16 v121; // [rsp+58h] [rbp-B8h]
  int v122; // [rsp+58h] [rbp-B8h]
  unsigned int v123; // [rsp+5Ch] [rbp-B4h]
  unsigned __int32 v124; // [rsp+60h] [rbp-B0h]
  int v125; // [rsp+60h] [rbp-B0h]
  int v126; // [rsp+60h] [rbp-B0h]
  __int128 v127; // [rsp+60h] [rbp-B0h]
  unsigned __int16 v128; // [rsp+60h] [rbp-B0h]
  int v129; // [rsp+60h] [rbp-B0h]
  __int128 v130; // [rsp+60h] [rbp-B0h]
  int v131; // [rsp+60h] [rbp-B0h]
  int v132; // [rsp+60h] [rbp-B0h]
  int v133; // [rsp+60h] [rbp-B0h]
  unsigned __int16 v134; // [rsp+60h] [rbp-B0h]
  __int128 v135; // [rsp+60h] [rbp-B0h]
  __int64 v136; // [rsp+68h] [rbp-A8h]
  __m128i v137; // [rsp+90h] [rbp-80h] BYREF
  __m128i v138; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v139; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v140; // [rsp+B8h] [rbp-58h]
  unsigned int v141; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v142; // [rsp+C8h] [rbp-48h]
  __int64 v143; // [rsp+D0h] [rbp-40h] BYREF
  int v144; // [rsp+D8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_QWORD *)v2;
  v137 = _mm_loadu_si128((const __m128i *)v2);
  v4 = *(_DWORD *)(v3 + 64);
  v138 = _mm_loadu_si128((const __m128i *)(v2 + 40));
  v5 = *(_QWORD *)(v3 + 48) + 16LL * v137.m128i_u32[2];
  v6 = *(_WORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  LOWORD(v139) = v6;
  v140 = v7;
  if ( !v4 )
    return 0;
  v12 = *(_DWORD *)(v3 + 24);
  v123 = *(_DWORD *)(a2 + 24);
  v13 = *(__int64 **)(v3 + 40);
  v14 = *v13;
  v15 = *((unsigned int *)v13 + 2);
  v16 = *(const __m128i **)(v138.m128i_i64[0] + 40);
  v17 = v16->m128i_i64[0];
  v115 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(v3 + 40));
  v18 = *(_QWORD *)(v14 + 48) + 16 * v15;
  v19 = *(_QWORD *)(a2 + 80);
  v114 = (__int128)_mm_loadu_si128(v16);
  v20 = *(_WORD *)v18;
  v21 = *(_QWORD *)(v18 + 8);
  v124 = v16->m128i_u32[2];
  v143 = v19;
  LOWORD(v141) = v20;
  v142 = v21;
  if ( v19 )
  {
    v121 = v20;
    v105 = v21;
    v118 = v12;
    v107 = v6;
    v110 = v17;
    sub_B96E90((__int64)&v143, v19, 1);
    v20 = v121;
    v21 = v105;
    v12 = v118;
    v6 = v107;
    v17 = v110;
  }
  v144 = *(_DWORD *)(a2 + 72);
  if ( (unsigned int)(v12 - 213) <= 2 || (unsigned int)(v12 - 223) <= 2 )
    goto LABEL_7;
  if ( v12 == 222 )
  {
    v42 = *(_QWORD *)(v3 + 40);
    v43 = *(_QWORD *)(v138.m128i_i64[0] + 40);
    if ( *(_QWORD *)(v43 + 40) != *(_QWORD *)(v42 + 40) || *(_DWORD *)(v43 + 48) != *(_DWORD *)(v42 + 48) )
      goto LABEL_77;
LABEL_7:
    v22 = *(_QWORD *)(v3 + 56);
    if ( !v22 )
      goto LABEL_51;
    v23 = 1;
    do
    {
      while ( v137.m128i_i32[2] != *(_DWORD *)(v22 + 8) )
      {
        v22 = *(_QWORD *)(v22 + 32);
        if ( !v22 )
          goto LABEL_15;
      }
      if ( !v23 )
        goto LABEL_51;
      v24 = *(_QWORD *)(v22 + 32);
      if ( !v24 )
        goto LABEL_16;
      if ( v137.m128i_i32[2] == *(_DWORD *)(v24 + 8) )
        goto LABEL_51;
      v22 = *(_QWORD *)(v24 + 32);
      v23 = 0;
    }
    while ( v22 );
LABEL_15:
    if ( v23 == 1 )
    {
LABEL_51:
      v36 = *(_QWORD *)(v138.m128i_i64[0] + 56);
      if ( !v36 )
        goto LABEL_48;
      v37 = 1;
      do
      {
        if ( v138.m128i_i32[2] == *(_DWORD *)(v36 + 8) )
        {
          if ( !v37 )
            goto LABEL_48;
          v36 = *(_QWORD *)(v36 + 32);
          if ( !v36 )
            goto LABEL_16;
          if ( v138.m128i_i32[2] == *(_DWORD *)(v36 + 8) )
            goto LABEL_48;
          v37 = 0;
        }
        v36 = *(_QWORD *)(v36 + 32);
      }
      while ( v36 );
      if ( v37 == 1 )
        goto LABEL_48;
    }
LABEL_16:
    v25 = *(_QWORD *)(v17 + 48) + 16LL * v124;
    if ( v20 != *(_WORD *)v25 || *(_QWORD *)(v25 + 8) != v21 && !v20 )
      goto LABEL_48;
    if ( v6 )
    {
      if ( (unsigned __int16)(v6 - 17) <= 0xD3u )
      {
LABEL_21:
        v26 = a1[1];
        if ( v20 != 1 )
        {
          if ( !v20 )
            goto LABEL_48;
          v27 = v20;
          if ( !*(_QWORD *)(v26 + 8LL * v20 + 112) )
            goto LABEL_48;
          if ( v123 > 0x1F3 )
            goto LABEL_25;
LABEL_63:
          if ( (*(_BYTE *)(v123 + v26 + 500 * v27 + 6414) & 0xFB) != 0 )
            goto LABEL_48;
          goto LABEL_25;
        }
        v27 = 1;
        if ( v123 <= 0x1F3 )
          goto LABEL_63;
LABEL_25:
        if ( (v12 & 0xFFFFFFF7) == 0xD7 && *((_BYTE *)a1 + 34) )
        {
          v28 = a1[1];
          v29 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v28 + 2192LL);
          if ( v29 == sub_302E170 )
          {
            if ( !v20 || !*(_QWORD *)(v28 + 8LL * v20 + 112) )
              goto LABEL_48;
          }
          else
          {
            v133 = v12;
            v92 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v29)(v28, v123, v141, v142);
            v12 = v133;
            if ( !v92 )
              goto LABEL_48;
          }
        }
        v125 = v12;
        *(_QWORD *)&v30 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v141, v142, v12, v115, v114);
        if ( v125 != 222 )
        {
          v31 = sub_33FAF80(*a1, v125, (unsigned int)&v143, v139, v140, v125, v30);
          goto LABEL_32;
        }
        v41 = sub_3406EB0(*a1, 222, (unsigned int)&v143, v139, v140, 222, v30, *(_OWORD *)(*(_QWORD *)(v3 + 40) + 40LL));
LABEL_74:
        v8 = v41;
        goto LABEL_49;
      }
    }
    else
    {
      v111 = v20;
      v126 = v12;
      v38 = sub_30070B0((__int64)&v139);
      v12 = v126;
      v20 = v111;
      if ( v38 )
        goto LABEL_21;
    }
    if ( !*((_BYTE *)a1 + 33) )
      goto LABEL_25;
    goto LABEL_21;
  }
  if ( v12 == 216 )
  {
    v39 = *(_QWORD *)(v3 + 56);
    if ( !v39 )
      goto LABEL_113;
    v40 = 1;
    do
    {
      if ( v137.m128i_i32[2] == *(_DWORD *)(v39 + 8) )
      {
        if ( !v40 )
          goto LABEL_113;
        v39 = *(_QWORD *)(v39 + 32);
        if ( !v39 )
          goto LABEL_99;
        if ( v137.m128i_i32[2] == *(_DWORD *)(v39 + 8) )
          goto LABEL_113;
        v40 = 0;
      }
      v39 = *(_QWORD *)(v39 + 32);
    }
    while ( v39 );
    if ( v40 == 1 )
    {
LABEL_113:
      v80 = *(_QWORD *)(v138.m128i_i64[0] + 56);
      if ( !v80 )
        goto LABEL_48;
      v81 = 1;
      do
      {
        if ( v138.m128i_i32[2] == *(_DWORD *)(v80 + 8) )
        {
          if ( !v81 )
            goto LABEL_48;
          v80 = *(_QWORD *)(v80 + 32);
          if ( !v80 )
            goto LABEL_99;
          if ( v138.m128i_i32[2] == *(_DWORD *)(v80 + 8) )
            goto LABEL_48;
          v81 = 0;
        }
        v80 = *(_QWORD *)(v80 + 32);
      }
      while ( v80 );
      if ( v81 == 1 )
        goto LABEL_48;
    }
LABEL_99:
    v64 = *(_QWORD *)(v17 + 48) + 16LL * v124;
    if ( v20 == *(_WORD *)v64 && (*(_QWORD *)(v64 + 8) == v21 || v20) )
    {
      v65 = a1[1];
      if ( !*((_BYTE *)a1 + 33) || (v128 = v20, v66 = sub_328D6E0(a1[1], v123, v141), v20 = v128, v66) )
      {
        v67 = *(__int64 (**)())(*(_QWORD *)v65 + 1432LL);
        if ( v67 != sub_2FE34A0 )
        {
          v134 = v20;
          v95 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v67)(v65, v139, v140, v141, v142);
          v65 = a1[1];
          v20 = v134;
          if ( v95 )
          {
            v96 = *(__int64 (**)())(*(_QWORD *)v65 + 1392LL);
            if ( v96 != sub_2FE3480 )
            {
              v97 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v96)(
                      a1[1],
                      v141,
                      v142,
                      v139,
                      v140);
              v20 = v134;
              if ( v97 )
                goto LABEL_48;
              v65 = a1[1];
            }
          }
        }
        if ( !v20 || !*(_QWORD *)(v65 + 8LL * v20 + 112) )
          goto LABEL_48;
        *(_QWORD *)&v68 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v141, v142, v12, v115, v114);
        v31 = sub_33FAF80(*a1, 216, (unsigned int)&v143, v139, v140, v69, v68);
        goto LABEL_32;
      }
    }
LABEL_48:
    v8 = 0;
    goto LABEL_49;
  }
  if ( (unsigned int)(v12 - 190) > 2 )
  {
    if ( v12 != 186 )
      goto LABEL_38;
    v32 = *(_QWORD *)(v138.m128i_i64[0] + 40);
    v33 = *(_QWORD *)(v3 + 40);
    if ( *(_QWORD *)(v33 + 40) != *(_QWORD *)(v32 + 40) )
      goto LABEL_77;
  }
  else
  {
    v32 = *(_QWORD *)(v138.m128i_i64[0] + 40);
    v33 = *(_QWORD *)(v3 + 40);
    if ( *(_QWORD *)(v33 + 40) != *(_QWORD *)(v32 + 40) )
      goto LABEL_38;
  }
  if ( *(_DWORD *)(v33 + 48) == *(_DWORD *)(v32 + 48) )
  {
    v131 = v12;
    if ( !(unsigned __int8)sub_3286E00(&v137) || !(unsigned __int8)sub_3286E00(&v138) )
      goto LABEL_48;
    *(_QWORD *)&v83 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v141, v142, v82, v115, v114);
    v41 = sub_3406EB0(*a1, v131, (unsigned int)&v143, v139, v140, v131, v83, *(_OWORD *)(*(_QWORD *)(v3 + 40) + 40LL));
    goto LABEL_74;
  }
LABEL_38:
  if ( v12 == 197 )
  {
    v34 = *(_QWORD *)(v3 + 56);
    if ( v34 )
    {
      v35 = 1;
      do
      {
        if ( v137.m128i_i32[2] == *(_DWORD *)(v34 + 8) )
        {
          if ( !v35 )
            goto LABEL_48;
          v34 = *(_QWORD *)(v34 + 32);
          if ( !v34 )
            goto LABEL_128;
          if ( v137.m128i_i32[2] == *(_DWORD *)(v34 + 8) )
            goto LABEL_48;
          v35 = 0;
        }
        v34 = *(_QWORD *)(v34 + 32);
      }
      while ( v34 );
      if ( v35 == 1 )
        goto LABEL_48;
LABEL_128:
      v84 = *(_QWORD *)(v138.m128i_i64[0] + 56);
      if ( v84 )
      {
        v85 = 1;
        do
        {
          if ( v138.m128i_i32[2] == *(_DWORD *)(v84 + 8) )
          {
            if ( !v85 )
              goto LABEL_48;
            v84 = *(_QWORD *)(v84 + 32);
            if ( !v84 )
              goto LABEL_153;
            if ( *(_DWORD *)(v84 + 8) == v138.m128i_i32[2] )
              goto LABEL_48;
            v85 = 0;
          }
          v84 = *(_QWORD *)(v84 + 32);
        }
        while ( v84 );
        if ( v85 == 1 )
          goto LABEL_48;
LABEL_153:
        *(_QWORD *)&v93 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v141, v142, 197, v115, v114);
        v31 = sub_33FAF80(*a1, 197, (unsigned int)&v143, v139, v140, v94, v93);
        goto LABEL_32;
      }
    }
    goto LABEL_48;
  }
LABEL_77:
  if ( (unsigned int)(v12 - 195) > 1 )
  {
    if ( v12 != 234 && v12 != 167 || (v106 = v20, v119 = v21, v108 = v6, v112 = v17, *((int *)a1 + 6) > 1) )
    {
      if ( v12 != 165 )
        goto LABEL_48;
      if ( *((int *)a1 + 6) > 2 )
        goto LABEL_48;
      v44 = *(_QWORD *)(v3 + 56);
      if ( !v44 )
        goto LABEL_48;
      if ( *(_QWORD *)(v44 + 32) )
        goto LABEL_48;
      v45 = *(_QWORD *)(v138.m128i_i64[0] + 56);
      if ( !v45 )
        goto LABEL_48;
      if ( *(_QWORD *)(v45 + 32) )
        goto LABEL_48;
      v46 = (const void *)sub_3288400(v3, v19);
      v48 = v47;
      v49 = (const void *)sub_3288400(v138.m128i_i64[0], v19);
      if ( v48 != v51 || 4 * v48 && memcmp(v46, v49, 4 * v48) )
        goto LABEL_48;
      v52 = *(_QWORD *)(v3 + 40);
      v53 = *(_QWORD *)(v52 + 48);
      v54 = *(_QWORD *)(v52 + 40);
      if ( v123 == 188 )
      {
        v56 = *(_QWORD *)(v52 + 40);
        if ( *(_DWORD *)(v54 + 24) != 51 )
        {
          v136 = *(_QWORD *)(v52 + 48);
          v56 = sub_3261DF0((int)&v143, a1[1], v139, v140, *a1, *((_BYTE *)a1 + 33));
          v52 = *(_QWORD *)(v3 + 40);
          v53 = v99 | v136 & 0xFFFFFFFF00000000LL;
        }
        v55 = *(_QWORD *)(v138.m128i_i64[0] + 40);
        if ( *(_QWORD *)(v52 + 40) != *(_QWORD *)(v55 + 40) )
        {
          v57 = *(_QWORD *)v52;
          v58 = *(_DWORD *)(v52 + 8);
LABEL_161:
          if ( *(_DWORD *)(v57 + 24) != 51 )
          {
            v57 = sub_3261DF0((int)&v143, a1[1], v139, v140, *a1, *((_BYTE *)a1 + 33));
            v52 = *(_QWORD *)(v3 + 40);
            v58 = v98;
            v55 = *(_QWORD *)(v138.m128i_i64[0] + 40);
          }
LABEL_93:
          if ( *(_QWORD *)v52 == *(_QWORD *)v55 && *(_DWORD *)(v52 + 8) == *(_DWORD *)(v55 + 8) && v57 )
          {
            v116 = v58;
            *(_QWORD *)&v59 = sub_3406EB0(
                                *a1,
                                v123,
                                (unsigned int)&v143,
                                v139,
                                v140,
                                v50,
                                *(_OWORD *)(v52 + 40),
                                *(_OWORD *)(v55 + 40));
            v60 = *a1;
            v127 = v59;
            v61 = sub_3288400(v3, v123);
            v63 = sub_33FCE10(v60, v139, v140, (unsigned int)&v143, v57, v116, v127, *((__int64 *)&v127 + 1), v61, v62);
LABEL_97:
            v8 = v63;
            goto LABEL_49;
          }
          goto LABEL_48;
        }
      }
      else
      {
        v55 = *(_QWORD *)(v138.m128i_i64[0] + 40);
        v56 = *(_QWORD *)(v55 + 40);
        if ( v54 != v56 )
        {
          v57 = *(_QWORD *)v52;
          v58 = *(_DWORD *)(v52 + 8);
          goto LABEL_93;
        }
      }
      if ( *(_DWORD *)(v52 + 48) == *(_DWORD *)(v55 + 48) && v56 )
      {
        v117 = v53;
        *(_QWORD *)&v100 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v139, v140, v50, *(_OWORD *)v52, *(_OWORD *)v55);
        v101 = *a1;
        v135 = v100;
        v102 = sub_3288400(v3, v123);
        v63 = sub_33FCE10(v101, v139, v140, (unsigned int)&v143, v135, DWORD2(v135), v56, v117, v102, v103);
        goto LABEL_97;
      }
      v57 = *(_QWORD *)v52;
      v58 = *(_DWORD *)(v52 + 8);
      if ( v123 != 188 )
        goto LABEL_93;
      goto LABEL_161;
    }
    v122 = v12;
    if ( !sub_3280180((__int64)&v141) )
      goto LABEL_48;
    v86 = *(_QWORD *)(v112 + 48) + 16LL * v124;
    if ( v106 != *(_WORD *)v86 || *(_QWORD *)(v86 + 8) != v119 && !v106 )
      goto LABEL_48;
    v87 = sub_32801E0((__int64)&v139);
    v88 = v122;
    if ( v87 )
    {
      if ( v108 )
      {
        v89 = a1[1];
        if ( *(_QWORD *)(v89 + 8LL * v108 + 112) )
        {
          v90 = sub_32801E0((__int64)&v141);
          v88 = v122;
          if ( !v90 && (!v106 || !*(_QWORD *)(v89 + 8LL * v106 + 112)) )
            goto LABEL_48;
        }
      }
    }
    v132 = v88;
    *(_QWORD *)&v91 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v141, v142, v88, v115, v114);
    v31 = sub_33FAF80(*a1, v132, (unsigned int)&v143, v139, v140, v132, v91);
LABEL_32:
    v8 = v31;
    goto LABEL_49;
  }
  v70 = *(_QWORD *)(v138.m128i_i64[0] + 40);
  v71 = *(_QWORD *)(v3 + 40);
  if ( *(_QWORD *)(v71 + 80) != *(_QWORD *)(v70 + 80) )
    goto LABEL_48;
  if ( *(_DWORD *)(v71 + 88) != *(_DWORD *)(v70 + 88) )
    goto LABEL_48;
  v129 = v12;
  if ( !(unsigned __int8)sub_3286E00(&v137) )
    goto LABEL_48;
  v72 = sub_3286E00(&v138);
  v73 = v129;
  if ( !v72 )
    goto LABEL_48;
  v120 = v129;
  v130 = (__int128)_mm_loadu_si128((const __m128i *)(v71 + 40));
  v113 = (__int128)_mm_loadu_si128((const __m128i *)(v71 + 80));
  v109 = *(_OWORD *)(v70 + 40);
  v74 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v139, v140, v73, v115, v114);
  v76 = v75;
  v77 = v74;
  *(_QWORD *)&v79 = sub_3406EB0(*a1, v123, (unsigned int)&v143, v139, v140, v78, v130, v109);
  *((_QWORD *)&v104 + 1) = v76;
  *(_QWORD *)&v104 = v77;
  v8 = sub_340F900(*a1, v120, (unsigned int)&v143, v139, v140, v120, v104, v79, v113);
LABEL_49:
  if ( v143 )
    sub_B91220((__int64)&v143, v143);
  return v8;
}
