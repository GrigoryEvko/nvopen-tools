// Function: sub_32C64A0
// Address: 0x32c64a0
//
__int64 __fastcall sub_32C64A0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r10d
  __int64 v7; // r12
  __int64 v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // rcx
  __int64 *v12; // r13
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r9
  const __m128i *v16; // r12
  const __m128i *v17; // r15
  __int64 v18; // r14
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rsi
  unsigned __int64 v23; // rax
  unsigned int v24; // edx
  char *v25; // rax
  __int64 v26; // rax
  __m128i v27; // xmm0
  int v28; // edx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r11
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r11
  unsigned __int64 v35; // r15
  __int64 v36; // rdi
  int v37; // r12d
  _QWORD *v38; // r8
  __int64 v39; // rcx
  __int64 v40; // r14
  __m128i *v41; // rcx
  _QWORD *v42; // rax
  _QWORD *i; // rdx
  __int64 v44; // r15
  __int64 v45; // r12
  unsigned __int64 v46; // r13
  __int64 *v47; // rbx
  __int64 v48; // r14
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  _BYTE **v52; // r11
  __int64 m128i_i64; // r9
  unsigned __int64 v54; // r14
  unsigned __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int64 *v58; // r12
  unsigned int v59; // esi
  unsigned __int64 v60; // rbx
  __int64 v61; // r13
  __int64 *v62; // rdx
  __int64 v63; // rdx
  unsigned __int64 v64; // r8
  __int64 j; // r15
  int v66; // r14d
  _DWORD *v67; // rax
  __int64 v68; // rdx
  unsigned int v69; // r13d
  int v70; // eax
  __int64 *v71; // r12
  __int64 *v72; // rax
  __int64 *v73; // r15
  __int64 *v74; // r12
  __int64 v75; // rdx
  __int64 v76; // rsi
  unsigned __int64 v77; // rdx
  __int64 v78; // r8
  const __m128i *v79; // r12
  const __m128i *v80; // rbx
  __int64 v81; // rsi
  char *v82; // rax
  char *v83; // rdx
  __int64 v84; // r12
  __int64 v85; // rsi
  __int64 v86; // r12
  __int64 v87; // rsi
  __int64 v88; // rax
  __m128i v89; // xmm0
  unsigned __int64 v90; // rdx
  _BYTE **v91; // [rsp+0h] [rbp-470h]
  __int64 v92; // [rsp+0h] [rbp-470h]
  __int64 v93; // [rsp+8h] [rbp-468h]
  _BYTE **v94; // [rsp+8h] [rbp-468h]
  unsigned __int64 v95; // [rsp+10h] [rbp-460h]
  __int64 *v96; // [rsp+10h] [rbp-460h]
  __int64 v97; // [rsp+18h] [rbp-458h]
  __m128i v98; // [rsp+20h] [rbp-450h] BYREF
  _QWORD *v99; // [rsp+30h] [rbp-440h]
  const __m128i *v100; // [rsp+38h] [rbp-438h]
  __m128i v101; // [rsp+40h] [rbp-430h] BYREF
  const __m128i **v102; // [rsp+50h] [rbp-420h]
  __int64 v103; // [rsp+58h] [rbp-418h]
  __int64 *v104; // [rsp+60h] [rbp-410h]
  __int64 *v105; // [rsp+68h] [rbp-408h]
  char v106; // [rsp+7Ah] [rbp-3F6h] BYREF
  char v107; // [rsp+7Bh] [rbp-3F5h] BYREF
  unsigned int v108; // [rsp+7Ch] [rbp-3F4h] BYREF
  __int64 v109; // [rsp+80h] [rbp-3F0h] BYREF
  int v110; // [rsp+88h] [rbp-3E8h]
  _BYTE *v111; // [rsp+90h] [rbp-3E0h] BYREF
  __int64 v112; // [rsp+98h] [rbp-3D8h]
  _BYTE v113[32]; // [rsp+A0h] [rbp-3D0h] BYREF
  __int64 *v114; // [rsp+C0h] [rbp-3B0h] BYREF
  char *v115; // [rsp+C8h] [rbp-3A8h]
  char *v116; // [rsp+D0h] [rbp-3A0h]
  __int64 v117; // [rsp+D8h] [rbp-398h]
  _BYTE **v118; // [rsp+E0h] [rbp-390h]
  __int64 v119; // [rsp+E8h] [rbp-388h]
  unsigned int *v120; // [rsp+F0h] [rbp-380h]
  __int64 *v121; // [rsp+F8h] [rbp-378h]
  _QWORD *v122; // [rsp+100h] [rbp-370h] BYREF
  __int64 v123; // [rsp+108h] [rbp-368h]
  _QWORD v124[8]; // [rsp+110h] [rbp-360h] BYREF
  const __m128i *v125; // [rsp+150h] [rbp-320h] BYREF
  __int64 v126; // [rsp+158h] [rbp-318h]
  _BYTE v127[128]; // [rsp+160h] [rbp-310h] BYREF
  _BYTE *v128; // [rsp+1E0h] [rbp-290h] BYREF
  __int64 v129; // [rsp+1E8h] [rbp-288h]
  _BYTE v130[128]; // [rsp+1F0h] [rbp-280h] BYREF
  _BYTE *v131; // [rsp+270h] [rbp-200h] BYREF
  __int64 v132; // [rsp+278h] [rbp-1F8h]
  _BYTE v133[128]; // [rsp+280h] [rbp-1F0h] BYREF
  __int64 v134; // [rsp+300h] [rbp-170h] BYREF
  char *v135; // [rsp+308h] [rbp-168h]
  __int64 v136; // [rsp+310h] [rbp-160h]
  int v137; // [rsp+318h] [rbp-158h]
  char v138; // [rsp+31Ch] [rbp-154h]
  char v139; // [rsp+320h] [rbp-150h] BYREF
  __int64 v140; // [rsp+3A0h] [rbp-D0h] BYREF
  char *v141; // [rsp+3A8h] [rbp-C8h]
  __int64 v142; // [rsp+3B0h] [rbp-C0h]
  int v143; // [rsp+3B8h] [rbp-B8h]
  char v144; // [rsp+3BCh] [rbp-B4h]
  char v145; // [rsp+3C0h] [rbp-B0h] BYREF

  v6 = *(_DWORD *)(a2 + 64);
  v104 = (__int64 *)a1;
  v103 = a2;
  if ( v6 != 2 )
    goto LABEL_5;
  v29 = sub_325F250(*(unsigned int **)(**(_QWORD **)(a2 + 40) + 40LL), *(_DWORD *)(**(_QWORD **)(a2 + 40) + 64LL));
  v32 = *(_QWORD *)(v31 + 40);
  if ( v32 == v29 && *(_DWORD *)(v31 + 48) == v28 )
    return v30;
  v33 = sub_325F250(*(unsigned int **)(v32 + 40), *(_DWORD *)(v32 + 64));
  if ( a6 == v33 && *(_DWORD *)(v34 + 8) == (_DWORD)a3 )
    return *(_QWORD *)(v34 + 40);
LABEL_5:
  if ( !*((_DWORD *)v104 + 7) || (unsigned int)qword_5038088 < v6 )
    return 0;
  v9 = *(_QWORD *)(v103 + 56);
  if ( v9 )
  {
    if ( !*(_QWORD *)(v9 + 32) )
    {
      v76 = *(_QWORD *)(v9 + 16);
      if ( *(_DWORD *)(v76 + 24) == 2 )
        sub_32B3E80((__int64)v104, v76, 1, 0, a5, a6);
    }
  }
  v10 = 0;
  v134 = 0;
  v100 = (const __m128i *)v127;
  v11 = 0;
  v12 = &v134;
  v125 = (const __m128i *)v127;
  v126 = 0x800000000LL;
  v135 = &v139;
  v99 = v124;
  v124[0] = v103;
  v122 = v124;
  v123 = 0x800000001LL;
  v13 = v124;
  v102 = &v125;
  v136 = 16;
  v137 = 0;
  v138 = 1;
  v106 = 0;
  v105 = &v140;
  while ( 1 )
  {
    v14 = v13[v11];
    v15 = *(_QWORD *)(v14 + 40);
    v16 = (const __m128i *)v15;
    v17 = (const __m128i *)(v15 + 40LL * *(unsigned int *)(v14 + 64));
    v101.m128i_i64[0] = (__int64)&v122;
    if ( (const __m128i *)v15 != v17 )
    {
      while ( 1 )
      {
        v18 = v16->m128i_i64[0];
        v19 = *(_DWORD *)(v16->m128i_i64[0] + 24);
        if ( v19 == 1 )
          goto LABEL_32;
        if ( v19 != 2 )
          goto LABEL_27;
        v20 = *(_QWORD *)(v18 + 56);
        a3 = v16->m128i_u32[2];
        v11 = 1;
        if ( !v20 )
          goto LABEL_27;
        do
        {
          while ( (_DWORD)a3 != *(_DWORD *)(v20 + 8) )
          {
            v20 = *(_QWORD *)(v20 + 32);
            if ( !v20 )
              goto LABEL_19;
          }
          if ( !(_DWORD)v11 )
            goto LABEL_27;
          v21 = *(_QWORD *)(v20 + 32);
          if ( !v21 )
            goto LABEL_20;
          if ( *(_DWORD *)(v21 + 8) == (_DWORD)a3 )
            goto LABEL_27;
          v20 = *(_QWORD *)(v21 + 32);
          v11 = 0;
        }
        while ( v20 );
LABEL_19:
        if ( (_DWORD)v11 == 1 )
          goto LABEL_27;
LABEL_20:
        v140 = v16->m128i_i64[0];
        v22 = &v122[(unsigned int)v123];
        if ( v22 == sub_325EB50(v122, (__int64)v22, v105) )
        {
          a3 = v15 + 1;
          if ( v15 + 1 > (unsigned __int64)HIDWORD(v123) )
          {
            sub_C8D5F0(v101.m128i_i64[0], v99, a3, 8u, a5, v15);
            a3 = (unsigned int)v123;
            v22 = &v122[(unsigned int)v123];
          }
          v16 = (const __m128i *)((char *)v16 + 40);
          *v22 = v18;
          v106 = 1;
          LODWORD(v123) = v123 + 1;
          if ( v17 == v16 )
            break;
        }
        else
        {
LABEL_27:
          if ( v138 )
          {
            v25 = v135;
            v11 = HIDWORD(v136);
            a3 = (unsigned __int64)&v135[8 * HIDWORD(v136)];
            if ( v135 != (char *)a3 )
            {
              while ( v18 != *(_QWORD *)v25 )
              {
                v25 += 8;
                if ( (char *)a3 == v25 )
                  goto LABEL_39;
              }
              goto LABEL_32;
            }
LABEL_39:
            if ( HIDWORD(v136) < (unsigned int)v136 )
            {
              ++HIDWORD(v136);
              *(_QWORD *)a3 = v18;
              ++v134;
LABEL_36:
              v26 = (unsigned int)v126;
              v11 = HIDWORD(v126);
              v27 = _mm_loadu_si128(v16);
              a3 = (unsigned int)v126 + 1LL;
              if ( a3 > HIDWORD(v126) )
              {
                v98 = v27;
                sub_C8D5F0((__int64)v102, v100, a3, 0x10u, a5, v15);
                v26 = (unsigned int)v126;
                v27 = _mm_load_si128(&v98);
              }
              v125[v26] = v27;
              LODWORD(v126) = v126 + 1;
              goto LABEL_33;
            }
          }
          sub_C8CC70((__int64)&v134, v18, a3, v11, a5, v15);
          if ( (_BYTE)a3 )
            goto LABEL_36;
LABEL_32:
          v106 = 1;
LABEL_33:
          v16 = (const __m128i *)((char *)v16 + 40);
          if ( v17 == v16 )
            break;
        }
      }
    }
    LODWORD(v23) = v123;
    v11 = v10 + 1;
    v10 = v11;
    v24 = v123;
    if ( (unsigned int)v11 >= (unsigned int)v123 )
      goto LABEL_61;
    a3 = (unsigned int)v126;
    if ( (unsigned int)qword_5038088 < (unsigned int)v126 )
      break;
    v13 = v122;
  }
  v35 = v11;
  v36 = v11;
  v37 = v11;
  while ( 1 )
  {
    v38 = v122;
    v39 = (unsigned int)a3;
    v40 = v122[v36];
    if ( (unsigned int)a3 >= (unsigned __int64)HIDWORD(v126) )
    {
      v77 = (unsigned int)a3 + 1LL;
      v38 = (_QWORD *)(v97 & 0xFFFFFFFF00000000LL);
      v97 &= 0xFFFFFFFF00000000LL;
      if ( HIDWORD(v126) < (unsigned __int64)(v39 + 1) )
      {
        v101.m128i_i64[0] = (__int64)v38;
        sub_C8D5F0((__int64)v102, v100, v77, 0x10u, (__int64)v38, v15);
        v39 = (unsigned int)v126;
        v38 = (_QWORD *)v101.m128i_i64[0];
      }
      v23 = (unsigned __int64)&v125[v39];
      *(_QWORD *)v23 = v40;
      *(_QWORD *)(v23 + 8) = v38;
      LODWORD(v23) = v123;
      LODWORD(v126) = v126 + 1;
    }
    else
    {
      v41 = (__m128i *)&v125[(unsigned int)a3];
      if ( v41 )
      {
        v41->m128i_i64[0] = v40;
        v41->m128i_i32[2] = 0;
        LODWORD(a3) = v126;
        LODWORD(v23) = v123;
      }
      LODWORD(v126) = a3 + 1;
    }
    v36 = (unsigned int)(v37 + 1);
    v24 = v23;
    v37 = v36;
    if ( (unsigned int)v36 >= (unsigned int)v23 )
      break;
    LODWORD(a3) = v126;
  }
  v23 = (unsigned int)v23;
  if ( v35 != (unsigned int)v23 )
  {
    if ( v35 >= (unsigned int)v23 )
    {
      if ( v35 > HIDWORD(v123) )
      {
        sub_C8D5F0((__int64)&v122, v99, v35, 8u, (__int64)v38, v15);
        v23 = (unsigned int)v123;
      }
      v42 = &v122[v23];
      for ( i = &v122[v35]; i != v42; ++v42 )
      {
        if ( v42 )
          *v42 = 0;
      }
    }
    LODWORD(v123) = v10;
    v24 = v10;
  }
LABEL_61:
  v44 = (__int64)(v104 + 71);
  if ( v24 > 1 )
  {
    v98.m128i_i64[0] = (__int64)&v134;
    v45 = (__int64)v104;
    v46 = 8;
    v101.m128i_i64[0] = 8LL * v24;
    v47 = v105;
    do
    {
      while ( 1 )
      {
        v48 = v122[v46 / 8];
        if ( *(_DWORD *)(v48 + 24) != 328 )
        {
          v140 = v122[v46 / 8];
          sub_32B3B20(v44, v47);
          if ( *(int *)(v48 + 88) < 0 )
            break;
        }
        v46 += 8LL;
        if ( v101.m128i_i64[0] == v46 )
          goto LABEL_69;
      }
      *(_DWORD *)(v48 + 88) = *(_DWORD *)(v45 + 48);
      v51 = *(unsigned int *)(v45 + 48);
      if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(v45 + 52) )
      {
        sub_C8D5F0(v45 + 40, (const void *)(v45 + 56), v51 + 1, 8u, v49, v50);
        v51 = *(unsigned int *)(v45 + 48);
      }
      v46 += 8LL;
      *(_QWORD *)(*(_QWORD *)(v45 + 40) + 8 * v51) = v48;
      ++*(_DWORD *)(v45 + 48);
    }
    while ( v101.m128i_i64[0] != v46 );
LABEL_69:
    v12 = (__int64 *)v98.m128i_i64[0];
  }
  v140 = 0;
  v52 = &v128;
  v128 = v130;
  v129 = 0x800000000LL;
  v112 = 0x800000000LL;
  v141 = &v145;
  v101.m128i_i64[0] = (__int64)&v111;
  m128i_i64 = (__int64)v125[(unsigned int)v126].m128i_i64;
  v98.m128i_i64[0] = (__int64)v113;
  v111 = v113;
  v142 = 16;
  v143 = 0;
  v144 = 1;
  v107 = 0;
  v108 = 0;
  if ( (const __m128i *)m128i_i64 == v125 )
  {
    v114 = v12;
    v115 = &v106;
    v116 = &v107;
    v118 = &v128;
    v117 = (__int64)v102;
    v119 = v101.m128i_i64[0];
    v120 = &v108;
    v121 = v105;
  }
  else
  {
    v54 = v95;
    v55 = 8;
    v56 = 0;
    v57 = 0;
    v96 = v12;
    v58 = (__int64 *)v125;
    while ( 1 )
    {
      v59 = v57 + 1;
      v60 = v54 & 0xFFFFFFFF00000000LL | v57;
      v108 = v59;
      v61 = *v58;
      v54 = v60;
      if ( v56 + 1 > v55 )
      {
        v92 = m128i_i64;
        v94 = v52;
        sub_C8D5F0((__int64)v52, v130, v56 + 1, 0x10u, v56 + 1, m128i_i64);
        v56 = (unsigned int)v129;
        m128i_i64 = v92;
        v52 = v94;
      }
      v62 = (__int64 *)&v128[16 * v56];
      *v62 = v61;
      v62[1] = v60;
      v63 = (unsigned int)v112;
      LODWORD(v129) = v129 + 1;
      v64 = (unsigned int)v112 + 1LL;
      if ( v64 > HIDWORD(v112) )
      {
        v91 = v52;
        v93 = m128i_i64;
        sub_C8D5F0(v101.m128i_i64[0], (const void *)v98.m128i_i64[0], (unsigned int)v112 + 1LL, 4u, v64, m128i_i64);
        v63 = (unsigned int)v112;
        v52 = v91;
        m128i_i64 = v93;
      }
      v58 += 2;
      *(_DWORD *)&v111[4 * v63] = 1;
      LODWORD(v112) = v112 + 1;
      if ( (__int64 *)m128i_i64 == v58 )
        break;
      v57 = v108;
      v56 = (unsigned int)v129;
      v55 = HIDWORD(v129);
    }
    v118 = v52;
    v115 = &v106;
    v116 = &v107;
    v114 = v96;
    v117 = (__int64)v102;
    v119 = v101.m128i_i64[0];
    v120 = &v108;
    v121 = v105;
    if ( (_DWORD)v129 )
    {
      for ( j = 0; (unsigned int)v129 > (unsigned int)j && (_DWORD)j != 1024; ++j )
      {
        v66 = j;
        if ( v108 <= 1 )
          break;
        v67 = &v128[16 * j];
        v68 = *(_QWORD *)v67;
        v69 = v67[2];
        v70 = *(_DWORD *)(*(_QWORD *)v67 + 24LL);
        if ( v70 == 2 )
        {
          v71 = *(__int64 **)(v68 + 40);
          v72 = &v71[5 * *(unsigned int *)(v68 + 64)];
          if ( v71 != v72 )
          {
            v101.m128i_i64[0] = j;
            v73 = v71;
            v74 = v72;
            do
            {
              v75 = *v73;
              v73 += 5;
              sub_32603A0((__int64)&v114, v66, v75, v69, v64, m128i_i64);
            }
            while ( v74 != v73 );
            j = v101.m128i_i64[0];
          }
          goto LABEL_82;
        }
        if ( v70 <= 2 )
        {
          if ( v70 == 1 )
          {
            ++v108;
            goto LABEL_82;
          }
        }
        else if ( v70 > 50 )
        {
          if ( (unsigned int)(v70 - 366) <= 1 )
            goto LABEL_92;
          if ( v70 <= 365 )
          {
            if ( v70 > 337 )
              goto LABEL_92;
            if ( v70 <= 294 )
            {
              if ( v70 > 292 )
              {
LABEL_92:
                sub_32603A0((__int64)&v114, j, **(_QWORD **)(v68 + 40), v69, v64, m128i_i64);
                goto LABEL_82;
              }
            }
            else if ( (unsigned int)(v70 - 298) <= 1 )
            {
              goto LABEL_92;
            }
          }
          else if ( v70 > 470 )
          {
            if ( v70 == 497 )
              goto LABEL_92;
          }
          else if ( v70 > 464 )
          {
            goto LABEL_92;
          }
        }
        else if ( v70 > 48 )
        {
          goto LABEL_92;
        }
        if ( (*(_BYTE *)(v68 + 32) & 2) != 0 )
          goto LABEL_92;
LABEL_82:
        if ( !--*(_DWORD *)&v111[4 * v69] )
          --v108;
      }
    }
  }
  if ( !v106 )
  {
    v7 = 0;
    goto LABEL_113;
  }
  if ( !(_DWORD)v126 )
  {
    v7 = *v104 + 288;
    goto LABEL_113;
  }
  if ( v107 )
  {
    v78 = (__int64)v125;
    v131 = v133;
    v132 = 0x800000000LL;
    v79 = &v125[(unsigned int)v126];
    v80 = v125;
    while ( 1 )
    {
      v81 = v80->m128i_i64[0];
      if ( v144 )
        break;
      if ( !sub_C8CA60((__int64)v105, v81) )
        goto LABEL_155;
LABEL_143:
      if ( v79 == ++v80 )
      {
        v84 = *v104;
        v85 = *(_QWORD *)(v103 + 80);
        v109 = v85;
        if ( v85 )
          sub_B96E90((__int64)&v109, v85, 1);
        v110 = *(_DWORD *)(v103 + 72);
        v7 = sub_3402E70(v84, &v109, &v131);
        if ( v109 )
          sub_B91220((__int64)&v109, v109);
        if ( v131 != v133 )
          _libc_free((unsigned __int64)v131);
        goto LABEL_113;
      }
    }
    v82 = v141;
    v83 = &v141[8 * HIDWORD(v142)];
    if ( v141 != v83 )
    {
      while ( v81 != *(_QWORD *)v82 )
      {
        v82 += 8;
        if ( v83 == v82 )
          goto LABEL_155;
      }
      goto LABEL_143;
    }
LABEL_155:
    v88 = (unsigned int)v132;
    v89 = _mm_loadu_si128(v80);
    v90 = (unsigned int)v132 + 1LL;
    if ( v90 > HIDWORD(v132) )
    {
      v101 = v89;
      sub_C8D5F0((__int64)&v131, v133, v90, 0x10u, v78, m128i_i64);
      v88 = (unsigned int)v132;
      v89 = _mm_load_si128(&v101);
    }
    *(__m128i *)&v131[16 * v88] = v89;
    LODWORD(v132) = v132 + 1;
    goto LABEL_143;
  }
  v86 = *v104;
  v87 = *(_QWORD *)(v103 + 80);
  v131 = (_BYTE *)v87;
  if ( v87 )
    sub_B96E90((__int64)&v131, v87, 1);
  LODWORD(v132) = *(_DWORD *)(v103 + 72);
  v7 = sub_3402E70(v86, &v131, v102);
  if ( v131 )
    sub_B91220((__int64)&v131, (__int64)v131);
LABEL_113:
  if ( !v144 )
    _libc_free((unsigned __int64)v141);
  if ( v111 != (_BYTE *)v98.m128i_i64[0] )
    _libc_free((unsigned __int64)v111);
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  if ( !v138 )
    _libc_free((unsigned __int64)v135);
  if ( v125 != v100 )
    _libc_free((unsigned __int64)v125);
  if ( v122 != v99 )
    _libc_free((unsigned __int64)v122);
  return v7;
}
