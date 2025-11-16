// Function: sub_276F9C0
// Address: 0x276f9c0
//
__int64 *__fastcall sub_276F9C0(__int64 *a1, _QWORD *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 *v16; // r14
  __int64 v17; // r12
  char *v18; // rax
  __int64 v19; // rcx
  __int64 i; // rdx
  __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 *v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // r12
  unsigned int v30; // eax
  _QWORD *v31; // rsi
  _QWORD *v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rcx
  _QWORD *v36; // rax
  _QWORD *v37; // r15
  __int64 v38; // rsi
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rcx
  __m128i *v43; // rdx
  unsigned __int64 v44; // r12
  __int64 *v45; // rbx
  __int64 v46; // rcx
  __int64 v47; // rcx
  __int64 *v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // r8
  unsigned int v54; // eax
  __int64 v55; // r8
  __int64 v56; // r11
  __m128i *v57; // r10
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 v60; // rax
  __int64 *v61; // rax
  __m128i *v62; // r14
  __int64 *v63; // rax
  __int64 v64; // r12
  __int64 v65; // r10
  __int64 v66; // r9
  __int64 v67; // r8
  __int64 v68; // rdi
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // r13
  __int64 v73; // rbx
  __int64 v74; // r11
  __int64 v75; // rax
  unsigned __int64 v76; // r13
  __int64 v77; // r15
  __int64 v78; // rax
  __int64 v79; // rdi
  bool v80; // cf
  unsigned __int64 v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rbx
  __int64 v84; // r15
  __int64 v85; // rax
  __int64 v86; // rdi
  __int64 v87; // r9
  __int64 v88; // r8
  __int64 v89; // rsi
  __int64 v90; // rcx
  __int64 v91; // rdx
  __int64 v92; // r11
  __int64 v93; // r10
  __int64 v94; // rax
  __int64 v95; // rsi
  __int64 v96; // r15
  unsigned __int64 v97; // rbx
  __int64 v98; // rax
  unsigned __int64 v99; // r15
  __int64 j; // r13
  unsigned int v101; // edx
  __int64 v102; // rax
  unsigned __int64 v103; // rdi
  unsigned __int64 *v104; // rdi
  unsigned int v105; // edi
  __int64 v106; // [rsp+10h] [rbp-210h]
  __int64 *v107; // [rsp+10h] [rbp-210h]
  __int64 v108; // [rsp+20h] [rbp-200h]
  __int64 v109; // [rsp+20h] [rbp-200h]
  __m128i *v110; // [rsp+28h] [rbp-1F8h]
  __int64 v111; // [rsp+30h] [rbp-1F0h]
  __int64 *v112; // [rsp+38h] [rbp-1E8h]
  int v113; // [rsp+48h] [rbp-1D8h]
  __int64 *v114; // [rsp+48h] [rbp-1D8h]
  __int64 v115; // [rsp+48h] [rbp-1D8h]
  __int64 v116; // [rsp+48h] [rbp-1D8h]
  __int64 v117; // [rsp+50h] [rbp-1D0h]
  __int64 v119; // [rsp+60h] [rbp-1C0h]
  __int64 v120; // [rsp+60h] [rbp-1C0h]
  __int64 *v123; // [rsp+78h] [rbp-1A8h]
  __m128i *v124; // [rsp+80h] [rbp-1A0h] BYREF
  __m128i *v125; // [rsp+88h] [rbp-198h]
  __int64 v126; // [rsp+90h] [rbp-190h]
  unsigned __int64 v127; // [rsp+A0h] [rbp-180h] BYREF
  __int64 v128; // [rsp+A8h] [rbp-178h]
  __int64 v129[4]; // [rsp+C0h] [rbp-160h] BYREF
  __int64 v130[4]; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v131; // [rsp+100h] [rbp-120h] BYREF
  __m128i *v132; // [rsp+108h] [rbp-118h]
  __int64 v133; // [rsp+110h] [rbp-110h]
  __int64 v134; // [rsp+118h] [rbp-108h]
  __int64 v135; // [rsp+120h] [rbp-100h] BYREF
  char *v136; // [rsp+128h] [rbp-F8h]
  __int64 v137; // [rsp+130h] [rbp-F0h]
  int v138; // [rsp+138h] [rbp-E8h]
  char v139; // [rsp+13Ch] [rbp-E4h]
  char v140; // [rsp+140h] [rbp-E0h] BYREF
  __m128i *v141; // [rsp+180h] [rbp-A0h] BYREF
  __m128i *v142; // [rsp+188h] [rbp-98h]
  __int64 v143; // [rsp+190h] [rbp-90h]
  __int64 v144; // [rsp+198h] [rbp-88h]
  __int64 v145; // [rsp+1A0h] [rbp-80h]
  __int64 v146; // [rsp+1A8h] [rbp-78h]
  __int64 *v147; // [rsp+1B0h] [rbp-70h]
  __int64 v148; // [rsp+1B8h] [rbp-68h]
  __int64 v149; // [rsp+1C0h] [rbp-60h]
  __int64 *v150; // [rsp+1C8h] [rbp-58h]
  unsigned __int64 v151; // [rsp+1D0h] [rbp-50h] BYREF
  unsigned int v152; // [rsp+1D8h] [rbp-48h]
  __int64 v153; // [rsp+1E0h] [rbp-40h]
  char v154; // [rsp+1E8h] [rbp-38h]

  v6 = a5;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v7 = *(_QWORD *)(a4 + 40);
  v8 = *(unsigned __int8 *)(a5 + 28);
  v112 = a3;
  if ( !(_BYTE)v8 )
  {
LABEL_71:
    sub_C8CC70(a5, v7, (__int64)a3, v8, a5, a6);
    LOBYTE(v8) = *(_BYTE *)(v6 + 28);
    goto LABEL_6;
  }
  v9 = *(__int64 **)(a5 + 8);
  v10 = *(unsigned int *)(a5 + 20);
  a3 = &v9[v10];
  if ( v9 == a3 )
  {
LABEL_70:
    if ( (unsigned int)v10 < *(_DWORD *)(a5 + 16) )
    {
      *(_DWORD *)(a5 + 20) = v10 + 1;
      *a3 = v7;
      LOBYTE(v8) = *(_BYTE *)(a5 + 28);
      ++*(_QWORD *)a5;
      goto LABEL_6;
    }
    goto LABEL_71;
  }
  while ( v7 != *v9 )
  {
    if ( a3 == ++v9 )
      goto LABEL_70;
  }
LABEL_6:
  v135 = 0;
  v136 = &v140;
  v11 = *(_QWORD *)(a4 - 8);
  v12 = *(_DWORD *)(a4 + 4);
  v137 = 8;
  v138 = 0;
  v119 = v11;
  v13 = *(unsigned int *)(a4 + 72);
  v139 = 1;
  v13 *= 32;
  v14 = v13 + 8LL * (v12 & 0x7FFFFFF);
  v15 = (__int64 *)(v119 + v13);
  v123 = (__int64 *)(v14 + v119);
  if ( (__int64 *)(v14 + v119) == v15 )
    goto LABEL_50;
  v16 = v15;
  v117 = v6;
  v17 = *v15;
  v120 = v7;
  while ( 1 )
  {
    v18 = v136;
    v19 = HIDWORD(v137);
    i = (__int64)&v136[8 * HIDWORD(v137)];
    if ( v136 == (char *)i )
    {
LABEL_59:
      if ( HIDWORD(v137) < (unsigned int)v137 )
      {
        v19 = (unsigned int)++HIDWORD(v137);
        *(_QWORD *)i = v17;
        ++v135;
        goto LABEL_15;
      }
      goto LABEL_14;
    }
    while ( v17 != *(_QWORD *)v18 )
    {
      v18 += 8;
      if ( (char *)i == v18 )
        goto LABEL_59;
    }
LABEL_12:
    if ( v123 == ++v16 )
      break;
    while ( 1 )
    {
      v17 = *v16;
      if ( v139 )
        break;
LABEL_14:
      sub_C8CC70((__int64)&v135, v17, i, v19, a5, a6);
      if ( !(_BYTE)i )
        goto LABEL_12;
LABEL_15:
      v21 = a2[8];
      if ( *(_BYTE *)(v21 + 84) )
      {
        v22 = *(_QWORD **)(v21 + 64);
        i = (__int64)&v22[*(unsigned int *)(v21 + 76)];
        if ( v22 == (_QWORD *)i )
          goto LABEL_12;
        while ( v17 != *v22 )
        {
          if ( (_QWORD *)i == ++v22 )
            goto LABEL_12;
        }
      }
      else if ( !sub_C8CA60(v21 + 56, v17) )
      {
        goto LABEL_12;
      }
      v23 = *(_QWORD *)(a4 - 8);
      v24 = 0x1FFFFFFFE0LL;
      v113 = *(_DWORD *)(a4 + 4);
      i = v113 & 0x7FFFFFF;
      if ( (v113 & 0x7FFFFFF) != 0 )
      {
        v25 = 0;
        v19 = v23 + 32LL * *(unsigned int *)(a4 + 72);
        do
        {
          if ( v17 == *(_QWORD *)(v19 + 8 * v25) )
          {
            v24 = 32 * v25;
            goto LABEL_25;
          }
          ++v25;
        }
        while ( (_DWORD)i != (_DWORD)v25 );
        v26 = *(_QWORD *)(v23 + 0x1FFFFFFFE0LL);
        if ( *(_BYTE *)v26 != 17 )
        {
LABEL_64:
          if ( *(_BYTE *)(v117 + 28) )
          {
            v36 = *(_QWORD **)(v117 + 8);
            for ( i = (__int64)&v36[*(unsigned int *)(v117 + 20)]; (_QWORD *)i != v36; ++v36 )
            {
              if ( v17 == *v36 )
                goto LABEL_12;
            }
          }
          else if ( sub_C8CA60(v117, v17) )
          {
            goto LABEL_12;
          }
          if ( a2[2] == v17 )
            goto LABEL_12;
          if ( *(_BYTE *)v26 != 84 )
            goto LABEL_12;
          v37 = *(_QWORD **)(v26 + 40);
          v19 = *((unsigned int *)v112 + 6);
          v38 = v112[1];
          if ( !(_DWORD)v19 )
            goto LABEL_12;
          v19 = (unsigned int)(v19 - 1);
          v39 = v19 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v40 = 16LL * v39;
          a5 = *(_QWORD *)(v38 + v40);
          if ( (_QWORD *)a5 != v37 )
          {
            for ( i = 1; ; i = v105 )
            {
              if ( a5 == -4096 )
                goto LABEL_12;
              v105 = i + 1;
              v39 = v19 & (i + v39);
              v40 = 16LL * v39;
              a5 = *(_QWORD *)(v38 + v40);
              if ( v37 == (_QWORD *)a5 )
                break;
            }
          }
          if ( (_QWORD *)v17 != v37 )
          {
            if ( (unsigned __int8)sub_B19060(v117, *(_QWORD *)(v26 + 40), v40, v19) )
              goto LABEL_12;
            v124 = 0;
            v125 = 0;
            v126 = 0;
            sub_276A3A0(&v131, (__int64)a2, v37, v17, v117, 1);
            v41 = v131;
            v42 = (__int64)v124;
            v43 = v125;
            v131 = 0;
            v124 = (__m128i *)v41;
            v143 = v126;
            v125 = v132;
            v141 = (__m128i *)v42;
            v126 = v133;
            v142 = v43;
            v132 = 0;
            v133 = 0;
            sub_27677F0((unsigned __int64 *)&v141);
            sub_27677F0((unsigned __int64 *)&v131);
            if ( v125 != v124 )
            {
              sub_276F9C0(&v127, a2, v112, v26, v117);
              v44 = v127;
              v106 = v128;
              if ( v128 != v127 )
              {
                while ( 1 )
                {
                  v114 = (__int64 *)v125;
                  v45 = (__int64 *)v124;
                  if ( v125 != v124 )
                    break;
LABEL_115:
                  v44 += 112LL;
                  if ( v106 == v44 )
                    goto LABEL_116;
                }
                while ( 2 )
                {
                  sub_276A0C0((__int64 *)&v141, (_QWORD *)v44);
                  v152 = *(_DWORD *)(v44 + 88);
                  if ( v152 > 0x40 )
                    sub_C43780((__int64)&v151, (const void **)(v44 + 80));
                  else
                    v151 = *(_QWORD *)(v44 + 80);
                  v153 = *(_QWORD *)(v44 + 96);
                  v154 = *(_BYTE *)(v44 + 104);
                  v49 = v45[3];
                  v56 = v45[6];
                  v57 = (__m128i *)v45[7];
                  v58 = v45[8];
                  v59 = v45[9];
                  v50 = v45[4];
                  v48 = (__int64 *)v45[5];
                  v60 = ((v45[2] - v49) >> 3) + 1;
                  if ( v60 >= 0 )
                  {
                    v46 = v45[2] + 8;
                    if ( v60 > 63 )
                    {
                      v47 = v60 >> 6;
                      goto LABEL_85;
                    }
                  }
                  else
                  {
                    v47 = ~((unsigned __int64)~v60 >> 6);
LABEL_85:
                    v48 += v47;
                    v49 = *v48;
                    v50 = *v48 + 512;
                    v46 = *v48 + 8 * (v60 - (v47 << 6));
                  }
                  v130[1] = v49;
                  v130[0] = v46;
                  v130[2] = v50;
                  v130[3] = (__int64)v48;
                  v131 = v56;
                  v132 = v57;
                  v133 = v58;
                  v134 = v59;
                  v129[0] = (__int64)v147;
                  v51 = *v150;
                  v129[3] = (__int64)v150;
                  v129[1] = v51;
                  v129[2] = v51 + 512;
                  sub_276F640((unsigned __int64 *)&v141, v129, v130, &v131);
                  v52 = v147;
                  v131 = v120;
                  if ( v147 == (__int64 *)(v149 - 8) )
                  {
                    sub_27698B0((unsigned __int64 *)&v141, &v131);
                  }
                  else
                  {
                    if ( v147 )
                    {
                      *v147 = v120;
                      v52 = v147;
                    }
                    v147 = v52 + 1;
                  }
                  v53 = a1[1];
                  if ( v53 == a1[2] )
                  {
                    sub_276BEE0((unsigned __int64 *)a1, a1[1], &v141);
                  }
                  else
                  {
                    if ( v53 )
                    {
                      v108 = a1[1];
                      sub_276A0C0((__int64 *)v108, &v141);
                      v54 = v152;
                      v55 = v108;
                      *(_DWORD *)(v108 + 88) = v152;
                      if ( v54 > 0x40 )
                      {
                        sub_C43780(v108 + 80, (const void **)&v151);
                        v55 = v108;
                      }
                      else
                      {
                        *(_QWORD *)(v108 + 80) = v151;
                      }
                      *(_QWORD *)(v55 + 96) = v153;
                      *(_BYTE *)(v55 + 104) = v154;
                      v53 = a1[1];
                    }
                    a1[1] = v53 + 112;
                  }
                  if ( v152 > 0x40 && v151 )
                    j_j___libc_free_0_0(v151);
                  v45 += 10;
                  sub_2767770((unsigned __int64 *)&v141);
                  if ( v114 == v45 )
                    goto LABEL_115;
                  continue;
                }
              }
LABEL_116:
              sub_2767860(&v127);
            }
            sub_27677F0((unsigned __int64 *)&v124);
            goto LABEL_12;
          }
          sub_276F9C0(&v141, a2, v112, v26, v117);
          v110 = v142;
          if ( v142 == v141 )
          {
LABEL_134:
            sub_2767860((unsigned __int64 *)&v141);
            goto LABEL_12;
          }
          v107 = v16;
          v62 = v141;
          while ( 2 )
          {
            v131 = v120;
            v63 = (__int64 *)v62[3].m128i_i64[0];
            if ( v63 == (__int64 *)(v62[4].m128i_i64[0] - 8) )
            {
              sub_27698B0((unsigned __int64 *)v62, &v131);
            }
            else
            {
              if ( v63 )
              {
                *v63 = v120;
                v63 = (__int64 *)v62[3].m128i_i64[0];
              }
              v62[3].m128i_i64[0] = (__int64)(v63 + 1);
            }
            v64 = a1[1];
            if ( v64 != a1[2] )
            {
              if ( v64 )
              {
                *(_QWORD *)v64 = 0;
                *(_QWORD *)(v64 + 8) = 0;
                *(_QWORD *)(v64 + 16) = 0;
                *(_QWORD *)(v64 + 24) = 0;
                *(_QWORD *)(v64 + 32) = 0;
                *(_QWORD *)(v64 + 40) = 0;
                *(_QWORD *)(v64 + 48) = 0;
                *(_QWORD *)(v64 + 56) = 0;
                *(_QWORD *)(v64 + 64) = 0;
                *(_QWORD *)(v64 + 72) = 0;
                sub_2768EA0((__int64 *)v64, 0);
                if ( v62->m128i_i64[0] )
                {
                  v65 = *(_QWORD *)(v64 + 24);
                  v66 = *(_QWORD *)(v64 + 32);
                  *(_QWORD *)(v64 + 24) = 0;
                  *(_QWORD *)(v64 + 32) = 0;
                  v67 = *(_QWORD *)(v64 + 40);
                  v68 = *(_QWORD *)(v64 + 48);
                  v69 = *(_QWORD *)(v64 + 56);
                  *(_QWORD *)(v64 + 40) = 0;
                  *(_QWORD *)(v64 + 48) = 0;
                  v70 = *(_QWORD *)(v64 + 64);
                  *(_QWORD *)(v64 + 56) = 0;
                  v71 = *(_QWORD *)(v64 + 72);
                  *(_QWORD *)(v64 + 64) = 0;
                  v72 = *(_QWORD *)v64;
                  *(_QWORD *)(v64 + 72) = 0;
                  v73 = *(_QWORD *)(v64 + 8);
                  *(_QWORD *)v64 = 0;
                  v74 = *(_QWORD *)(v64 + 16);
                  *(_QWORD *)(v64 + 8) = 0;
                  *(_QWORD *)(v64 + 16) = 0;
                  *(__m128i *)v64 = _mm_loadu_si128(v62);
                  *(__m128i *)(v64 + 16) = _mm_loadu_si128(v62 + 1);
                  *(__m128i *)(v64 + 32) = _mm_loadu_si128(v62 + 2);
                  *(__m128i *)(v64 + 48) = _mm_loadu_si128(v62 + 3);
                  *(__m128i *)(v64 + 64) = _mm_loadu_si128(v62 + 4);
                  v62->m128i_i64[0] = v72;
                  v62->m128i_i64[1] = v73;
                  v62[1].m128i_i64[0] = v74;
                  v62[1].m128i_i64[1] = v65;
                  v62[2].m128i_i64[0] = v66;
                  v62[2].m128i_i64[1] = v67;
                  v62[3].m128i_i64[0] = v68;
                  v62[3].m128i_i64[1] = v69;
                  v62[4].m128i_i64[0] = v70;
                  v62[4].m128i_i64[1] = v71;
                }
                *(_DWORD *)(v64 + 88) = v62[5].m128i_i32[2];
                *(_QWORD *)(v64 + 80) = v62[5].m128i_i64[0];
                v75 = v62[6].m128i_i64[0];
                v62[5].m128i_i32[2] = 0;
                *(_QWORD *)(v64 + 96) = v75;
                *(_BYTE *)(v64 + 104) = v62[6].m128i_i8[8];
                v64 = a1[1];
              }
              a1[1] = v64 + 112;
              goto LABEL_130;
            }
            v76 = *a1;
            v77 = v64 - *a1;
            v78 = 0x6DB6DB6DB6DB6DB7LL * (v77 >> 4);
            if ( v78 == 0x124924924924924LL )
              sub_4262D8((__int64)"vector::_M_realloc_insert");
            v79 = 1;
            if ( v78 )
              v79 = 0x6DB6DB6DB6DB6DB7LL * ((v64 - *a1) >> 4);
            v80 = __CFADD__(v79, v78);
            v81 = v79 + v78;
            if ( !v80 )
            {
              if ( v81 )
              {
                if ( v81 > 0x124924924924924LL )
                  v81 = 0x124924924924924LL;
LABEL_142:
                v82 = 112 * v81;
                v83 = sub_22077B0(112 * v81);
                v111 = v82 + v83;
                v115 = v83 + 112;
              }
              else
              {
                v115 = 112;
                v83 = 0;
                v111 = 0;
              }
              v84 = v83 + v77;
              if ( v84 )
              {
                *(_QWORD *)v84 = 0;
                *(_QWORD *)(v84 + 8) = 0;
                *(_QWORD *)(v84 + 16) = 0;
                *(_QWORD *)(v84 + 24) = 0;
                *(_QWORD *)(v84 + 32) = 0;
                *(_QWORD *)(v84 + 40) = 0;
                *(_QWORD *)(v84 + 48) = 0;
                *(_QWORD *)(v84 + 56) = 0;
                *(_QWORD *)(v84 + 64) = 0;
                *(_QWORD *)(v84 + 72) = 0;
                sub_2768EA0((__int64 *)v84, 0);
                if ( v62->m128i_i64[0] )
                {
                  v85 = *(_QWORD *)(v84 + 48);
                  v86 = *(_QWORD *)(v84 + 32);
                  *(_QWORD *)(v84 + 48) = 0;
                  v87 = *(_QWORD *)(v84 + 16);
                  v88 = *(_QWORD *)(v84 + 24);
                  *(_QWORD *)(v84 + 16) = 0;
                  v89 = *(_QWORD *)(v84 + 40);
                  v90 = *(_QWORD *)(v84 + 56);
                  *(_QWORD *)(v84 + 24) = 0;
                  v91 = *(_QWORD *)(v84 + 64);
                  v92 = *(_QWORD *)v84;
                  *(_QWORD *)(v84 + 32) = 0;
                  *(_QWORD *)v84 = 0;
                  v93 = *(_QWORD *)(v84 + 8);
                  *(_QWORD *)(v84 + 40) = 0;
                  *(_QWORD *)(v84 + 8) = 0;
                  *(_QWORD *)(v84 + 56) = 0;
                  *(_QWORD *)(v84 + 64) = 0;
                  v109 = v85;
                  v94 = *(_QWORD *)(v84 + 72);
                  *(_QWORD *)(v84 + 72) = 0;
                  *(__m128i *)v84 = _mm_loadu_si128(v62);
                  *(__m128i *)(v84 + 16) = _mm_loadu_si128(v62 + 1);
                  *(__m128i *)(v84 + 32) = _mm_loadu_si128(v62 + 2);
                  *(__m128i *)(v84 + 48) = _mm_loadu_si128(v62 + 3);
                  *(__m128i *)(v84 + 64) = _mm_loadu_si128(v62 + 4);
                  v62->m128i_i64[0] = v92;
                  v62->m128i_i64[1] = v93;
                  v62[2].m128i_i64[0] = v86;
                  v62[1].m128i_i64[0] = v87;
                  v62[1].m128i_i64[1] = v88;
                  v62[2].m128i_i64[1] = v89;
                  v62[3].m128i_i64[0] = v109;
                  v62[3].m128i_i64[1] = v90;
                  v62[4].m128i_i64[0] = v91;
                  v62[4].m128i_i64[1] = v94;
                }
                *(_DWORD *)(v84 + 88) = v62[5].m128i_i32[2];
                *(_QWORD *)(v84 + 80) = v62[5].m128i_i64[0];
                v95 = v62[6].m128i_i64[0];
                v62[5].m128i_i32[2] = 0;
                *(_QWORD *)(v84 + 96) = v95;
                *(_BYTE *)(v84 + 104) = v62[6].m128i_i8[8];
              }
              if ( v64 != v76 )
              {
                v96 = v83;
                v116 = v83;
                v97 = v76;
                v98 = v96;
                v99 = v76;
                for ( j = v98; ; j += 112 )
                {
                  if ( j )
                  {
                    sub_276A0C0((__int64 *)j, (_QWORD *)v97);
                    v101 = *(_DWORD *)(v97 + 88);
                    *(_DWORD *)(j + 88) = v101;
                    if ( v101 <= 0x40 )
                      *(_QWORD *)(j + 80) = *(_QWORD *)(v97 + 80);
                    else
                      sub_C43780(j + 80, (const void **)(v97 + 80));
                    *(_QWORD *)(j + 96) = *(_QWORD *)(v97 + 96);
                    *(_BYTE *)(j + 104) = *(_BYTE *)(v97 + 104);
                  }
                  v97 += 112LL;
                  if ( v64 == v97 )
                    break;
                }
                v102 = j;
                v83 = v116;
                v76 = v99;
                v115 = v102 + 224;
                do
                {
                  if ( *(_DWORD *)(v99 + 88) > 0x40u )
                  {
                    v103 = *(_QWORD *)(v99 + 80);
                    if ( v103 )
                      j_j___libc_free_0_0(v103);
                  }
                  v104 = (unsigned __int64 *)v99;
                  v99 += 112LL;
                  sub_2767770(v104);
                }
                while ( v64 != v99 );
              }
              if ( v76 )
                j_j___libc_free_0(v76);
              *a1 = v83;
              a1[1] = v115;
              a1[2] = v111;
LABEL_130:
              v62 += 7;
              if ( v110 == v62 )
              {
                v16 = v107;
                goto LABEL_134;
              }
              continue;
            }
            break;
          }
          v81 = 0x124924924924924LL;
          goto LABEL_142;
        }
      }
      else
      {
LABEL_25:
        v26 = *(_QWORD *)(v23 + v24);
        if ( *(_BYTE *)v26 != 17 )
          goto LABEL_64;
      }
      if ( a2[2] == v120 && v120 != *(_QWORD *)(**(_QWORD **)(a2[1] - 8LL) + 40LL) )
        goto LABEL_12;
      v141 = 0;
      v142 = 0;
      v143 = 0;
      v144 = 0;
      v145 = 0;
      v146 = 0;
      v147 = 0;
      v148 = 0;
      v149 = 0;
      v150 = 0;
      sub_2768EA0((__int64 *)&v141, 0);
      v152 = 1;
      v151 = 0;
      v154 = 0;
      v153 = v120;
      if ( *(_DWORD *)(v26 + 32) <= 0x40u )
      {
        v151 = *(_QWORD *)(v26 + 24);
        v152 = *(_DWORD *)(v26 + 32);
      }
      else
      {
        sub_C43990((__int64)&v151, v26 + 24);
      }
      v27 = v147;
      v154 = 1;
      v28 = (__int64 *)(v149 - 8);
      if ( a2[2] != v17 )
      {
        v131 = v17;
        if ( v28 == v147 )
        {
          sub_27698B0((unsigned __int64 *)&v141, &v131);
          v27 = v147;
          v28 = (__int64 *)(v149 - 8);
        }
        else
        {
          if ( v147 )
          {
            *v147 = v17;
            v27 = v147;
            v28 = (__int64 *)(v149 - 8);
          }
          v147 = ++v27;
        }
      }
      v131 = v120;
      if ( v28 == v27 )
      {
        sub_27698B0((unsigned __int64 *)&v141, &v131);
      }
      else
      {
        if ( v27 )
        {
          *v27 = v120;
          v27 = v147;
        }
        v147 = v27 + 1;
      }
      v29 = a1[1];
      if ( v29 == a1[2] )
      {
        sub_276BEE0((unsigned __int64 *)a1, a1[1], &v141);
      }
      else
      {
        if ( v29 )
        {
          sub_276A0C0((__int64 *)a1[1], &v141);
          v30 = v152;
          *(_DWORD *)(v29 + 88) = v152;
          if ( v30 > 0x40 )
            sub_C43780(v29 + 80, (const void **)&v151);
          else
            *(_QWORD *)(v29 + 80) = v151;
          *(_QWORD *)(v29 + 96) = v153;
          *(_BYTE *)(v29 + 104) = v154;
          v29 = a1[1];
        }
        a1[1] = v29 + 112;
      }
      if ( v152 > 0x40 && v151 )
        j_j___libc_free_0_0(v151);
      ++v16;
      sub_2767770((unsigned __int64 *)&v141);
      if ( v123 == v16 )
        goto LABEL_49;
    }
  }
LABEL_49:
  v6 = v117;
  v7 = v120;
  LOBYTE(v8) = *(_BYTE *)(v117 + 28);
LABEL_50:
  if ( (_BYTE)v8 )
  {
    v31 = *(_QWORD **)(v6 + 8);
    v32 = &v31[*(unsigned int *)(v6 + 20)];
    v33 = v31;
    if ( v31 != v32 )
    {
      while ( v7 != *v33 )
      {
        if ( v32 == ++v33 )
          goto LABEL_56;
      }
      v34 = (unsigned int)(*(_DWORD *)(v6 + 20) - 1);
      *(_DWORD *)(v6 + 20) = v34;
      *v33 = v31[v34];
      ++*(_QWORD *)v6;
    }
  }
  else
  {
    v61 = sub_C8CA60(v6, v7);
    if ( v61 )
    {
      *v61 = -2;
      ++*(_DWORD *)(v6 + 24);
      ++*(_QWORD *)v6;
    }
  }
LABEL_56:
  if ( !v139 )
    _libc_free((unsigned __int64)v136);
  return a1;
}
