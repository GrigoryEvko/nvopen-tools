// Function: sub_2FA04B0
// Address: 0x2fa04b0
//
__m128i *__fastcall sub_2FA04B0(__m128i *a1, __int64 *a2, unsigned __int8 *a3)
{
  __int64 v5; // rdx
  int v6; // eax
  unsigned __int8 *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rcx
  _QWORD *v10; // rax
  char *v11; // rdx
  __int64 v12; // rax
  char *v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 *v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned int v19; // edx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  char v22; // al
  __int64 *v23; // r8
  _BOOL4 v24; // edi
  __int64 v25; // rcx
  __int64 v26; // r14
  __int64 v27; // r11
  __int64 v28; // r9
  int v29; // r15d
  unsigned __int8 *v30; // rdx
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 *v33; // rsi
  __int64 v34; // r10
  __int64 v35; // rdx
  __int64 v36; // rdi
  char v37; // al
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rsi
  __m128i v41; // xmm1
  unsigned __int8 *v42; // rdx
  _BYTE *v43; // r13
  __int8 v44; // al
  __int64 v45; // rsi
  __m128i v46; // xmm5
  __int64 v48; // rax
  __int64 v49; // r14
  __int8 v50; // al
  __int64 v51; // rsi
  __m128i v52; // xmm3
  __int64 v53; // rax
  __int64 *v54; // r13
  unsigned __int8 *v55; // rcx
  __m128i *v56; // rbx
  __int64 v57; // r12
  __int64 v58; // r15
  __int16 v59; // ax
  bool v60; // al
  __int64 v61; // rdx
  __int64 v62; // r8
  bool v63; // r14
  __int8 v64; // al
  __int64 v65; // rbx
  __int64 v66; // rsi
  __m128i v67; // xmm7
  bool v68; // al
  __int64 v69; // rdx
  __int64 v70; // r8
  bool v71; // r14
  _BYTE *v72; // rdi
  char v73; // al
  _BYTE *v74; // rdi
  char v75; // al
  __int64 *v76; // rax
  __int64 v77; // rcx
  __int64 v78; // rdx
  __int64 v79; // rdx
  unsigned int v80; // r14d
  int v81; // eax
  bool v82; // al
  unsigned int v83; // r14d
  int v84; // eax
  bool v85; // al
  char v86; // al
  __int64 v87; // rsi
  __int64 v88; // rax
  int v89; // eax
  _BYTE *v90; // rax
  unsigned __int8 *v91; // r8
  unsigned int v92; // r14d
  int v93; // eax
  int v94; // eax
  _BYTE *v95; // rax
  unsigned __int8 *v96; // r8
  unsigned int v97; // r14d
  int v98; // eax
  __int64 *v99; // rdx
  char v100; // si
  __int64 v101; // rsi
  int v102; // esi
  __int64 v103; // rax
  __int8 v104; // dl
  char v105; // cl
  char v106; // al
  __int64 v107; // rsi
  int i; // r9d
  __int64 v109; // rax
  int v110; // r9d
  unsigned int v111; // r14d
  int v112; // eax
  int j; // r9d
  __int64 v114; // rax
  int v115; // r9d
  unsigned int v116; // r14d
  int v117; // eax
  __int64 v118; // rax
  int v119; // [rsp+4h] [rbp-BCh]
  int v120; // [rsp+4h] [rbp-BCh]
  unsigned __int8 *v121; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v122; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v123; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v124; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v125; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v126; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v127; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v128; // [rsp+10h] [rbp-B0h]
  int v129; // [rsp+10h] [rbp-B0h]
  int v130; // [rsp+10h] [rbp-B0h]
  __int64 v131; // [rsp+18h] [rbp-A8h]
  __int64 v132; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v133; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v134; // [rsp+18h] [rbp-A8h]
  __int64 v135; // [rsp+18h] [rbp-A8h]
  __int64 v136; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v137; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v138; // [rsp+18h] [rbp-A8h]
  __int64 *v140; // [rsp+28h] [rbp-98h]
  int v141; // [rsp+28h] [rbp-98h]
  _BYTE *v142; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int8 *v143; // [rsp+40h] [rbp-80h] BYREF
  __m128i v144; // [rsp+48h] [rbp-78h] BYREF
  __m128i v145; // [rsp+60h] [rbp-60h] BYREF
  __m128i v146[5]; // [rsp+70h] [rbp-50h] BYREF

  v5 = *((_QWORD *)a3 + 2);
  v6 = *a3;
  if ( v5 && !*(_QWORD *)(v5 + 8) && ((_BYTE)v6 == 68 || (_BYTE)v6 == 69) )
  {
    v35 = *((_QWORD *)a3 - 4);
    if ( !v35 )
      goto LABEL_58;
    v36 = *(_QWORD *)(v35 + 8);
    v142 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( sub_BCAC40(v36, 1) )
    {
      v49 = (__int64)v142;
      v145.m128i_i64[0] = 0;
      v145.m128i_i64[1] = (__int64)&v142;
      v50 = 0;
      if ( *v142 == 59 )
      {
        v86 = sub_995B10(&v145, *((_QWORD *)v142 - 8));
        v87 = *(_QWORD *)(v49 - 32);
        if ( v86 && v87 )
        {
          *(_QWORD *)v145.m128i_i64[1] = v87;
          v50 = 1;
          v49 = (__int64)v142;
        }
        else if ( (unsigned __int8)sub_995B10(&v145, v87) && (v88 = *(_QWORD *)(v49 - 64)) != 0 )
        {
          *(_QWORD *)v145.m128i_i64[1] = v88;
          v50 = 1;
          v49 = (__int64)v142;
        }
        else
        {
          v49 = (__int64)v142;
          v50 = 0;
        }
      }
      v143 = a3;
      v144.m128i_i64[0] = v49;
      v51 = *a2;
      v144.m128i_i8[8] = 1;
      v144.m128i_i8[9] = v50;
      v144.m128i_i32[3] = 0;
      sub_2F9B840((__int64)&v145, v51, (__int64 *)&v143, &v144);
      v52 = _mm_loadu_si128(v146);
      *a1 = _mm_loadu_si128(&v145);
      a1[1] = v52;
      return a1;
    }
    v6 = *a3;
  }
  v145.m128i_i64[0] = 0;
  v145.m128i_i64[1] = (__int64)&v142;
  if ( (_BYTE)v6 == 59 )
  {
    v37 = sub_995B10(&v145, *((_QWORD *)a3 - 8));
    v38 = *((_QWORD *)a3 - 4);
    if ( v37 && v38 )
    {
      *(_QWORD *)v145.m128i_i64[1] = v38;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v145, v38) || (v39 = *((_QWORD *)a3 - 8)) == 0 )
      {
        v6 = *a3;
        goto LABEL_3;
      }
      *(_QWORD *)v145.m128i_i64[1] = v39;
    }
    v143 = a3;
    v144.m128i_i32[3] = 0;
    v40 = *a2;
    v144.m128i_i64[0] = (__int64)v142;
    v144.m128i_i16[4] = 257;
    sub_2F9B840((__int64)&v145, v40, (__int64 *)&v143, &v144);
    v41 = _mm_loadu_si128(v146);
    *a1 = _mm_loadu_si128(&v145);
    a1[1] = v41;
    return a1;
  }
LABEL_3:
  if ( (_BYTE)v6 == 86 )
  {
    if ( (a3[7] & 0x40) != 0 )
      v42 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
    else
      v42 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v43 = *(_BYTE **)v42;
    if ( *(_QWORD *)v42 )
    {
      v142 = *(_BYTE **)v42;
      v145.m128i_i64[0] = 0;
      v145.m128i_i64[1] = (__int64)&v142;
      if ( *v43 == 59 )
      {
        v106 = sub_995B10(&v145, *((_QWORD *)v43 - 8));
        v107 = *((_QWORD *)v43 - 4);
        if ( v106 && v107 )
        {
          *(_QWORD *)v145.m128i_i64[1] = v107;
          v44 = 1;
          v43 = v142;
        }
        else if ( (unsigned __int8)sub_995B10(&v145, v107) && (v118 = *((_QWORD *)v43 - 8)) != 0 )
        {
          *(_QWORD *)v145.m128i_i64[1] = v118;
          v44 = 1;
          v43 = v142;
        }
        else
        {
          v43 = v142;
          v44 = 0;
        }
      }
      else
      {
        v44 = 0;
      }
      v143 = a3;
      v144.m128i_i64[0] = (__int64)v43;
      v45 = *a2;
      v144.m128i_i8[8] = 0;
      v144.m128i_i8[9] = v44;
      v144.m128i_i32[3] = 0;
      sub_2F9B840((__int64)&v145, v45, (__int64 *)&v143, &v144);
      v46 = _mm_loadu_si128(v146);
      *a1 = _mm_loadu_si128(&v145);
      a1[1] = v46;
      return a1;
    }
    goto LABEL_59;
  }
  if ( (unsigned int)(v6 - 55) > 1 )
  {
LABEL_58:
    if ( (unsigned int)(v6 - 42) > 0x11 )
      goto LABEL_59;
    goto LABEL_11;
  }
  if ( (a3[7] & 0x40) != 0 )
  {
    v7 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
    v8 = *(_QWORD *)v7;
    if ( !*(_QWORD *)v7 )
      goto LABEL_11;
  }
  else
  {
    v7 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v8 = *(_QWORD *)v7;
    if ( !*(_QWORD *)v7 )
      goto LABEL_11;
  }
  v9 = *((_QWORD *)v7 + 4);
  if ( *(_BYTE *)v9 == 17 )
  {
    v10 = *(_QWORD **)(v9 + 24);
    if ( *(_DWORD *)(v9 + 32) > 0x40u )
      v10 = (_QWORD *)*v10;
    if ( (_QWORD *)(*(_DWORD *)(*((_QWORD *)a3 + 1) + 8LL) >> 8) == (_QWORD *)((char *)v10 + 1) )
    {
      v53 = a2[1];
      v54 = *(__int64 **)(v53 + 32);
      v140 = &v54[*(unsigned int *)(v53 + 40)];
      if ( v54 == v140 )
        goto LABEL_86;
      v55 = a3;
      v56 = a1;
      v57 = v8;
LABEL_66:
      v58 = *v54;
      if ( *(_QWORD *)(*v54 - 64) != v57 )
        goto LABEL_65;
      v59 = *(_WORD *)(v58 + 2) & 0x3F;
      if ( v59 == 38 )
      {
        v74 = *(_BYTE **)(v58 - 32);
        if ( *v74 != 17 )
          goto LABEL_65;
        v134 = v55;
        v75 = sub_2F9A750((__int64)v74);
        v55 = v134;
        if ( !v75 )
          goto LABEL_65;
LABEL_71:
        a1 = v56;
        v64 = 1;
        v65 = (__int64)v55;
        goto LABEL_72;
      }
      if ( v59 != 39 )
      {
        if ( v59 == 40 )
        {
          if ( **(_BYTE **)(v58 - 32) > 0x15u )
            goto LABEL_65;
          v126 = v55;
          v132 = *(_QWORD *)(v58 - 32);
          v68 = sub_AC30F0(v132);
          v70 = v132;
          v55 = v126;
          v71 = v68;
          if ( v68 )
            goto LABEL_77;
          if ( *(_BYTE *)v132 == 17 )
          {
            v83 = *(_DWORD *)(v132 + 32);
            if ( v83 <= 0x40 )
            {
              v85 = *(_QWORD *)(v132 + 24) == 0;
            }
            else
            {
              v84 = sub_C444A0(v132 + 24);
              v55 = v126;
              v85 = v83 == v84;
            }
          }
          else
          {
            v94 = *(unsigned __int8 *)(*(_QWORD *)(v132 + 8) + 8LL);
            v136 = *(_QWORD *)(v132 + 8);
            if ( (unsigned int)(v94 - 17) > 1 )
              goto LABEL_65;
            v122 = v126;
            v128 = (unsigned __int8 *)v70;
            v95 = sub_AD7630(v70, 0, v69);
            v96 = v128;
            v55 = v122;
            if ( !v95 || *v95 != 17 )
            {
              if ( *(_BYTE *)(v136 + 8) != 17 )
                goto LABEL_65;
              v119 = *(_DWORD *)(v136 + 32);
              if ( !v119 )
                goto LABEL_65;
              for ( i = 1; ; i = v110 + 1 )
              {
                v123 = v55;
                v129 = i;
                v137 = v96;
                v109 = sub_AD69F0(v96, (unsigned int)(i - 1));
                v55 = v123;
                if ( !v109 )
                  goto LABEL_65;
                v96 = v137;
                v110 = v129;
                if ( *(_BYTE *)v109 == 13 )
                {
                  if ( v129 == v119 )
                  {
                    if ( !v71 )
                      goto LABEL_65;
                    goto LABEL_77;
                  }
                }
                else
                {
                  if ( *(_BYTE *)v109 != 17 )
                    goto LABEL_65;
                  v111 = *(_DWORD *)(v109 + 32);
                  if ( v111 <= 0x40 )
                  {
                    v71 = *(_QWORD *)(v109 + 24) == 0;
                  }
                  else
                  {
                    v112 = sub_C444A0(v109 + 24);
                    v96 = v137;
                    v110 = v129;
                    v55 = v123;
                    v71 = v111 == v112;
                  }
                  if ( !v71 )
                    goto LABEL_65;
                  if ( v110 == v119 )
                    goto LABEL_77;
                }
              }
            }
            v97 = *((_DWORD *)v95 + 8);
            if ( v97 <= 0x40 )
            {
              v85 = *((_QWORD *)v95 + 3) == 0;
            }
            else
            {
              v98 = sub_C444A0((__int64)(v95 + 24));
              v55 = v122;
              v85 = v97 == v98;
            }
          }
          if ( !v85 )
            goto LABEL_65;
        }
        else
        {
          if ( v59 != 41 )
            goto LABEL_65;
          v72 = *(_BYTE **)(v58 - 32);
          if ( *v72 != 17 )
            goto LABEL_65;
          v133 = v55;
          v73 = sub_2F9A750((__int64)v72);
          v55 = v133;
          if ( !v73 )
            goto LABEL_65;
        }
LABEL_77:
        a1 = v56;
        v64 = 0;
        v65 = (__int64)v55;
LABEL_72:
        v143 = (unsigned __int8 *)v65;
        v144.m128i_i64[0] = v58;
        v144.m128i_i8[8] = 1;
        v66 = *a2;
        v144.m128i_i8[9] = v64;
        v144.m128i_i32[3] = 0;
        goto LABEL_73;
      }
      if ( **(_BYTE **)(v58 - 32) > 0x15u )
        goto LABEL_65;
      v125 = v55;
      v131 = *(_QWORD *)(v58 - 32);
      v60 = sub_AC30F0(v131);
      v62 = v131;
      v55 = v125;
      v63 = v60;
      if ( v60 )
        goto LABEL_71;
      if ( *(_BYTE *)v131 == 17 )
      {
        v80 = *(_DWORD *)(v131 + 32);
        if ( v80 <= 0x40 )
        {
          v82 = *(_QWORD *)(v131 + 24) == 0;
        }
        else
        {
          v81 = sub_C444A0(v131 + 24);
          v55 = v125;
          v82 = v80 == v81;
        }
        goto LABEL_90;
      }
      v89 = *(unsigned __int8 *)(*(_QWORD *)(v131 + 8) + 8LL);
      v135 = *(_QWORD *)(v131 + 8);
      if ( (unsigned int)(v89 - 17) > 1 )
        goto LABEL_65;
      v121 = v125;
      v127 = (unsigned __int8 *)v62;
      v90 = sub_AD7630(v62, 0, v61);
      v91 = v127;
      v55 = v121;
      if ( v90 && *v90 == 17 )
      {
        v92 = *((_DWORD *)v90 + 8);
        if ( v92 <= 0x40 )
        {
          if ( !*((_QWORD *)v90 + 3) )
            goto LABEL_71;
          goto LABEL_65;
        }
        v93 = sub_C444A0((__int64)(v90 + 24));
        v55 = v121;
        v82 = v92 == v93;
LABEL_90:
        if ( v82 )
          goto LABEL_71;
        goto LABEL_65;
      }
      if ( *(_BYTE *)(v135 + 8) != 17 )
        goto LABEL_65;
      v120 = *(_DWORD *)(v135 + 32);
      if ( !v120 )
        goto LABEL_65;
      for ( j = 1; ; j = v115 + 1 )
      {
        v124 = v55;
        v130 = j;
        v138 = v91;
        v114 = sub_AD69F0(v91, (unsigned int)(j - 1));
        v55 = v124;
        if ( !v114 )
          goto LABEL_65;
        v91 = v138;
        v115 = v130;
        if ( *(_BYTE *)v114 == 13 )
        {
          if ( v120 == v130 )
          {
            if ( v63 )
              goto LABEL_71;
LABEL_65:
            if ( v140 == ++v54 )
            {
              a1 = v56;
              goto LABEL_86;
            }
            goto LABEL_66;
          }
        }
        else
        {
          if ( *(_BYTE *)v114 != 17 )
            goto LABEL_65;
          v116 = *(_DWORD *)(v114 + 32);
          if ( v116 <= 0x40 )
          {
            v63 = *(_QWORD *)(v114 + 24) == 0;
          }
          else
          {
            v117 = sub_C444A0(v114 + 24);
            v91 = v138;
            v115 = v130;
            v55 = v124;
            v63 = v116 == v117;
          }
          if ( !v63 )
            goto LABEL_65;
          if ( v120 == v115 )
            goto LABEL_71;
        }
      }
    }
  }
LABEL_11:
  v11 = (char *)*((_QWORD *)a3 - 4);
  v12 = *((_QWORD *)v11 + 2);
  if ( !v12 )
  {
    v13 = (char *)*((_QWORD *)a3 - 8);
    v14 = *((_QWORD *)v13 + 2);
    if ( !v14 )
      goto LABEL_59;
LABEL_13:
    if ( !*(_QWORD *)(v14 + 8) )
    {
      v100 = *v13;
      if ( (unsigned __int8)*v13 > 0x1Cu && (v100 == 68 || v100 == 69) )
      {
        v101 = *((_QWORD *)v13 - 4);
        if ( v101 )
          goto LABEL_133;
      }
    }
    if ( !v12 )
    {
      v15 = *((_QWORD *)v13 + 2);
LABEL_16:
      if ( !*(_QWORD *)(v15 + 8) && (unsigned __int8)(*v13 - 55) <= 1u )
      {
        if ( (v13[7] & 0x40) != 0 )
        {
          v16 = (__int64 *)*((_QWORD *)v13 - 1);
          v17 = *v16;
          if ( !*v16 )
            goto LABEL_193;
        }
        else
        {
          v16 = (__int64 *)&v13[-32 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)];
          v17 = *v16;
          if ( !*v16 )
            goto LABEL_193;
        }
        v18 = v16[4];
        if ( *(_BYTE *)v18 == 17 )
          goto LABEL_21;
LABEL_193:
        v23 = (__int64 *)*a2;
        v28 = *(_QWORD *)(*a2 + 8) + 24LL * *(unsigned int *)(*a2 + 24);
        goto LABEL_60;
      }
LABEL_59:
      v23 = (__int64 *)*a2;
      v28 = *(_QWORD *)(*a2 + 8) + 24LL * *(unsigned int *)(*a2 + 24);
LABEL_60:
      v48 = *v23;
      a1->m128i_i64[0] = (__int64)v23;
      a1[1].m128i_i64[0] = v28;
      a1->m128i_i64[1] = v48;
      a1[1].m128i_i64[1] = v28;
      return a1;
    }
LABEL_101:
    if ( !*(_QWORD *)(v12 + 8) && (unsigned __int8)(*v11 - 55) <= 1u )
    {
      if ( (v11[7] & 0x40) != 0 )
      {
        v99 = (__int64 *)*((_QWORD *)v11 - 1);
        v17 = *v99;
        if ( !*v99 )
          goto LABEL_102;
      }
      else
      {
        v99 = (__int64 *)&v11[-32 * (*((_DWORD *)v11 + 1) & 0x7FFFFFF)];
        v17 = *v99;
        if ( !*v99 )
          goto LABEL_102;
      }
      v18 = v99[4];
      if ( *(_BYTE *)v18 == 17 )
      {
LABEL_21:
        v19 = *(_DWORD *)(*(_QWORD *)(v17 + 8) + 8LL);
        v20 = *(_QWORD **)(v18 + 24);
        v21 = (_QWORD *)(v19 >> 8);
        if ( *(_DWORD *)(v18 + 32) > 0x40u )
          v20 = (_QWORD *)*v20;
        if ( v21 == (_QWORD *)((char *)v20 + 1) )
          goto LABEL_24;
        goto LABEL_59;
      }
    }
LABEL_102:
    v13 = (char *)*((_QWORD *)a3 - 8);
    v15 = *((_QWORD *)v13 + 2);
    if ( !v15 )
      goto LABEL_59;
    goto LABEL_16;
  }
  if ( *(_QWORD *)(v12 + 8)
    || (v105 = *v11, (unsigned __int8)*v11 <= 0x1Cu)
    || v105 != 68 && v105 != 69
    || (v101 = *((_QWORD *)v11 - 4)) == 0 )
  {
    v13 = (char *)*((_QWORD *)a3 - 8);
    v14 = *((_QWORD *)v13 + 2);
    if ( !v14 )
      goto LABEL_101;
    goto LABEL_13;
  }
LABEL_133:
  if ( !sub_BCAC40(*(_QWORD *)(v101 + 8), 1) )
  {
    if ( (unsigned int)*a3 - 42 > 0x11 )
      goto LABEL_193;
    v11 = (char *)*((_QWORD *)a3 - 4);
    v12 = *((_QWORD *)v11 + 2);
    if ( !v12 )
      goto LABEL_102;
    goto LABEL_101;
  }
LABEL_24:
  v22 = *a3;
  if ( *a3 != 42 )
  {
    if ( v22 == 58 )
    {
      if ( sub_BCAC40(*((_QWORD *)a3 + 1), 1) )
        goto LABEL_86;
      v22 = *a3;
      goto LABEL_27;
    }
    if ( v22 != 44 )
    {
LABEL_86:
      v76 = (__int64 *)*a2;
      v77 = 3LL * *(unsigned int *)(*a2 + 24);
      v78 = *(_QWORD *)(*a2 + 8);
      a1->m128i_i64[0] = *a2;
      v79 = v78 + 8 * v77;
      a1->m128i_i64[1] = *v76;
      a1[1].m128i_i64[0] = v79;
      a1[1].m128i_i64[1] = v79;
      return a1;
    }
  }
LABEL_27:
  v23 = (__int64 *)*a2;
  v24 = v22 == 44;
  v25 = 32LL * (v22 == 44);
  v26 = *(_QWORD *)(*a2 + 8);
  v27 = *(unsigned int *)(*a2 + 24);
  v28 = v26 + 24 * v27;
  v29 = v27 - 1;
  while ( 1 )
  {
    if ( (a3[7] & 0x40) != 0 )
      v30 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
    else
      v30 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v31 = *(_QWORD *)&v30[v25];
    if ( !(_DWORD)v27 )
      goto LABEL_33;
    v32 = v29 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v33 = (__int64 *)(v26 + 24LL * v32);
    v34 = *v33;
    if ( *v33 == v31 )
      break;
    v102 = 1;
    while ( v34 != -4096 )
    {
      v32 = v29 & (v102 + v32);
      v141 = v102 + 1;
      v33 = (__int64 *)(v26 + 24LL * v32);
      v34 = *v33;
      if ( v31 == *v33 )
        goto LABEL_31;
      v102 = v141;
    }
LABEL_33:
    v25 += 32;
    if ( v24 )
      goto LABEL_60;
    v24 = 1;
  }
LABEL_31:
  if ( (__int64 *)v28 == v33 || !*((_BYTE *)v33 + 16) )
    goto LABEL_33;
  v103 = v33[1];
  v104 = *((_BYTE *)v33 + 17);
  v144.m128i_i32[3] = v24;
  v143 = a3;
  v66 = (__int64)v23;
  v144.m128i_i8[9] = v104;
  v142 = (_BYTE *)v103;
  v144.m128i_i64[0] = v103;
  v144.m128i_i8[8] = 0;
LABEL_73:
  sub_2F9B840((__int64)&v145, v66, (__int64 *)&v143, &v144);
  v67 = _mm_loadu_si128(v146);
  *a1 = _mm_loadu_si128(&v145);
  a1[1] = v67;
  return a1;
}
