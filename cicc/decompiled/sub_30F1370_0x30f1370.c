// Function: sub_30F1370
// Address: 0x30f1370
//
void __fastcall sub_30F1370(__int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // r15
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // r8
  __int64 v6; // r9
  const char *v7; // rax
  _BYTE *v8; // rax
  _BYTE *v9; // rax
  int v10; // edx
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  int v16; // ebx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // ecx
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rcx
  unsigned __int8 *v26; // rax
  int v27; // edx
  __int64 v28; // r12
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // rax
  unsigned int v32; // ebx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rbx
  int v36; // ebx
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int8 *v39; // r14
  __m128i *v40; // r15
  int v41; // ebx
  __int64 v42; // rax
  int v43; // r14d
  __int64 i; // rax
  __int64 v45; // r12
  __int64 v46; // rax
  int v47; // ebx
  unsigned __int8 *v48; // r12
  unsigned int v49; // r13d
  int v50; // edx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdi
  const char *v57; // rdx
  int v58; // r12d
  __m128i v59; // rax
  char v60; // al
  const char **v61; // rdx
  __int64 v62; // rdx
  __int16 v63; // r12
  __int64 v64; // r9
  __int16 v65; // r12
  __int16 v66; // r12
  int v67; // edx
  __int64 v68; // rsi
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // r12
  unsigned __int64 v72; // rbx
  unsigned int v73; // r13d
  unsigned __int64 v74; // rdx
  __int64 v75; // r12
  unsigned __int8 *v76; // r13
  unsigned __int8 *v77; // rax
  __int16 v78; // r12
  __int64 v79; // r9
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // rdx
  unsigned __int64 v83; // rax
  __int64 v84; // rdi
  __int16 v85; // ax
  int v86; // r12d
  __m128i v87; // rax
  char v88; // al
  const char **v89; // rdx
  bool v90; // sf
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rbx
  int v94; // ebx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rbx
  __int64 *v98; // rbx
  int v99; // r15d
  __int64 *v100; // r13
  __int64 v101; // r12
  __int64 v102; // r8
  __int64 v103; // r9
  _BYTE *v104; // r12
  __m128i v105; // xmm1
  __m128i v106; // xmm3
  __int64 v107; // [rsp+8h] [rbp-178h]
  __int64 v108; // [rsp+10h] [rbp-170h]
  __int64 v110; // [rsp+28h] [rbp-158h]
  __int64 v111; // [rsp+38h] [rbp-148h]
  unsigned __int8 *v112; // [rsp+40h] [rbp-140h]
  __int64 v113; // [rsp+40h] [rbp-140h]
  int v114; // [rsp+40h] [rbp-140h]
  __int64 v115; // [rsp+50h] [rbp-130h]
  __int64 v116; // [rsp+60h] [rbp-120h]
  __int64 v117; // [rsp+68h] [rbp-118h]
  unsigned __int8 *v118; // [rsp+70h] [rbp-110h]
  unsigned __int8 *v119; // [rsp+70h] [rbp-110h]
  __int64 v120; // [rsp+78h] [rbp-108h]
  __int64 *v121; // [rsp+78h] [rbp-108h]
  __int64 v122; // [rsp+88h] [rbp-F8h] BYREF
  _QWORD v123[4]; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v124; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i v125; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v126; // [rsp+D0h] [rbp-B0h]
  const char *v127; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v128; // [rsp+E8h] [rbp-98h]
  __int64 v129; // [rsp+F0h] [rbp-90h]
  __int64 v130; // [rsp+F8h] [rbp-88h]
  __int64 v131; // [rsp+100h] [rbp-80h]
  __int64 v132; // [rsp+108h] [rbp-78h]
  __m128i v133; // [rsp+110h] [rbp-70h] BYREF
  __m128i v134; // [rsp+120h] [rbp-60h]
  __int64 v135; // [rsp+130h] [rbp-50h] BYREF
  __int64 v136; // [rsp+138h] [rbp-48h]

  v2 = a2;
  v3 = (__int64)a1;
  v4 = *((_QWORD *)a2 - 4);
  v133.m128i_i64[1] = 0xBFFFFFFFFFFFFFFELL;
  v134 = 0u;
  v133.m128i_i64[0] = v4;
  v135 = 0;
  v136 = 0;
  sub_30F09A0((__int64)a1, a2, v133.m128i_i64, 0, 0, 4);
  v133.m128i_i64[1] = (__int64)&v135;
  v133.m128i_i64[0] = 0;
  v134.m128i_i64[0] = 4;
  v134.m128i_i32[2] = 0;
  v134.m128i_i8[12] = 1;
  v120 = sub_30EFD90(a1, v4, 0, (__int64)&v133, v5, v6);
  if ( v134.m128i_i8[12] )
  {
    if ( *(_BYTE *)v120 )
      goto LABEL_36;
  }
  else
  {
    _libc_free(v133.m128i_u64[1]);
    if ( *(_BYTE *)v120 )
      goto LABEL_36;
  }
  if ( ((*(_WORD *)(v120 + 2) >> 4) & 0x3FF) != ((*((_WORD *)a2 + 1) >> 2) & 0x3FF) )
  {
    BYTE1(v135) = 1;
    v7 = "Undefined behavior: Caller and callee calling convention differ";
    goto LABEL_5;
  }
  v10 = *a2;
  v11 = *(_QWORD *)(v120 + 24);
  if ( v10 == 40 )
  {
    v12 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v12 = 0;
    if ( v10 != 85 )
    {
      v12 = 64;
      if ( v10 != 34 )
LABEL_183:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_23;
  v13 = sub_BD2BC0((__int64)a2);
  v15 = v13 + v14;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v15 >> 4) )
LABEL_182:
      BUG();
LABEL_23:
    v19 = 0;
    goto LABEL_24;
  }
  if ( !(unsigned int)((v15 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_23;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_182;
  v16 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v17 = sub_BD2BC0((__int64)a2);
  v19 = 32LL * (unsigned int)(*(_DWORD *)(v17 + v18 - 4) - v16);
LABEL_24:
  v20 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  v21 = (v20 - 32 - v12 - v19) >> 5;
  v22 = *(_DWORD *)(v11 + 12) - 1;
  v23 = *(_DWORD *)(v11 + 8) >> 8;
  if ( !(_DWORD)v23 )
  {
    if ( v22 == (_DWORD)v21 )
      goto LABEL_26;
LABEL_51:
    BYTE1(v135) = 1;
    v7 = "Undefined behavior: Call argument count mismatches callee argument count";
    goto LABEL_5;
  }
  if ( v22 > (unsigned int)v21 )
    goto LABEL_51;
LABEL_26:
  v24 = *((_QWORD *)v2 + 1);
  if ( **(_QWORD **)(v11 + 16) != v24 )
  {
    BYTE1(v135) = 1;
    v7 = "Undefined behavior: Call return type mismatches callee return type";
    goto LABEL_5;
  }
  if ( (*(_BYTE *)(v120 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v120, v23, v20, v24);
    v25 = *(_QWORD *)(v120 + 96);
    v116 = v25;
    if ( (*(_BYTE *)(v120 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v120, v23, v62, v25);
      v25 = *(_QWORD *)(v120 + 96);
    }
    v20 = 32LL * (*((_DWORD *)v2 + 1) & 0x7FFFFFF);
  }
  else
  {
    v116 = *(_QWORD *)(v120 + 96);
    v25 = v116;
  }
  v110 = v25 + 40LL * *(_QWORD *)(v120 + 104);
  v26 = &v2[-v20];
  v27 = *v2;
  v117 = (__int64)v26;
  if ( v27 == 40 )
  {
    v28 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v2);
  }
  else
  {
    v28 = -32;
    if ( v27 != 85 )
    {
      v28 = -96;
      if ( v27 != 34 )
        goto LABEL_183;
    }
  }
  if ( (v2[7] & 0x80u) == 0 )
    goto LABEL_59;
  v33 = sub_BD2BC0((__int64)v2);
  v35 = v33 + v34;
  if ( (v2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v35 >> 4) )
      goto LABEL_185;
  }
  else if ( (unsigned int)((v35 - sub_BD2BC0((__int64)v2)) >> 4) )
  {
    if ( (v2[7] & 0x80u) != 0 )
    {
      v36 = *(_DWORD *)(sub_BD2BC0((__int64)v2) + 8);
      if ( (v2[7] & 0x80u) == 0 )
        BUG();
      v37 = sub_BD2BC0((__int64)v2);
      v28 -= 32LL * (unsigned int)(*(_DWORD *)(v37 + v38 - 4) - v36);
      goto LABEL_59;
    }
LABEL_185:
    BUG();
  }
LABEL_59:
  v118 = &v2[v28];
  if ( (unsigned __int8 *)v117 == &v2[v28] )
    goto LABEL_36;
  v39 = v2;
  do
  {
    if ( v110 == v116 )
      goto LABEL_74;
    v111 = *(_QWORD *)v117;
    if ( *(_QWORD *)(v116 + 8) != *(_QWORD *)(*(_QWORD *)v117 + 8LL) )
    {
      v2 = v39;
      BYTE1(v135) = 1;
      v3 = (__int64)a1;
      v7 = "Undefined behavior: Call argument type mismatches callee parameter type";
      goto LABEL_5;
    }
    if ( (unsigned __int8)sub_B2D700(v116) )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v111 + 8) + 8LL) == 14 )
      {
        v47 = 0;
        v124.m128i_i64[0] = *((_QWORD *)v39 + 9);
        v48 = &v39[-32 * (*((_DWORD *)v39 + 1) & 0x7FFFFFF)];
        if ( v48 != v118 )
        {
          while ( 1 )
          {
            v49 = v47++;
            if ( (unsigned __int8)sub_A74710(&v124, v47, 81) )
              goto LABEL_98;
            if ( (unsigned __int8)sub_B2BD80(v116) )
              break;
LABEL_93:
            if ( sub_CF49B0(v39, v49, 50) != 1
              && v48 != (unsigned __int8 *)v117
              && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v48 + 8LL) + 8LL) == 14
              && **(_BYTE **)v48 != 20 )
            {
              v133.m128i_i64[0] = *(_QWORD *)v48;
              v133.m128i_i64[1] = -1;
              v56 = a1[3];
              v134 = 0u;
              v57 = *(const char **)v117;
              v135 = 0;
              v136 = 0;
              v127 = v57;
              v128 = -1;
              v129 = 0;
              v130 = 0;
              v131 = 0;
              v132 = 0;
              if ( (unsigned __int8)(sub_CF4E00(v56, (__int64)&v127, (__int64)&v133) - 2) <= 1u )
              {
                v2 = v39;
                BYTE1(v135) = 1;
                v3 = (__int64)a1;
                v7 = "Unusual: noalias argument aliases another argument";
                goto LABEL_5;
              }
            }
LABEL_98:
            v48 += 32;
            if ( v48 == v118 )
              goto LABEL_65;
          }
          v50 = *v39;
          if ( v50 == 40 )
          {
            v115 = 32LL * (unsigned int)sub_B491D0((__int64)v39);
          }
          else
          {
            v115 = 0;
            if ( v50 != 85 )
            {
              if ( v50 != 34 )
                goto LABEL_183;
              v115 = 64;
            }
          }
          if ( (v39[7] & 0x80u) != 0 )
          {
            v51 = sub_BD2BC0((__int64)v39);
            v113 = v52 + v51;
            if ( (v39[7] & 0x80u) == 0 )
            {
              if ( (unsigned int)(v113 >> 4) )
LABEL_189:
                BUG();
            }
            else if ( (unsigned int)((v113 - sub_BD2BC0((__int64)v39)) >> 4) )
            {
              if ( (v39[7] & 0x80u) == 0 )
                goto LABEL_189;
              v114 = *(_DWORD *)(sub_BD2BC0((__int64)v39) + 8);
              if ( (v39[7] & 0x80u) == 0 )
                BUG();
              v53 = sub_BD2BC0((__int64)v39);
              v55 = 32LL * (unsigned int)(*(_DWORD *)(v53 + v54 - 4) - v114);
LABEL_89:
              if ( v49 < (unsigned int)((32LL * (*((_DWORD *)v39 + 1) & 0x7FFFFFF) - 32 - v115 - v55) >> 5)
                && (unsigned __int8)sub_B49B80((__int64)v39, v49, 81)
                || sub_CF49B0(v39, v49, 51)
                || sub_CF49B0(v39, v49, 50) )
              {
                goto LABEL_98;
              }
              goto LABEL_93;
            }
          }
          v55 = 0;
          goto LABEL_89;
        }
      }
    }
LABEL_65:
    if ( (unsigned __int8)sub_B2D720(v116) && *(_BYTE *)(*(_QWORD *)(v111 + 8) + 8LL) == 14 )
    {
      v80 = sub_B2BD30(v116);
      v81 = sub_9208B0(a1[2], v80);
      v133.m128i_i64[1] = v82;
      v83 = (unsigned __int64)(v81 + 7) >> 3;
      if ( (_BYTE)v82 )
        v83 |= 0x4000000000000000uLL;
      v133.m128i_i64[1] = v83;
      v134 = 0u;
      v84 = a1[2];
      v133.m128i_i64[0] = v111;
      v135 = 0;
      v136 = 0;
      LOBYTE(v85) = sub_AE5020(v84, v80);
      HIBYTE(v85) = 1;
      sub_30F09A0((__int64)a1, v39, v133.m128i_i64, v85, v80, 3);
    }
    v40 = (__m128i *)((char *)v123 + 4);
    v112 = v39;
    v41 = sub_BD2910(v117);
    v123[0] = 0x360000004FLL;
    v123[1] = 0x510000000FLL;
    v123[2] = 0x5300000050LL;
    v123[3] = 0x5500000054LL;
    v42 = *((_QWORD *)v39 + 9);
    v43 = 79;
    v122 = v42;
    for ( i = sub_A747F0(&v122, v41 + 1, 79); ; i = sub_A747F0(&v122, v41 + 1, v43) )
    {
      v45 = i;
      v46 = sub_B2D8D0(v120, v41, v43);
      if ( (v46 != 0) != (v45 != 0) )
      {
        v58 = v43;
        v3 = (__int64)a1;
        v127 = " not present on both function and call-site";
        v2 = v112;
        LOWORD(v131) = 259;
        v59.m128i_i64[0] = (__int64)sub_A6FBB0(v58);
        v125 = v59;
        v60 = v131;
        v124.m128i_i64[0] = (__int64)"Undefined behavior: ABI attribute ";
        LOWORD(v126) = 1283;
        if ( !(_BYTE)v131 )
          goto LABEL_110;
        if ( (_BYTE)v131 == 1 )
        {
          v105 = _mm_loadu_si128(&v125);
          v133 = _mm_loadu_si128(&v124);
          v135 = v126;
          v134 = v105;
        }
        else
        {
          if ( BYTE1(v131) == 1 )
          {
            v61 = (const char **)v127;
            v108 = v128;
          }
          else
          {
            v61 = &v127;
            v60 = 2;
          }
          v134.m128i_i64[0] = (__int64)v61;
          LOBYTE(v135) = 2;
          v133.m128i_i64[0] = (__int64)&v124;
          BYTE1(v135) = v60;
          v134.m128i_i64[1] = v108;
        }
LABEL_6:
        sub_CA0E80((__int64)&v133, v3 + 88);
        v8 = *(_BYTE **)(v3 + 120);
        if ( (unsigned __int64)v8 >= *(_QWORD *)(v3 + 112) )
        {
          sub_CB5D20(v3 + 88, 10);
        }
        else
        {
          *(_QWORD *)(v3 + 120) = v8 + 1;
          *v8 = 10;
        }
        if ( *v2 <= 0x1Cu )
        {
          sub_A5BF40(v2, v3 + 88, 1, *(_QWORD *)v3);
          v9 = *(_BYTE **)(v3 + 120);
          if ( (unsigned __int64)v9 < *(_QWORD *)(v3 + 112) )
            goto LABEL_10;
        }
        else
        {
          sub_A69870((__int64)v2, (_BYTE *)(v3 + 88), 0);
          v9 = *(_BYTE **)(v3 + 120);
          if ( (unsigned __int64)v9 < *(_QWORD *)(v3 + 112) )
          {
LABEL_10:
            *(_QWORD *)(v3 + 120) = v9 + 1;
            *v9 = 10;
            return;
          }
        }
        sub_CB5D20(v3 + 88, 10);
        return;
      }
      if ( v45 != v46 && v46 )
      {
        v86 = v43;
        v3 = (__int64)a1;
        v127 = " does not have same argument for function and call-site";
        v2 = v112;
        LOWORD(v131) = 259;
        v87.m128i_i64[0] = (__int64)sub_A6FBB0(v86);
        v125 = v87;
        v88 = v131;
        v124.m128i_i64[0] = (__int64)"Undefined behavior: ABI attribute ";
        LOWORD(v126) = 1283;
        if ( (_BYTE)v131 )
        {
          if ( (_BYTE)v131 == 1 )
          {
            v106 = _mm_loadu_si128(&v125);
            v133 = _mm_loadu_si128(&v124);
            v135 = v126;
            v134 = v106;
          }
          else
          {
            if ( BYTE1(v131) == 1 )
            {
              v89 = (const char **)v127;
              v107 = v128;
            }
            else
            {
              v89 = &v127;
              v88 = 2;
            }
            v134.m128i_i64[0] = (__int64)v89;
            LOBYTE(v135) = 2;
            v133.m128i_i64[0] = (__int64)&v124;
            BYTE1(v135) = v88;
            v134.m128i_i64[1] = v107;
          }
          goto LABEL_6;
        }
LABEL_110:
        LOWORD(v135) = 256;
        goto LABEL_6;
      }
      if ( &v124 == v40 )
        break;
      v43 = v40->m128i_i32[0];
      v40 = (__m128i *)((char *)v40 + 4);
    }
    v116 += 40;
    v39 = v112;
LABEL_74:
    v117 += 32;
  }
  while ( (unsigned __int8 *)v117 != v118 );
  v2 = v39;
  v3 = (__int64)a1;
LABEL_36:
  if ( *v2 != 85 )
    return;
  if ( (*((_WORD *)v2 + 1) & 3u) - 1 <= 1 )
  {
    v90 = (v2[7] & 0x80u) != 0;
    v127 = (const char *)*((_QWORD *)v2 + 9);
    if ( v90 )
    {
      v91 = sub_BD2BC0((__int64)v2);
      v93 = v91 + v92;
      if ( (v2[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v93 >> 4) )
          goto LABEL_181;
      }
      else if ( (unsigned int)((v93 - sub_BD2BC0((__int64)v2)) >> 4) )
      {
        if ( (v2[7] & 0x80u) != 0 )
        {
          v94 = *(_DWORD *)(sub_BD2BC0((__int64)v2) + 8);
          if ( (v2[7] & 0x80u) == 0 )
            BUG();
          v95 = sub_BD2BC0((__int64)v2);
          v97 = -32 - 32LL * (unsigned int)(*(_DWORD *)(v95 + v96 - 4) - v94);
          goto LABEL_151;
        }
LABEL_181:
        BUG();
      }
    }
    v97 = -32;
LABEL_151:
    v121 = (__int64 *)&v2[v97];
    v98 = (__int64 *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
    if ( v98 == v121 )
    {
LABEL_159:
      if ( *v2 == 85 )
        goto LABEL_38;
      return;
    }
    v119 = v2;
    v99 = 0;
    v100 = v98;
    while ( 1 )
    {
      ++v99;
      v101 = *v100;
      if ( !(unsigned __int8)sub_A74710(&v127, v99, 81) )
      {
        v134.m128i_i8[12] = 1;
        v133.m128i_i64[0] = 0;
        v134.m128i_i64[0] = 4;
        v133.m128i_i64[1] = (__int64)&v135;
        v134.m128i_i32[2] = 0;
        v104 = (_BYTE *)sub_30EFD90((__int64 *)v3, v101, 1, (__int64)&v133, v102, v103);
        if ( !v134.m128i_i8[12] )
          _libc_free(v133.m128i_u64[1]);
        if ( *v104 == 60 )
          break;
      }
      v100 += 4;
      if ( v121 == v100 )
      {
        v2 = v119;
        goto LABEL_159;
      }
    }
    BYTE1(v135) = 1;
    v2 = v119;
    v7 = "Undefined behavior: Call with \"tail\" keyword references alloca";
LABEL_5:
    v133.m128i_i64[0] = (__int64)v7;
    LOBYTE(v135) = 3;
    goto LABEL_6;
  }
LABEL_38:
  v29 = *((_QWORD *)v2 - 4);
  if ( !v29 || *(_BYTE *)v29 || *(_QWORD *)(v29 + 24) != *((_QWORD *)v2 + 10) || (*(_BYTE *)(v29 + 33) & 0x20) == 0 )
    return;
  v30 = *(_DWORD *)(v29 + 36);
  if ( v30 > 0xF5 )
  {
    if ( v30 != 374 )
    {
      if ( v30 <= 0x176 )
      {
        if ( v30 != 342 )
        {
          if ( v30 != 373 )
            return;
          sub_D669C0(&v133, (__int64)v2, 0, *(__int64 **)(v3 + 48));
          sub_30F09A0(v3, v2, v133.m128i_i64, 0, 0, 2);
          sub_D669C0(&v133, (__int64)v2, 1u, *(__int64 **)(v3 + 48));
          v79 = 1;
LABEL_133:
          sub_30F09A0(v3, v2, v133.m128i_i64, 0, 0, v79);
          return;
        }
      }
      else if ( v30 != 375 )
      {
        return;
      }
    }
    sub_D669C0(&v133, (__int64)v2, 0, *(__int64 **)(v3 + 48));
    v79 = 3;
    goto LABEL_133;
  }
  if ( v30 > 0xED )
  {
    switch ( v30 )
    {
      case 0xEEu:
      case 0xF0u:
        v65 = sub_A74840((_QWORD *)v2 + 9, 0);
        sub_D67210(&v133, (__int64)v2);
        sub_30F09A0(v3, v2, v133.m128i_i64, v65, 0, 2);
        v66 = sub_A74840((_QWORD *)v2 + 9, 1);
        sub_D671D0(&v133, (__int64)v2);
        sub_30F09A0(v3, v2, v133.m128i_i64, v66, 0, 1);
        v67 = *((_DWORD *)v2 + 1);
        v133.m128i_i64[0] = 0;
        v134.m128i_i64[0] = 4;
        v134.m128i_i32[2] = 0;
        v134.m128i_i8[12] = 1;
        v68 = *(_QWORD *)&v2[32 * (2LL - (v67 & 0x7FFFFFF))];
        v133.m128i_i64[1] = (__int64)&v135;
        v71 = sub_30EFD90((__int64 *)v3, v68, 0, (__int64)&v133, v69, v70);
        if ( !v134.m128i_i8[12] )
          _libc_free(v133.m128i_u64[1]);
        v72 = 0xBFFFFFFFFFFFFFFELL;
        if ( *(_BYTE *)v71 != 17 )
          goto LABEL_126;
        v73 = *(_DWORD *)(v71 + 32);
        if ( v73 > 0x40 )
        {
          if ( v73 - (unsigned int)sub_C444A0(v71 + 24) > 0x20 )
            goto LABEL_126;
          v72 = **(_QWORD **)(v71 + 24);
        }
        else
        {
          v72 = *(_QWORD *)(v71 + 24);
          if ( !v72 )
            goto LABEL_126;
          _BitScanReverse64(&v74, v72);
          if ( 64 - ((unsigned int)v74 ^ 0x3F) > 0x20 )
          {
            v72 = 0xBFFFFFFFFFFFFFFELL;
            goto LABEL_126;
          }
        }
        if ( v72 > 0x3FFFFFFFFFFFFFFBLL )
          v72 = 0xBFFFFFFFFFFFFFFELL;
LABEL_126:
        v75 = *(_QWORD *)(v3 + 24);
        v76 = sub_BD3990(*(unsigned __int8 **)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)], v68);
        v77 = sub_BD3990(*(unsigned __int8 **)&v2[32 * (1LL - (*((_DWORD *)v2 + 1) & 0x7FFFFFF))], v68);
        v133.m128i_i64[0] = (__int64)v76;
        v133.m128i_i64[1] = v72;
        v134 = 0u;
        v135 = 0;
        v136 = 0;
        v127 = (const char *)v77;
        v128 = v72;
        v129 = 0;
        v130 = 0;
        v131 = 0;
        v132 = 0;
        if ( (unsigned __int8)sub_CF4E00(v75, (__int64)&v127, (__int64)&v133) == 3 )
        {
          BYTE1(v135) = 1;
          v7 = "Undefined behavior: memcpy source and destination overlap";
          goto LABEL_5;
        }
        break;
      case 0xF1u:
        v78 = sub_A74840((_QWORD *)v2 + 9, 0);
        sub_D67210(&v133, (__int64)v2);
        sub_30F09A0(v3, v2, v133.m128i_i64, v78, 0, 2);
        v63 = sub_A74840((_QWORD *)v2 + 9, 1);
        sub_D671D0(&v133, (__int64)v2);
        v64 = 1;
        goto LABEL_118;
      case 0xF3u:
      case 0xF5u:
        v63 = sub_A74840((_QWORD *)v2 + 9, 0);
        sub_D67210(&v133, (__int64)v2);
        v64 = 2;
LABEL_118:
        sub_30F09A0(v3, v2, v133.m128i_i64, v63, 0, v64);
        return;
      default:
        return;
    }
    return;
  }
  if ( v30 != 185 )
    return;
  v31 = *(_QWORD *)&v2[32 * (1LL - (*((_DWORD *)v2 + 1) & 0x7FFFFFF))];
  if ( *(_BYTE *)v31 != 17 )
    return;
  v32 = *(_DWORD *)(v31 + 32);
  if ( v32 <= 0x40 )
  {
    if ( *(_QWORD *)(v31 + 24) )
      return;
LABEL_48:
    BYTE1(v135) = 1;
    v7 = "get_active_lane_mask: operand #2 must be greater than 0";
    goto LABEL_5;
  }
  if ( v32 == (unsigned int)sub_C444A0(v31 + 24) )
    goto LABEL_48;
}
