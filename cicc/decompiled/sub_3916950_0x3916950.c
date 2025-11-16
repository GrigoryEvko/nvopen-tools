// Function: sub_3916950
// Address: 0x3916950
//
void __fastcall sub_3916950(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r14
  __int64 v6; // rbx
  __int64 *v7; // r12
  __int64 *v8; // r11
  unsigned int v9; // esi
  __int64 *v10; // r14
  unsigned __int64 v11; // rdx
  int v12; // r15d
  __int64 v13; // r11
  unsigned int v14; // edi
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rbx
  unsigned int v18; // ecx
  int v19; // edx
  __int64 v20; // rdi
  __int64 *v21; // r13
  __int64 *v22; // rbx
  __int64 *v23; // r8
  __int64 v24; // rsi
  _QWORD *v25; // r15
  unsigned int v26; // r12d
  unsigned __int64 v27; // rax
  _BYTE *v28; // r12
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 *v32; // r12
  __int64 v33; // rsi
  char *v34; // r15
  unsigned __int64 v35; // rax
  void *v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rbx
  char v39; // al
  __int64 *v40; // r8
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  char v43; // dl
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // r8
  __int64 v47; // rsi
  __int64 *v48; // r15
  __int64 *v49; // rbx
  __int64 v50; // r13
  char v51; // r12
  __int64 v52; // rsi
  char *v53; // r9
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 v56; // rsi
  __int64 *v57; // rax
  __int64 v58; // r10
  __int64 v59; // rdi
  __int64 v60; // rsi
  __m128i *v61; // r12
  __m128i *v62; // r15
  unsigned __int64 v63; // rax
  __m128i *v64; // r13
  const __m128i *v65; // rdi
  __m128i *v66; // r12
  __m128i *v67; // r15
  unsigned __int64 v68; // rax
  __m128i *v69; // r13
  const __m128i *v70; // rdi
  char **v71; // rax
  int v72; // r8d
  __m128i *i; // rdi
  char *v74; // r11
  char *v75; // rcx
  char *v76; // rax
  int v77; // edx
  __int64 v78; // rsi
  int v79; // r10d
  unsigned __int64 v80; // rdx
  __int64 v81; // rax
  __int64 *v82; // r15
  __int64 *v83; // r12
  __int64 j; // r10
  unsigned int v85; // esi
  __int64 v86; // r13
  __int64 v87; // r8
  unsigned int v88; // edi
  _QWORD *v89; // rax
  __int64 v90; // rcx
  _DWORD *v91; // rdx
  _DWORD *v92; // rcx
  int v93; // esi
  int v94; // eax
  int v95; // r11d
  _QWORD *v96; // rdx
  int v97; // eax
  int v98; // ecx
  unsigned __int64 v99; // rax
  __int64 v100; // rdi
  unsigned __int64 v101; // rax
  __int64 *v102; // r9
  unsigned __int64 v103; // rdx
  __int64 v104; // rax
  __int64 v105; // rdi
  unsigned __int64 v106; // rax
  int v107; // r10d
  unsigned __int64 v108; // r9
  unsigned __int64 v109; // r8
  unsigned int v110; // r13d
  int v111; // r10d
  __int64 v112; // rsi
  int v113; // eax
  int v114; // edi
  __int64 v115; // rsi
  unsigned int v116; // eax
  __int64 v117; // r8
  int v118; // r11d
  _QWORD *v119; // r9
  int v120; // eax
  int v121; // eax
  __int64 v122; // rdi
  _QWORD *v123; // r8
  unsigned int v124; // r14d
  int v125; // r9d
  __int64 v126; // rsi
  int v127; // edx
  __int64 *v128; // r11
  int v129; // eax
  int v130; // eax
  int v131; // r13d
  unsigned __int64 v132; // r9
  __int64 v133; // [rsp+0h] [rbp-B0h]
  __int64 v137; // [rsp+28h] [rbp-88h]
  unsigned int v138; // [rsp+28h] [rbp-88h]
  unsigned int v139; // [rsp+28h] [rbp-88h]
  __int64 v140; // [rsp+28h] [rbp-88h]
  __int64 v141; // [rsp+30h] [rbp-80h]
  __int64 *v142; // [rsp+30h] [rbp-80h]
  char *v143; // [rsp+30h] [rbp-80h]
  __int64 v145; // [rsp+38h] [rbp-78h]
  __int64 v146; // [rsp+38h] [rbp-78h]
  __int64 v147; // [rsp+38h] [rbp-78h]
  __m128i v148; // [rsp+40h] [rbp-70h] BYREF
  __int64 v149; // [rsp+50h] [rbp-60h]
  char v150; // [rsp+58h] [rbp-58h] BYREF
  __int64 v151; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v152; // [rsp+68h] [rbp-48h]
  __int64 v153; // [rsp+70h] [rbp-40h]
  unsigned int v154; // [rsp+78h] [rbp-38h]

  v5 = a2;
  v6 = a1;
  v7 = (__int64 *)a2[4];
  v8 = (__int64 *)a2[5];
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  if ( v7 != v8 )
  {
    v9 = 0;
    v10 = v8;
    v11 = 0;
    v12 = 1;
    v13 = a1;
    while ( 1 )
    {
      v17 = *v7;
      if ( v9 )
      {
        v14 = (v9 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v15 = v11 + 16LL * v14;
        v16 = *(_QWORD *)v15;
        if ( v17 == *(_QWORD *)v15 )
          goto LABEL_4;
        v107 = 1;
        v108 = 0;
        while ( v16 != -8 )
        {
          if ( v16 == -16 && !v108 )
            v108 = v15;
          v14 = (v9 - 1) & (v107 + v14);
          v15 = v11 + 16LL * v14;
          v16 = *(_QWORD *)v15;
          if ( v17 == *(_QWORD *)v15 )
            goto LABEL_4;
          ++v107;
        }
        if ( v108 )
          v15 = v108;
        ++v151;
        v19 = v153 + 1;
        if ( 4 * ((int)v153 + 1) < 3 * v9 )
        {
          if ( v9 - (v19 + HIDWORD(v153)) <= v9 >> 3 )
          {
            v140 = v13;
            sub_3916790((__int64)&v151, v9);
            if ( !v154 )
            {
LABEL_210:
              LODWORD(v153) = v153 + 1;
              BUG();
            }
            v109 = 0;
            v110 = (v154 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v13 = v140;
            v111 = 1;
            v19 = v153 + 1;
            v15 = v152 + 16LL * v110;
            v112 = *(_QWORD *)v15;
            if ( v17 != *(_QWORD *)v15 )
            {
              while ( v112 != -8 )
              {
                if ( !v109 && v112 == -16 )
                  v109 = v15;
                v110 = (v154 - 1) & (v111 + v110);
                v15 = v152 + 16LL * v110;
                v112 = *(_QWORD *)v15;
                if ( v17 == *(_QWORD *)v15 )
                  goto LABEL_10;
                ++v111;
              }
              if ( v109 )
                v15 = v109;
            }
          }
          goto LABEL_10;
        }
      }
      else
      {
        ++v151;
      }
      v137 = v13;
      sub_3916790((__int64)&v151, 2 * v9);
      if ( !v154 )
        goto LABEL_210;
      v13 = v137;
      v18 = (v154 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v19 = v153 + 1;
      v15 = v152 + 16LL * v18;
      v20 = *(_QWORD *)v15;
      if ( v17 != *(_QWORD *)v15 )
      {
        v131 = 1;
        v132 = 0;
        while ( v20 != -8 )
        {
          if ( v20 == -16 && !v132 )
            v132 = v15;
          v18 = (v154 - 1) & (v131 + v18);
          v15 = v152 + 16LL * v18;
          v20 = *(_QWORD *)v15;
          if ( v17 == *(_QWORD *)v15 )
            goto LABEL_10;
          ++v131;
        }
        if ( v132 )
          v15 = v132;
      }
LABEL_10:
      LODWORD(v153) = v19;
      if ( *(_QWORD *)v15 != -8 )
        --HIDWORD(v153);
      *(_QWORD *)v15 = v17;
      *(_BYTE *)(v15 + 8) = 0;
LABEL_4:
      ++v7;
      *(_BYTE *)(v15 + 8) = v12++;
      if ( v10 == v7 )
      {
        v5 = a2;
        v6 = v13;
        break;
      }
      v11 = v152;
      v9 = v154;
    }
  }
  v21 = (__int64 *)v5[8];
  v145 = v6 + 112;
  if ( v21 != (__int64 *)v5[7] )
  {
    v141 = v6;
    v22 = (__int64 *)v5[7];
    do
    {
      v28 = (_BYTE *)*v22;
      if ( sub_390B160((__int64)v5, *v22) )
      {
        if ( (*v28 & 4) != 0 )
        {
          v23 = (__int64 *)*((_QWORD *)v28 - 1);
          v24 = *v23;
          v25 = v23 + 2;
          v26 = *v23;
        }
        else
        {
          v26 = 0;
          v24 = 0;
          v25 = 0;
        }
        v27 = sub_16D3930(v25, v24);
        sub_1680880(v145, (__int64)v25, (v27 << 32) | v26);
      }
      ++v22;
    }
    while ( v21 != v22 );
    v6 = v141;
  }
  sub_1680590(v145);
  v32 = (__int64 *)v5[7];
  if ( (__int64 *)v5[8] != v32 )
  {
    v142 = (__int64 *)v5[8];
    v133 = v6;
    while ( 1 )
    {
      v38 = *v32;
      if ( !sub_390B160((__int64)v5, *v32) )
        goto LABEL_32;
      v39 = *(_BYTE *)(v38 + 8);
      if ( (v39 & 0x10) == 0 )
      {
        if ( (*(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_32;
        if ( (*(_BYTE *)(v38 + 9) & 0xC) == 8 )
        {
          v105 = *(_QWORD *)(v38 + 24);
          *(_BYTE *)(v38 + 8) = v39 | 4;
          v106 = (unsigned __int64)sub_38CE440(v105);
          *(_QWORD *)v38 = v106 | *(_QWORD *)v38 & 7LL;
          if ( v106 )
            goto LABEL_32;
        }
      }
      v148.m128i_i64[0] = v38;
      if ( (*(_BYTE *)v38 & 4) != 0 )
      {
        v40 = *(__int64 **)(v38 - 8);
        v33 = *v40;
        v34 = (char *)(v40 + 2);
        v138 = *v40;
      }
      else
      {
        v138 = 0;
        v33 = 0;
        v34 = 0;
      }
      v35 = sub_16D3930(v34, v33);
      v148.m128i_i64[1] = sub_167FE60(v145, v34, (v35 << 32) | v138);
      v36 = (void *)(*(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v36 )
        break;
      if ( (*(_BYTE *)(v38 + 9) & 0xC) != 8
        || (*(_BYTE *)(v38 + 8) |= 4u,
            v41 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v38 + 24)),
            v42 = v41 | *(_QWORD *)v38 & 7LL,
            *(_QWORD *)v38 = v42,
            !v41) )
      {
        LOBYTE(v149) = 0;
        v37 = *(_QWORD *)(a5 + 8);
        if ( v37 == *(_QWORD *)(a5 + 16) )
        {
          sub_3915C10(a5, (_BYTE *)v37, &v148);
        }
        else
        {
          if ( v37 )
          {
            *(__m128i *)v37 = _mm_loadu_si128(&v148);
            *(_QWORD *)(v37 + 16) = v149;
            v37 = *(_QWORD *)(a5 + 8);
          }
          *(_QWORD *)(a5 + 8) = v37 + 24;
        }
        goto LABEL_32;
      }
      v36 = (void *)(v42 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v36 )
        break;
      if ( (*(_BYTE *)(v38 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v38 + 8) |= 4u;
        v80 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v38 + 24));
        v81 = v80 | *(_QWORD *)v38 & 7LL;
        *(_QWORD *)v38 = v81;
        if ( off_4CF6DB8 == (_UNKNOWN *)v80 )
          goto LABEL_88;
        v99 = v81 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v99 )
          goto LABEL_114;
        if ( (*(_BYTE *)(v38 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(v38 + 8) |= 4u;
          v99 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v38 + 24));
          *(_QWORD *)v38 = v99 | *(_QWORD *)v38 & 7LL;
          if ( v99 )
            goto LABEL_114;
        }
      }
      else if ( !off_4CF6DB8 )
      {
        goto LABEL_88;
      }
      v29 = 0;
LABEL_44:
      v43 = 0;
      if ( v154 )
      {
        v44 = (v154 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v45 = (__int64 *)(v152 + 16LL * v44);
        v46 = *v45;
        if ( v29 == *v45 )
        {
LABEL_46:
          v43 = *((_BYTE *)v45 + 8);
        }
        else
        {
          v129 = 1;
          while ( v46 != -8 )
          {
            v31 = (unsigned int)(v129 + 1);
            v44 = (v154 - 1) & (v129 + v44);
            v45 = (__int64 *)(v152 + 16LL * v44);
            v46 = *v45;
            if ( v29 == *v45 )
              goto LABEL_46;
            v129 = v31;
          }
          v43 = 0;
        }
      }
      LOBYTE(v149) = v43;
      v47 = *(_QWORD *)(a4 + 8);
      if ( v47 == *(_QWORD *)(a4 + 16) )
      {
LABEL_148:
        sub_3915C10(a4, (_BYTE *)v47, &v148);
LABEL_32:
        if ( v142 == ++v32 )
          goto LABEL_51;
      }
      else
      {
        if ( v47 )
          goto LABEL_49;
LABEL_50:
        ++v32;
        *(_QWORD *)(a4 + 8) = v47 + 24;
        if ( v142 == v32 )
        {
LABEL_51:
          v48 = (__int64 *)v5[8];
          v30 = v5[7];
          v6 = v133;
          if ( v48 == (__int64 *)v30 )
            goto LABEL_72;
          v49 = (__int64 *)v5[7];
          while ( 1 )
          {
            v50 = *v49;
            if ( !sub_390B160((__int64)v5, *v49) )
              goto LABEL_54;
            v51 = *(_BYTE *)(v50 + 8) & 0x10;
            if ( v51 )
              goto LABEL_54;
            if ( (*(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            {
              v148.m128i_i64[0] = v50;
              if ( (*(_BYTE *)v50 & 4) == 0 )
                goto LABEL_59;
            }
            else
            {
              if ( (*(_BYTE *)(v50 + 9) & 0xC) != 8 )
                goto LABEL_54;
              v100 = *(_QWORD *)(v50 + 24);
              *(_BYTE *)(v50 + 8) |= 4u;
              v101 = (unsigned __int64)sub_38CE440(v100);
              *(_QWORD *)v50 = v101 | *(_QWORD *)v50 & 7LL;
              if ( !v101 )
                goto LABEL_54;
              v148.m128i_i64[0] = v50;
              if ( (*(_BYTE *)v50 & 4) == 0 )
              {
LABEL_59:
                v139 = 0;
                v52 = 0;
                v53 = 0;
                goto LABEL_60;
              }
            }
            v102 = *(__int64 **)(v50 - 8);
            v52 = *v102;
            v53 = (char *)(v102 + 2);
            v139 = v52;
LABEL_60:
            v143 = v53;
            v54 = sub_16D3930(v53, v52);
            v148.m128i_i64[1] = sub_167FE60(v145, v143, (v54 << 32) | v139);
            v55 = *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v55 )
            {
              if ( (_UNKNOWN *)v55 == off_4CF6DB8 )
                goto LABEL_116;
            }
            else
            {
              if ( (*(_BYTE *)(v50 + 9) & 0xC) != 8 )
              {
                if ( !off_4CF6DB8 )
                  goto LABEL_116;
LABEL_63:
                v56 = 0;
                goto LABEL_64;
              }
              *(_BYTE *)(v50 + 8) |= 4u;
              v103 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v50 + 24));
              v104 = v103 | *(_QWORD *)v50 & 7LL;
              *(_QWORD *)v50 = v104;
              if ( off_4CF6DB8 == (_UNKNOWN *)v103 )
              {
LABEL_116:
                v59 = a3;
                LOBYTE(v149) = 0;
                v60 = *(_QWORD *)(a3 + 8);
                if ( v60 == *(_QWORD *)(a3 + 16) )
                  goto LABEL_147;
                if ( v60 )
                  goto LABEL_69;
                goto LABEL_70;
              }
              v55 = v104 & 0xFFFFFFFFFFFFFFF8LL;
              if ( !v55 )
              {
                if ( (*(_BYTE *)(v50 + 9) & 0xC) != 8 )
                  goto LABEL_63;
                *(_BYTE *)(v50 + 8) |= 4u;
                v55 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v50 + 24));
                *(_QWORD *)v50 = v55 | *(_QWORD *)v50 & 7LL;
                if ( !v55 )
                  goto LABEL_63;
              }
            }
            v56 = *(_QWORD *)(v55 + 24);
LABEL_64:
            if ( v154 )
            {
              v29 = (v154 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
              v57 = (__int64 *)(v152 + 16 * v29);
              v58 = *v57;
              if ( v56 == *v57 )
              {
LABEL_66:
                v51 = *((_BYTE *)v57 + 8);
              }
              else
              {
                v130 = 1;
                while ( v58 != -8 )
                {
                  v30 = (unsigned int)(v130 + 1);
                  v29 = (v154 - 1) & (v130 + (_DWORD)v29);
                  v57 = (__int64 *)(v152 + 16LL * (unsigned int)v29);
                  v58 = *v57;
                  if ( v56 == *v57 )
                    goto LABEL_66;
                  v130 = v30;
                }
              }
            }
            v59 = a3;
            LOBYTE(v149) = v51;
            v60 = *(_QWORD *)(a3 + 8);
            if ( v60 == *(_QWORD *)(a3 + 16) )
            {
LABEL_147:
              sub_3915C10(a3, (_BYTE *)v60, &v148);
LABEL_54:
              if ( v48 == ++v49 )
                goto LABEL_71;
              continue;
            }
            if ( v60 )
            {
LABEL_69:
              *(__m128i *)v60 = _mm_loadu_si128(&v148);
              *(_QWORD *)(v60 + 16) = v149;
              v60 = *(_QWORD *)(v59 + 8);
            }
LABEL_70:
            ++v49;
            *(_QWORD *)(a3 + 8) = v60 + 24;
            if ( v48 == v49 )
            {
LABEL_71:
              v6 = v133;
              goto LABEL_72;
            }
          }
        }
      }
    }
    if ( v36 == off_4CF6DB8 )
    {
LABEL_88:
      LOBYTE(v149) = 0;
      v47 = *(_QWORD *)(a4 + 8);
      if ( v47 != *(_QWORD *)(a4 + 16) )
      {
        if ( !v47 )
          goto LABEL_50;
LABEL_49:
        *(__m128i *)v47 = _mm_loadu_si128(&v148);
        *(_QWORD *)(v47 + 16) = v149;
        v47 = *(_QWORD *)(a4 + 8);
        goto LABEL_50;
      }
      goto LABEL_148;
    }
    v99 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_114:
    v29 = *(_QWORD *)(v99 + 24);
    goto LABEL_44;
  }
LABEL_72:
  v61 = *(__m128i **)(a4 + 8);
  v62 = *(__m128i **)a4;
  if ( *(__m128i **)a4 != v61 )
  {
    _BitScanReverse64(&v63, 0xAAAAAAAAAAAAAAABLL * (((char *)v61 - (char *)v62) >> 3));
    sub_3913BB0(*(_QWORD *)a4, *(__m128i **)(a4 + 8), 2LL * (int)(63 - (v63 ^ 0x3F)), v29, v30, v31);
    if ( (char *)v61 - (char *)v62 <= 384 )
    {
      sub_3913EE0(v62->m128i_i8, v61);
    }
    else
    {
      v64 = v62 + 24;
      sub_3913EE0(v62->m128i_i8, v62 + 24);
      if ( v61 != &v62[24] )
      {
        do
        {
          v65 = v64;
          v64 = (__m128i *)((char *)v64 + 24);
          sub_3913B30(v65);
        }
        while ( v61 != v64 );
      }
    }
  }
  v66 = *(__m128i **)(a5 + 8);
  v67 = *(__m128i **)a5;
  if ( *(__m128i **)a5 != v66 )
  {
    _BitScanReverse64(&v68, 0xAAAAAAAAAAAAAAABLL * (((char *)v66 - (char *)v67) >> 3));
    sub_3913BB0(*(_QWORD *)a5, *(__m128i **)(a5 + 8), 2LL * (int)(63 - (v68 ^ 0x3F)), v29, v30, v31);
    if ( (char *)v66 - (char *)v67 <= 384 )
    {
      sub_3913EE0(v67->m128i_i8, v66);
    }
    else
    {
      v69 = v67 + 24;
      sub_3913EE0(v67->m128i_i8, v67 + 24);
      if ( v66 != &v67[24] )
      {
        do
        {
          v70 = v69;
          v69 = (__m128i *)((char *)v69 + 24);
          sub_3913B30(v70);
        }
        while ( v66 != v69 );
      }
    }
  }
  v71 = (char **)a3;
  v72 = 0;
  v148.m128i_i64[1] = a4;
  v148.m128i_i64[0] = a3;
  v149 = a5;
  for ( i = &v148; ; v71 = (char **)i->m128i_i64[0] )
  {
    v74 = *v71;
    v75 = v71[1];
    if ( v75 != *v71 )
    {
      v76 = *v71;
      v77 = v72;
      do
      {
        v78 = *(_QWORD *)v76;
        v79 = v77;
        v76 += 24;
        ++v77;
        *(_DWORD *)(v78 + 16) = v79;
      }
      while ( v75 != v76 );
      v72 = v72 - 1431655765 * ((unsigned __int64)(v75 - 24 - v74) >> 3) + 1;
    }
    i = (__m128i *)((char *)i + 8);
    if ( &v150 == (char *)i )
      break;
  }
  v82 = (__int64 *)v5[5];
  v83 = (__int64 *)v5[4];
  for ( j = v6 + 16; v82 != v83; ++v83 )
  {
    v85 = *(_DWORD *)(v6 + 40);
    v86 = *v83;
    if ( v85 )
    {
      v87 = *(_QWORD *)(v6 + 24);
      v88 = (v85 - 1) & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
      v89 = (_QWORD *)(v87 + 32LL * v88);
      v90 = *v89;
      if ( v86 == *v89 )
      {
        v91 = (_DWORD *)v89[2];
        v92 = (_DWORD *)v89[1];
LABEL_97:
        while ( v91 != v92 )
        {
          if ( *(_QWORD *)v92 )
          {
            v93 = *(_DWORD *)(*(_QWORD *)v92 + 16LL);
            v94 = v92[3];
            if ( *(_DWORD *)(v6 + 248) == 1 )
              v92[3] = v93 | v94 & 0xFF000000 | 0x8000000;
            else
              v92[3] = (v93 << 8) | (unsigned __int8)v94 | 0x10;
          }
          v92 += 4;
        }
        continue;
      }
      v95 = 1;
      v96 = 0;
      while ( v90 != -8 )
      {
        if ( v90 != -16 || v96 )
          v89 = v96;
        v127 = v95 + 1;
        v88 = (v85 - 1) & (v95 + v88);
        v128 = (__int64 *)(v87 + 32LL * v88);
        v90 = *v128;
        if ( v86 == *v128 )
        {
          v91 = (_DWORD *)v128[2];
          v92 = (_DWORD *)v128[1];
          goto LABEL_97;
        }
        v95 = v127;
        v96 = v89;
        v89 = (_QWORD *)(v87 + 32LL * v88);
      }
      if ( !v96 )
        v96 = v89;
      v97 = *(_DWORD *)(v6 + 32);
      ++*(_QWORD *)(v6 + 16);
      v98 = v97 + 1;
      if ( 4 * (v97 + 1) < 3 * v85 )
      {
        if ( v85 - *(_DWORD *)(v6 + 36) - v98 <= v85 >> 3 )
        {
          v147 = j;
          sub_3915DB0(j, v85);
          v120 = *(_DWORD *)(v6 + 40);
          if ( !v120 )
          {
LABEL_209:
            ++*(_DWORD *)(v6 + 32);
            BUG();
          }
          v121 = v120 - 1;
          v122 = *(_QWORD *)(v6 + 24);
          v123 = 0;
          v124 = v121 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
          j = v147;
          v125 = 1;
          v98 = *(_DWORD *)(v6 + 32) + 1;
          v96 = (_QWORD *)(v122 + 32LL * v124);
          v126 = *v96;
          if ( v86 != *v96 )
          {
            while ( v126 != -8 )
            {
              if ( !v123 && v126 == -16 )
                v123 = v96;
              v124 = v121 & (v125 + v124);
              v96 = (_QWORD *)(v122 + 32LL * v124);
              v126 = *v96;
              if ( v86 == *v96 )
                goto LABEL_107;
              ++v125;
            }
            if ( v123 )
              v96 = v123;
          }
        }
        goto LABEL_107;
      }
    }
    else
    {
      ++*(_QWORD *)(v6 + 16);
    }
    v146 = j;
    sub_3915DB0(j, 2 * v85);
    v113 = *(_DWORD *)(v6 + 40);
    if ( !v113 )
      goto LABEL_209;
    v114 = v113 - 1;
    v115 = *(_QWORD *)(v6 + 24);
    j = v146;
    v116 = (v113 - 1) & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
    v98 = *(_DWORD *)(v6 + 32) + 1;
    v96 = (_QWORD *)(v115 + 32LL * v116);
    v117 = *v96;
    if ( v86 != *v96 )
    {
      v118 = 1;
      v119 = 0;
      while ( v117 != -8 )
      {
        if ( v117 == -16 && !v119 )
          v119 = v96;
        v116 = v114 & (v118 + v116);
        v96 = (_QWORD *)(v115 + 32LL * v116);
        v117 = *v96;
        if ( v86 == *v96 )
          goto LABEL_107;
        ++v118;
      }
      if ( v119 )
        v96 = v119;
    }
LABEL_107:
    *(_DWORD *)(v6 + 32) = v98;
    if ( *v96 != -8 )
      --*(_DWORD *)(v6 + 36);
    *v96 = v86;
    v96[1] = 0;
    v96[2] = 0;
    v96[3] = 0;
  }
  j___libc_free_0(v152);
}
