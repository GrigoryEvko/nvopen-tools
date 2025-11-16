// Function: sub_28706C0
// Address: 0x28706c0
//
__int64 __fastcall sub_28706C0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r9
  __int64 v3; // rdi
  unsigned __int64 v4; // rcx
  __int64 v5; // rsi
  char v6; // r15
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r9
  int v12; // r11d
  __int64 *v13; // rdx
  __int64 i; // r8
  __int64 *v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // rdi
  int v18; // edi
  __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // rsi
  __int64 *v23; // r13
  unsigned __int64 v24; // r15
  __int64 v25; // rcx
  __int64 v26; // rbx
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  __int64 v31; // r12
  __int64 *v32; // r14
  __int64 v33; // rdi
  int v34; // eax
  unsigned int v35; // r14d
  unsigned __int64 v36; // r15
  __int64 *v37; // r13
  __int64 v38; // rbx
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // rsi
  unsigned __int64 v42; // rdi
  __int64 v43; // r12
  __int64 *v44; // r14
  __int64 v45; // rdi
  int v46; // eax
  unsigned __int64 v47; // rsi
  char v48; // al
  int v49; // edi
  int v50; // edi
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // r8
  __int64 v55; // r9
  int v56; // eax
  unsigned __int64 v57; // rcx
  __int64 v58; // rsi
  unsigned int v59; // eax
  unsigned int v60; // eax
  unsigned int v61; // ecx
  _QWORD *v62; // rdi
  __int64 v63; // r8
  unsigned int v64; // eax
  int v65; // eax
  unsigned __int64 v66; // rax
  __int64 v67; // rax
  unsigned int v68; // ebx
  __int64 v69; // r12
  _QWORD *v70; // rax
  _QWORD *m; // rdx
  int v72; // edi
  __int64 *v73; // rcx
  unsigned int k; // eax
  __int64 v75; // r8
  unsigned int v76; // eax
  int v77; // r8d
  int v78; // esi
  unsigned int j; // ebx
  __int64 *v80; // rax
  __int64 v81; // r8
  unsigned int v82; // ebx
  _QWORD *v83; // r8
  __int64 v84; // [rsp+8h] [rbp-258h]
  __int64 v85; // [rsp+10h] [rbp-250h]
  __int64 v86; // [rsp+18h] [rbp-248h]
  __int64 v87; // [rsp+20h] [rbp-240h]
  __int64 v89; // [rsp+30h] [rbp-230h]
  unsigned __int64 v90; // [rsp+38h] [rbp-228h]
  __int64 *v91; // [rsp+40h] [rbp-220h]
  __int64 v92; // [rsp+48h] [rbp-218h]
  __int64 v93; // [rsp+50h] [rbp-210h]
  __int64 v94; // [rsp+58h] [rbp-208h]
  unsigned int v95; // [rsp+60h] [rbp-200h]
  __int64 v96; // [rsp+68h] [rbp-1F8h]
  __int64 *v97; // [rsp+70h] [rbp-1F0h]
  unsigned int v98; // [rsp+70h] [rbp-1F0h]
  unsigned int v99; // [rsp+7Ch] [rbp-1E4h]
  unsigned int v100; // [rsp+7Ch] [rbp-1E4h]
  __int64 *v101; // [rsp+80h] [rbp-1E0h]
  __int64 v102; // [rsp+88h] [rbp-1D8h]
  __int64 v103; // [rsp+90h] [rbp-1D0h]
  __int64 v104; // [rsp+A0h] [rbp-1C0h] BYREF
  _QWORD *v105; // [rsp+A8h] [rbp-1B8h]
  __int64 v106; // [rsp+B0h] [rbp-1B0h]
  unsigned int v107; // [rsp+B8h] [rbp-1A8h]
  __int64 v108; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 v109; // [rsp+C8h] [rbp-198h]
  __int64 v110; // [rsp+D0h] [rbp-190h]
  __int64 v111; // [rsp+D8h] [rbp-188h]
  _QWORD v112[2]; // [rsp+E0h] [rbp-180h] BYREF
  unsigned __int64 v113; // [rsp+F0h] [rbp-170h]
  _QWORD v114[4]; // [rsp+F8h] [rbp-168h] BYREF
  int v115; // [rsp+118h] [rbp-148h]
  __int64 v116; // [rsp+120h] [rbp-140h] BYREF
  unsigned __int128 v117; // [rsp+128h] [rbp-138h]
  __int64 v118; // [rsp+138h] [rbp-128h]
  __int64 v119; // [rsp+140h] [rbp-120h]
  char *v120; // [rsp+148h] [rbp-118h] BYREF
  __int64 v121; // [rsp+150h] [rbp-110h]
  _DWORD v122[8]; // [rsp+158h] [rbp-108h] BYREF
  __int64 v123; // [rsp+178h] [rbp-E8h]
  __m128i v124; // [rsp+180h] [rbp-E0h]
  __int64 v125; // [rsp+190h] [rbp-D0h] BYREF
  void *s; // [rsp+198h] [rbp-C8h]
  _BYTE v127[12]; // [rsp+1A0h] [rbp-C0h]
  char v128; // [rsp+1ACh] [rbp-B4h]
  char v129; // [rsp+1B0h] [rbp-B0h] BYREF

  result = *(unsigned int *)(a1 + 1328);
  v2 = *(_QWORD *)(a1 + 1320);
  v86 = result;
  v3 = v2 + 2184 * result;
  if ( v2 == v3 )
  {
    if ( (unsigned int)qword_5001308 > 1 )
      return result;
  }
  else
  {
    result = v2;
    v4 = 1;
    while ( 1 )
    {
      v5 = *(unsigned int *)(result + 768);
      if ( (unsigned int)v5 >= (unsigned int)qword_5001308 )
        break;
      v4 *= v5;
      if ( (unsigned int)qword_5001308 <= v4 )
        break;
      result += 2184;
      if ( v3 == result )
        return result;
    }
  }
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v125 = 0;
  s = &v129;
  *(_QWORD *)v127 = 16;
  *(_DWORD *)&v127[8] = 0;
  v128 = 1;
  if ( !v86 )
  {
    v17 = 0;
    goto LABEL_22;
  }
  v84 = 0;
  v85 = 0;
  while ( 2 )
  {
    v6 = 0;
    v96 = 0;
    v93 = v84 + v2;
    v92 = *(unsigned int *)(v84 + v2 + 768);
    if ( !*(_DWORD *)(v84 + v2 + 768) )
      goto LABEL_33;
    do
    {
      v7 = *(_QWORD *)(v93 + 760);
      v8 = v7 + 112 * v96;
      v9 = *(_QWORD *)(v8 + 88);
      v94 = v8;
      if ( !v9 )
        goto LABEL_29;
      v10 = *(_QWORD *)(v8 + 32);
      if ( !v107 )
      {
        ++v104;
        goto LABEL_115;
      }
      v11 = v107 - 1;
      v12 = 1;
      v13 = 0;
      for ( i = ((unsigned int)((0xBF58476D1CE4E5B9LL
                               * ((unsigned int)(37 * v10)
                                | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)
               ^ (756364221 * (_DWORD)v10))
              & (v107 - 1); ; i = (unsigned int)v11 & v77 )
      {
        v15 = &v105[3 * (unsigned int)i];
        v16 = *v15;
        if ( v9 == *v15 && v10 == v15[1] )
          break;
        if ( v16 == -4096 )
        {
          if ( v15[1] == 0x7FFFFFFFFFFFFFFFLL )
          {
            if ( !v13 )
              v13 = &v105[3 * (unsigned int)i];
            ++v104;
            v18 = v106 + 1;
            if ( 4 * ((int)v106 + 1) < 3 * v107 )
            {
              if ( v107 - HIDWORD(v106) - v18 > v107 >> 3 )
                goto LABEL_26;
              sub_28703E0((__int64)&v104, v107);
              if ( v107 )
              {
                v78 = 1;
                v13 = 0;
                for ( j = (v107 - 1)
                        & (((0xBF58476D1CE4E5B9LL
                           * ((unsigned int)(37 * v10)
                            | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)
                         ^ (756364221 * v10)); ; j = (v107 - 1) & v82 )
                {
                  v80 = &v105[3 * j];
                  v81 = *v80;
                  if ( v9 == *v80 && v10 == v80[1] )
                  {
                    v13 = &v105[3 * j];
                    v18 = v106 + 1;
                    goto LABEL_26;
                  }
                  if ( v81 == -4096 )
                  {
                    if ( v80[1] == 0x7FFFFFFFFFFFFFFFLL )
                    {
                      if ( !v13 )
                        v13 = &v105[3 * j];
                      v18 = v106 + 1;
                      goto LABEL_26;
                    }
                  }
                  else if ( v81 == -8192 && v80[1] == 0x7FFFFFFFFFFFFFFELL && !v13 )
                  {
                    v13 = &v105[3 * j];
                  }
                  v82 = v78 + j;
                  ++v78;
                }
              }
              goto LABEL_163;
            }
LABEL_115:
            sub_28703E0((__int64)&v104, 2 * v107);
            if ( v107 )
            {
              v72 = 1;
              v73 = 0;
              for ( k = (v107 - 1)
                      & (((0xBF58476D1CE4E5B9LL
                         * ((unsigned int)(37 * v10)
                          | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)
                       ^ (756364221 * v10)); ; k = (v107 - 1) & v76 )
              {
                v13 = &v105[3 * k];
                v75 = *v13;
                if ( v9 == *v13 && v10 == v13[1] )
                {
                  v18 = v106 + 1;
                  goto LABEL_26;
                }
                if ( v75 == -4096 )
                {
                  if ( v13[1] == 0x7FFFFFFFFFFFFFFFLL )
                  {
                    if ( v73 )
                      v13 = v73;
                    v18 = v106 + 1;
LABEL_26:
                    LODWORD(v106) = v18;
                    if ( *v13 != -4096 || v13[1] != 0x7FFFFFFFFFFFFFFFLL )
                      --HIDWORD(v106);
                    *v13 = v9;
                    v13[1] = v10;
                    v13[2] = v96;
LABEL_29:
                    ++v96;
                    goto LABEL_30;
                  }
                }
                else if ( v75 == -8192 && v13[1] == 0x7FFFFFFFFFFFFFFELL && !v73 )
                {
                  v73 = &v105[3 * k];
                }
                v76 = v72 + k;
                ++v72;
              }
            }
LABEL_163:
            LODWORD(v106) = v106 + 1;
            BUG();
          }
        }
        else if ( v16 == -8192 && v15[1] == 0x7FFFFFFFFFFFFFFELL && !v13 )
        {
          v13 = &v105[3 * (unsigned int)i];
        }
        v77 = v12 + i;
        ++v12;
      }
      v22 = v7 + 112 * v15[2];
      v87 = v22;
      v23 = *(__int64 **)(v94 + 40);
      v91 = *(__int64 **)(v22 + 40);
      v97 = &v23[*(unsigned int *)(v94 + 48)];
      v101 = &v91[*(unsigned int *)(v22 + 48)];
      if ( v23 == v97 )
      {
        if ( v101 == v91 )
          goto LABEL_85;
        v90 = 0;
        v35 = *(_DWORD *)(a1 + 36304);
        v102 = *(_QWORD *)(a1 + 36288);
        v103 = v86 + 1;
        v89 = v35;
LABEL_57:
        v100 = v35;
        v36 = 0;
        v37 = v91;
        v98 = v35 - 1;
        while ( 1 )
        {
          v25 = *v37;
          if ( !v100 )
            goto LABEL_69;
          v39 = v98 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v40 = (__int64 *)(v102 + 16LL * v39);
          v41 = *v40;
          if ( v25 != *v40 )
            break;
LABEL_62:
          v42 = v40[1];
          if ( (v42 & 1) != 0 )
          {
            v38 = (int)sub_39FAC40(~(-1LL << (v42 >> 58)) & (v42 >> 1));
          }
          else
          {
            v43 = *(_QWORD *)v42 + 8LL * *(unsigned int *)(v42 + 8);
            if ( *(_QWORD *)v42 == v43 )
            {
              v38 = 0;
            }
            else
            {
              v44 = *(__int64 **)v42;
              LODWORD(v38) = 0;
              do
              {
                v45 = *v44++;
                v38 = (unsigned int)sub_39FAC40(v45) + (unsigned int)v38;
              }
              while ( v44 != (__int64 *)v43 );
            }
          }
          ++v37;
          v36 += v103 - v38;
          if ( v101 == v37 )
          {
            v47 = v36;
            goto LABEL_71;
          }
        }
        v46 = 1;
        while ( v41 != -4096 )
        {
          v39 = v98 & (v46 + v39);
          v49 = v46 + 1;
          v40 = (__int64 *)(v102 + 16LL * v39);
          v41 = *v40;
          if ( v25 == *v40 )
            goto LABEL_62;
          v46 = v49;
        }
LABEL_69:
        v40 = (__int64 *)(v102 + 16 * v89);
        goto LABEL_62;
      }
      v24 = 0;
      v102 = *(_QWORD *)(a1 + 36288);
      v99 = *(_DWORD *)(a1 + 36304);
      v103 = v86 + 1;
      v89 = v99;
      v95 = v99 - 1;
      do
      {
        v25 = *v23;
        if ( v99 )
        {
          v27 = v95 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v28 = (__int64 *)(v102 + 16LL * v27);
          v29 = *v28;
          if ( v25 == *v28 )
            goto LABEL_48;
          v34 = 1;
          while ( v29 != -4096 )
          {
            v27 = v95 & (v34 + v27);
            v50 = v34 + 1;
            v28 = (__int64 *)(v102 + 16LL * v27);
            v29 = *v28;
            if ( v25 == *v28 )
              goto LABEL_48;
            v34 = v50;
          }
        }
        v28 = (__int64 *)(v102 + 16LL * v99);
LABEL_48:
        v30 = v28[1];
        if ( (v30 & 1) != 0 )
        {
          v26 = (int)sub_39FAC40(~(-1LL << (v30 >> 58)) & (v30 >> 1));
        }
        else
        {
          v31 = *(_QWORD *)v30 + 8LL * *(unsigned int *)(v30 + 8);
          if ( *(_QWORD *)v30 == v31 )
          {
            v26 = 0;
          }
          else
          {
            v32 = *(__int64 **)v30;
            LODWORD(v26) = 0;
            do
            {
              v33 = *v32++;
              v26 = (unsigned int)sub_39FAC40(v33) + (unsigned int)v26;
            }
            while ( (__int64 *)v31 != v32 );
          }
        }
        ++v23;
        v24 += v103 - v26;
      }
      while ( v97 != v23 );
      v90 = v24;
      v35 = v99;
      if ( v101 != v91 )
        goto LABEL_57;
      v47 = 0;
LABEL_71:
      if ( v90 != v47 )
      {
        v48 = v90 < v47;
        goto LABEL_73;
      }
LABEL_85:
      ++v125;
      memset(v114, 0, sizeof(v114));
      v56 = *(_DWORD *)(a1 + 72);
      v57 = *(_QWORD *)(a1 + 8);
      v58 = *(_QWORD *)(a1 + 56);
      v113 = *(_QWORD *)(a1 + 48);
      v112[1] = v57;
      v112[0] = v58;
      v115 = v56;
      v116 = v58;
      v117 = __PAIR128__(v113, v57);
      v122[0] = v56;
      v118 = 0;
      v119 = 0;
      v120 = 0;
      v121 = 0;
      if ( v128 )
      {
LABEL_90:
        *(_QWORD *)&v127[4] = 0;
      }
      else
      {
        v59 = 4 * (*(_DWORD *)&v127[4] - *(_DWORD *)&v127[8]);
        if ( v59 < 0x20 )
          v59 = 32;
        if ( v59 >= *(_DWORD *)v127 )
        {
          memset(s, -1, 8LL * *(unsigned int *)v127);
          goto LABEL_90;
        }
        sub_C8C990((__int64)&v125, v58);
      }
      sub_285BFD0((__int64)v112, v94, (__int64)&v125, (__int64)&v108, v93, 0);
      ++v125;
      if ( v128 )
      {
LABEL_96:
        *(_QWORD *)&v127[4] = 0;
      }
      else
      {
        v60 = 4 * (*(_DWORD *)&v127[4] - *(_DWORD *)&v127[8]);
        if ( v60 < 0x20 )
          v60 = 32;
        if ( *(_DWORD *)v127 <= v60 )
        {
          memset(s, -1, 8LL * *(unsigned int *)v127);
          goto LABEL_96;
        }
        sub_C8C990((__int64)&v125, v94);
      }
      sub_285BFD0((__int64)&v116, v87, (__int64)&v125, (__int64)&v108, v93, 0);
      v48 = sub_2851E20((__int64)v112, (__int64)&v116);
LABEL_73:
      if ( v48 )
      {
        v116 = *(_QWORD *)v94;
        v117 = (unsigned __int128)_mm_loadu_si128((const __m128i *)(v94 + 8));
        LOBYTE(v118) = *(_BYTE *)(v94 + 24);
        v51 = *(_QWORD *)(v94 + 32);
        v120 = (char *)v122;
        v119 = v51;
        v121 = 0x400000000LL;
        v52 = *(unsigned int *)(v94 + 48);
        if ( (_DWORD)v52 )
          sub_28502F0((__int64)&v120, (char **)(v94 + 40), v52, v25, i, v11);
        v123 = *(_QWORD *)(v94 + 88);
        v124 = _mm_loadu_si128((const __m128i *)(v94 + 96));
        *(_QWORD *)v94 = *(_QWORD *)v87;
        *(_QWORD *)(v94 + 8) = *(_QWORD *)(v87 + 8);
        *(_BYTE *)(v94 + 16) = *(_BYTE *)(v87 + 16);
        *(_BYTE *)(v94 + 24) = *(_BYTE *)(v87 + 24);
        *(_QWORD *)(v94 + 32) = *(_QWORD *)(v87 + 32);
        sub_28502F0(v94 + 40, (char **)(v87 + 40), v52, v94, i, v11);
        *(_QWORD *)(v94 + 88) = *(_QWORD *)(v87 + 88);
        *(_QWORD *)(v94 + 96) = *(_QWORD *)(v87 + 96);
        *(_BYTE *)(v94 + 104) = *(_BYTE *)(v87 + 104);
        *(_QWORD *)v87 = v116;
        *(_QWORD *)(v87 + 8) = v117;
        *(_BYTE *)(v87 + 16) = BYTE8(v117);
        *(_BYTE *)(v87 + 24) = v118;
        *(_QWORD *)(v87 + 32) = v119;
        sub_28502F0(v87 + 40, &v120, v53, v94, v54, v55);
        *(_QWORD *)(v87 + 88) = v123;
        *(_QWORD *)(v87 + 96) = v124.m128i_i64[0];
        *(_BYTE *)(v87 + 104) = v124.m128i_i8[8];
        if ( v120 != (char *)v122 )
          _libc_free((unsigned __int64)v120);
      }
      v6 = 1;
      sub_28532A0(v93, (__int64 *)v94);
      --v92;
LABEL_30:
      ;
    }
    while ( v92 != v96 );
    if ( v6 )
      sub_2855860(v93, v85, a1 + 36280);
LABEL_33:
    ++v104;
    if ( (_DWORD)v106 )
    {
      v61 = 4 * v106;
      v19 = v107;
      if ( (unsigned int)(4 * v106) < 0x40 )
        v61 = 64;
      if ( v107 <= v61 )
      {
LABEL_36:
        v20 = v105;
        v21 = &v105[3 * v19];
        if ( v105 != v21 )
        {
          do
          {
            *v20 = -4096;
            v20 += 3;
            *(v20 - 2) = 0x7FFFFFFFFFFFFFFFLL;
          }
          while ( v21 != v20 );
        }
        v106 = 0;
        goto LABEL_39;
      }
      v62 = v105;
      v63 = 3LL * v107;
      if ( (_DWORD)v106 == 1 )
      {
        v69 = 3072;
        v68 = 128;
      }
      else
      {
        _BitScanReverse(&v64, v106 - 1);
        v65 = 1 << (33 - (v64 ^ 0x1F));
        if ( v65 < 64 )
          v65 = 64;
        if ( v107 == v65 )
        {
          v106 = 0;
          v83 = &v105[v63];
          do
          {
            if ( v62 )
            {
              *v62 = -4096;
              v62[1] = 0x7FFFFFFFFFFFFFFFLL;
            }
            v62 += 3;
          }
          while ( v83 != v62 );
          goto LABEL_39;
        }
        v66 = ((unsigned __int64)(4 * v65 / 3u + 1) >> 1)
            | (4 * v65 / 3u + 1)
            | ((((unsigned __int64)(4 * v65 / 3u + 1) >> 1) | (4 * v65 / 3u + 1)) >> 2);
        v67 = (((v66 | (v66 >> 4)) >> 8) | v66 | (v66 >> 4) | ((((v66 | (v66 >> 4)) >> 8) | v66 | (v66 >> 4)) >> 16))
            + 1;
        v68 = v67;
        v69 = 24 * v67;
      }
      sub_C7D6A0((__int64)v105, v63 * 8, 8);
      v107 = v68;
      v70 = (_QWORD *)sub_C7D670(v69, 8);
      v106 = 0;
      v105 = v70;
      for ( m = &v70[3 * v107]; m != v70; v70 += 3 )
      {
        if ( v70 )
        {
          *v70 = -4096;
          v70[1] = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
      goto LABEL_39;
    }
    if ( HIDWORD(v106) )
    {
      v19 = v107;
      if ( v107 <= 0x40 )
        goto LABEL_36;
      sub_C7D6A0((__int64)v105, 24LL * v107, 8);
      v105 = 0;
      v106 = 0;
      v107 = 0;
    }
LABEL_39:
    ++v85;
    v84 += 2184;
    if ( v85 != v86 )
    {
      v2 = *(_QWORD *)(a1 + 1320);
      continue;
    }
    break;
  }
  if ( !v128 )
    _libc_free((unsigned __int64)s);
  v17 = v109;
  v86 = 8LL * (unsigned int)v111;
LABEL_22:
  sub_C7D6A0(v17, v86, 8);
  return sub_C7D6A0((__int64)v105, 24LL * v107, 8);
}
