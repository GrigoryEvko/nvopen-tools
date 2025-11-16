// Function: sub_2A465E0
// Address: 0x2a465e0
//
__int64 __fastcall sub_2A465E0(_QWORD *a1, _DWORD *a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r15
  unsigned int *v20; // rax
  int v21; // ecx
  unsigned int *v22; // rdx
  __int64 v23; // r15
  __int64 v24; // rcx
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rbx
  unsigned int v32; // esi
  __int64 v33; // r10
  __int64 v34; // r8
  unsigned int v35; // edi
  __int64 *v36; // rax
  __int64 v37; // rcx
  __int64 v39; // rax
  unsigned __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // r15
  unsigned int *v44; // rax
  int v45; // ecx
  unsigned int *v46; // rdx
  __int64 v47; // r15
  __int64 v48; // rcx
  __int64 v49; // r9
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdi
  __m128i v53; // rax
  char v54; // al
  unsigned __int64 *v55; // rcx
  unsigned __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rbx
  unsigned int v59; // esi
  __int64 v60; // r15
  __int64 v61; // r8
  unsigned int v62; // r12d
  unsigned int v63; // edi
  __int64 *v64; // rdx
  __int64 v65; // rcx
  int v66; // r11d
  __int64 *v67; // rdx
  int v68; // eax
  int v69; // eax
  unsigned __int64 v70; // r8
  unsigned __int64 v71; // r9
  __m128i v72; // xmm1
  int v73; // r10d
  int v74; // r10d
  __int64 v75; // r8
  __int64 v76; // rcx
  __int64 v77; // rsi
  int v78; // r9d
  __int64 *v79; // rdi
  int v80; // r10d
  __int64 *v81; // r11
  int v82; // ecx
  int v83; // edx
  int v84; // r9d
  int v85; // r9d
  __int64 v86; // rdi
  __int64 *v87; // rsi
  __int64 v88; // r15
  int v89; // r8d
  __int64 v90; // rcx
  unsigned __int64 v91; // r8
  unsigned __int64 v92; // r9
  int v93; // r12d
  int v94; // r12d
  __int64 v95; // r10
  __int64 v96; // rcx
  __int64 v97; // r8
  int v98; // edi
  __int64 *v99; // rsi
  int v100; // r10d
  int v101; // r10d
  __int64 v102; // r9
  int v103; // esi
  __int64 v104; // r12
  __int64 *v105; // rcx
  __int64 v106; // rdi
  __int64 v107; // [rsp+0h] [rbp-1E0h]
  __int64 v108; // [rsp+8h] [rbp-1D8h]
  __int64 v109; // [rsp+18h] [rbp-1C8h]
  __int64 v111; // [rsp+28h] [rbp-1B8h]
  __int64 v113; // [rsp+38h] [rbp-1A8h]
  int v114; // [rsp+40h] [rbp-1A0h]
  unsigned __int64 v115; // [rsp+40h] [rbp-1A0h]
  int v116; // [rsp+48h] [rbp-198h]
  int v117; // [rsp+48h] [rbp-198h]
  __int64 v118; // [rsp+48h] [rbp-198h]
  __int64 v119; // [rsp+48h] [rbp-198h]
  unsigned __int64 v120; // [rsp+48h] [rbp-198h]
  __int64 v122; // [rsp+78h] [rbp-168h]
  __int64 v123; // [rsp+88h] [rbp-158h] BYREF
  __m128i v124; // [rsp+90h] [rbp-150h] BYREF
  __m128i v125; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v126; // [rsp+B0h] [rbp-130h]
  unsigned __int64 *v127; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v128; // [rsp+C8h] [rbp-118h]
  __int64 v129; // [rsp+D0h] [rbp-110h]
  __int16 v130; // [rsp+E0h] [rbp-100h]
  __m128i v131; // [rsp+F0h] [rbp-F0h] BYREF
  __m128i v132; // [rsp+100h] [rbp-E0h]
  __int64 v133; // [rsp+110h] [rbp-D0h]
  unsigned int *v134; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+128h] [rbp-B8h]
  _BYTE v136[32]; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v137; // [rsp+150h] [rbp-90h]
  unsigned __int64 v138; // [rsp+158h] [rbp-88h]
  __int16 v139; // [rsp+160h] [rbp-80h]
  __int64 v140; // [rsp+168h] [rbp-78h]
  void **v141; // [rsp+170h] [rbp-70h]
  _QWORD *v142; // [rsp+178h] [rbp-68h]
  __int64 v143; // [rsp+180h] [rbp-60h]
  int v144; // [rsp+188h] [rbp-58h]
  __int16 v145; // [rsp+18Ch] [rbp-54h]
  char v146; // [rsp+18Eh] [rbp-52h]
  __int64 v147; // [rsp+190h] [rbp-50h]
  __int64 v148; // [rsp+198h] [rbp-48h]
  void *v149; // [rsp+1A0h] [rbp-40h] BYREF
  _QWORD v150[7]; // [rsp+1A8h] [rbp-38h] BYREF

  v7 = *a3;
  v8 = 48LL * *((unsigned int *)a3 + 2);
  v9 = *a3 + v8;
  v10 = v8;
  v11 = v9;
  if ( v9 != v7 )
  {
    while ( !*(_QWORD *)(v11 - 32) )
    {
      v11 -= 48;
      if ( v7 == v11 )
      {
        v11 = v7;
        break;
      }
    }
  }
  v113 = v9 - v11;
  v12 = v7 + v8 - (v9 - v11);
  if ( v12 != v9 )
  {
    v111 = -48 - (v9 - v11);
    do
    {
      v13 = a4;
      if ( v12 != v7 )
        v13 = *(_QWORD *)(v12 - 32);
      v123 = v13;
      v122 = *(_QWORD *)(v12 + 32);
      v14 = a4;
      if ( v113 != v10 )
        v14 = *(_QWORD *)(v7 + v10 + v111 + 16);
      *(_QWORD *)(v122 + 40) = v14;
      v116 = *(_DWORD *)(v122 + 24);
      if ( (v116 & 0xFFFFFFFD) == 0 )
      {
        v39 = *(_QWORD *)(v122 + 56);
        v40 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v40 == v39 + 48 )
          goto LABEL_171;
        if ( !v40 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 > 0xA )
        {
LABEL_171:
          v5 = sub_BD5C60(0);
          v146 = 7;
          v140 = v5;
          v141 = &v149;
          v142 = v150;
          v134 = (unsigned int *)v136;
          v135 = 0x200000000LL;
          v149 = &unk_49DA100;
          v143 = 0;
          v144 = 0;
          v145 = 512;
          v147 = 0;
          v148 = 0;
          v137 = 0;
          v138 = 0;
          v139 = 0;
          v150[0] = &unk_49DA0B0;
          BUG();
        }
        v140 = sub_BD5C60(v40 - 24);
        v141 = &v149;
        v142 = v150;
        v135 = 0x200000000LL;
        v149 = &unk_49DA100;
        v145 = 512;
        v134 = (unsigned int *)v136;
        v137 = 0;
        v138 = 0;
        v143 = 0;
        v144 = 0;
        v146 = 7;
        v147 = 0;
        v148 = 0;
        v139 = 0;
        v150[0] = &unk_49DA0B0;
        v41 = *(_QWORD *)(v40 + 16);
        v138 = v40;
        v137 = v41;
        v42 = *(_QWORD *)sub_B46C60(v40 - 24);
        v131.m128i_i64[0] = v42;
        if ( !v42 || (sub_B96E90((__int64)&v131, v42, 1), (v43 = v131.m128i_i64[0]) == 0) )
        {
          sub_93FB40((__int64)&v134, 0);
          v43 = v131.m128i_i64[0];
          goto LABEL_76;
        }
        v44 = v134;
        v45 = v135;
        v46 = &v134[4 * (unsigned int)v135];
        if ( v134 != v46 )
        {
          while ( *v44 )
          {
            v44 += 4;
            if ( v46 == v44 )
              goto LABEL_78;
          }
          *((_QWORD *)v44 + 1) = v131.m128i_i64[0];
          goto LABEL_49;
        }
LABEL_78:
        if ( (unsigned int)v135 >= (unsigned __int64)HIDWORD(v135) )
        {
          v91 = (unsigned int)v135 + 1LL;
          v92 = v107 & 0xFFFFFFFF00000000LL;
          v107 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v135) < v91 )
          {
            v120 = v92;
            sub_C8D5F0((__int64)&v134, v136, v91, 0x10u, v91, v92);
            v92 = v120;
            v46 = &v134[4 * (unsigned int)v135];
          }
          *(_QWORD *)v46 = v92;
          *((_QWORD *)v46 + 1) = v43;
          v43 = v131.m128i_i64[0];
          LODWORD(v135) = v135 + 1;
        }
        else
        {
          if ( v46 )
          {
            *v46 = 0;
            *((_QWORD *)v46 + 1) = v43;
            v45 = v135;
            v43 = v131.m128i_i64[0];
          }
          LODWORD(v135) = v45 + 1;
        }
LABEL_76:
        if ( v43 )
LABEL_49:
          sub_B91220((__int64)&v131, v43);
        v117 = sub_BA8BD0(*(_QWORD *)(a1[1] + 40LL));
        v131.m128i_i64[0] = *(_QWORD *)(v123 + 8);
        v47 = sub_B6E160(*(__int64 **)(a1[1] + 40LL), 0x150u, (__int64)&v131, 1);
        if ( (unsigned int)sub_BA8BD0(*(_QWORD *)(a1[1] + 40LL)) != v117 )
        {
          v50 = *a1;
          v127 = 0;
          v128 = 0;
          v51 = v50 + 56;
          v129 = v47;
          if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
            sub_BD73F0((__int64)&v127);
          sub_2A46180((__int64)&v131, v51, (unsigned __int64 *)&v127, v48, (__int64)&v127, v49);
          if ( v129 != 0 && v129 != -4096 && v129 != -8192 )
            sub_BD60C0(&v127);
        }
        v52 = v123;
        LODWORD(v127) = *a2;
        *a2 = (_DWORD)v127 + 1;
        v130 = 265;
        v53.m128i_i64[0] = (__int64)sub_BD5D20(v52);
        v124 = v53;
        v125.m128i_i64[0] = (__int64)".";
        v54 = v130;
        LOWORD(v126) = 773;
        if ( (_BYTE)v130 )
        {
          if ( (_BYTE)v130 == 1 )
          {
            v72 = _mm_load_si128(&v125);
            v131 = _mm_load_si128(&v124);
            v133 = v126;
            v132 = v72;
          }
          else
          {
            if ( HIBYTE(v130) == 1 )
            {
              v55 = v127;
              v109 = v128;
            }
            else
            {
              v55 = (unsigned __int64 *)&v127;
              v54 = 2;
            }
            v132.m128i_i64[0] = (__int64)v55;
            v131.m128i_i64[0] = (__int64)&v124;
            v132.m128i_i64[1] = v109;
            LOBYTE(v133) = 2;
            BYTE1(v133) = v54;
          }
        }
        else
        {
          LOWORD(v133) = 256;
        }
        v56 = 0;
        if ( v47 )
          v56 = *(_QWORD *)(v47 + 24);
        v57 = sub_921880(&v134, v56, v47, (int)&v123, 1, (__int64)&v131, 0);
        v58 = *a1;
        v59 = *(_DWORD *)(*a1 + 48LL);
        v60 = *a1 + 24LL;
        if ( v59 )
        {
          v61 = *(_QWORD *)(v58 + 32);
          v62 = ((unsigned int)v57 >> 4) ^ ((unsigned int)v57 >> 9);
          v63 = (v59 - 1) & v62;
          v64 = (__int64 *)(v61 + 16LL * v63);
          v65 = *v64;
          if ( v57 == *v64 )
          {
LABEL_74:
            *(_QWORD *)(v12 + 16) = v57;
            goto LABEL_35;
          }
          v80 = 1;
          v81 = 0;
          while ( v65 != -4096 )
          {
            if ( v81 || v65 != -8192 )
              v64 = v81;
            v63 = (v59 - 1) & (v80 + v63);
            v65 = *(_QWORD *)(v61 + 16LL * v63);
            if ( v57 == v65 )
              goto LABEL_74;
            ++v80;
            v81 = v64;
            v64 = (__int64 *)(v61 + 16LL * v63);
          }
          v82 = *(_DWORD *)(v58 + 40);
          if ( !v81 )
            v81 = v64;
          ++*(_QWORD *)(v58 + 24);
          v83 = v82 + 1;
          if ( 4 * (v82 + 1) < 3 * v59 )
          {
            if ( v59 - *(_DWORD *)(v58 + 44) - v83 <= v59 >> 3 )
            {
              v119 = v57;
              sub_2A46400(v60, v59);
              v100 = *(_DWORD *)(v58 + 48);
              if ( !v100 )
              {
LABEL_169:
                ++*(_DWORD *)(v58 + 40);
                BUG();
              }
              v101 = v100 - 1;
              v102 = *(_QWORD *)(v58 + 32);
              v103 = 1;
              LODWORD(v104) = v101 & v62;
              v105 = 0;
              v83 = *(_DWORD *)(v58 + 40) + 1;
              v57 = v119;
              v81 = (__int64 *)(v102 + 16LL * (unsigned int)v104);
              v106 = *v81;
              if ( v119 != *v81 )
              {
                while ( v106 != -4096 )
                {
                  if ( v106 == -8192 && !v105 )
                    v105 = v81;
                  v104 = v101 & (unsigned int)(v104 + v103);
                  v81 = (__int64 *)(v102 + 16 * v104);
                  v106 = *v81;
                  if ( v119 == *v81 )
                    goto LABEL_110;
                  ++v103;
                }
                if ( v105 )
                  v81 = v105;
              }
            }
            goto LABEL_110;
          }
        }
        else
        {
          ++*(_QWORD *)(v58 + 24);
        }
        v118 = v57;
        sub_2A46400(v60, 2 * v59);
        v93 = *(_DWORD *)(v58 + 48);
        if ( !v93 )
          goto LABEL_169;
        v57 = v118;
        v94 = v93 - 1;
        v95 = *(_QWORD *)(v58 + 32);
        v83 = *(_DWORD *)(v58 + 40) + 1;
        LODWORD(v96) = v94 & (((unsigned int)v118 >> 9) ^ ((unsigned int)v118 >> 4));
        v81 = (__int64 *)(v95 + 16LL * (unsigned int)v96);
        v97 = *v81;
        if ( v118 != *v81 )
        {
          v98 = 1;
          v99 = 0;
          while ( v97 != -4096 )
          {
            if ( !v99 && v97 == -8192 )
              v99 = v81;
            v96 = v94 & (unsigned int)(v96 + v98);
            v81 = (__int64 *)(v95 + 16 * v96);
            v97 = *v81;
            if ( v118 == *v81 )
              goto LABEL_110;
            ++v98;
          }
          if ( v99 )
            v81 = v99;
        }
LABEL_110:
        *(_DWORD *)(v58 + 40) = v83;
        if ( *v81 != -4096 )
          --*(_DWORD *)(v58 + 44);
        *v81 = v57;
        v81[1] = v122;
        goto LABEL_74;
      }
      if ( v116 != 1 )
        BUG();
      v15 = *(_QWORD *)(v122 + 56);
      v16 = *(_QWORD *)(v15 + 32);
      if ( v16 == *(_QWORD *)(v15 + 40) + 48LL || !v16 )
      {
        v4 = sub_BD5C60(0);
        v146 = 7;
        v140 = v4;
        v141 = &v149;
        v142 = v150;
        v134 = (unsigned int *)v136;
        v135 = 0x200000000LL;
        v149 = &unk_49DA100;
        v143 = 0;
        v144 = 0;
        v145 = 512;
        v147 = 0;
        v148 = 0;
        v137 = 0;
        v138 = 0;
        v139 = 0;
        v150[0] = &unk_49DA0B0;
        BUG();
      }
      v140 = sub_BD5C60(v16 - 24);
      v141 = &v149;
      v142 = v150;
      v135 = 0x200000000LL;
      v149 = &unk_49DA100;
      v145 = 512;
      v134 = (unsigned int *)v136;
      v137 = 0;
      v138 = 0;
      v143 = 0;
      v144 = 0;
      v146 = 7;
      v147 = 0;
      v148 = 0;
      v139 = 0;
      v150[0] = &unk_49DA0B0;
      v17 = *(_QWORD *)(v16 + 16);
      v138 = v16;
      v137 = v17;
      v18 = *(_QWORD *)sub_B46C60(v16 - 24);
      v131.m128i_i64[0] = v18;
      if ( v18 && (sub_B96E90((__int64)&v131, v18, 1), (v19 = v131.m128i_i64[0]) != 0) )
      {
        v20 = v134;
        v21 = v135;
        v22 = &v134[4 * (unsigned int)v135];
        if ( v134 != v22 )
        {
          while ( *v20 )
          {
            v20 += 4;
            if ( v22 == v20 )
              goto LABEL_62;
          }
          *((_QWORD *)v20 + 1) = v131.m128i_i64[0];
LABEL_22:
          sub_B91220((__int64)&v131, v19);
          goto LABEL_23;
        }
LABEL_62:
        if ( (unsigned int)v135 >= (unsigned __int64)HIDWORD(v135) )
        {
          v70 = (unsigned int)v135 + 1LL;
          v71 = v108 & 0xFFFFFFFF00000000LL;
          v108 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v135) < v70 )
          {
            v115 = v71;
            sub_C8D5F0((__int64)&v134, v136, v70, 0x10u, v70, v71);
            v71 = v115;
            v22 = &v134[4 * (unsigned int)v135];
          }
          *(_QWORD *)v22 = v71;
          *((_QWORD *)v22 + 1) = v19;
          v19 = v131.m128i_i64[0];
          LODWORD(v135) = v135 + 1;
        }
        else
        {
          if ( v22 )
          {
            *v22 = 0;
            *((_QWORD *)v22 + 1) = v19;
            v21 = v135;
            v19 = v131.m128i_i64[0];
          }
          LODWORD(v135) = v21 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v134, 0);
        v19 = v131.m128i_i64[0];
      }
      if ( v19 )
        goto LABEL_22;
LABEL_23:
      v114 = sub_BA8BD0(*(_QWORD *)(a1[1] + 40LL));
      v131.m128i_i64[0] = *(_QWORD *)(v123 + 8);
      v23 = sub_B6E160(*(__int64 **)(a1[1] + 40LL), 0x150u, (__int64)&v131, 1);
      if ( (unsigned int)sub_BA8BD0(*(_QWORD *)(a1[1] + 40LL)) != v114 )
      {
        v26 = *a1;
        v127 = 0;
        v128 = 0;
        v27 = v26 + 56;
        v129 = v23;
        if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
          sub_BD73F0((__int64)&v127);
        sub_2A46180((__int64)&v131, v27, (unsigned __int64 *)&v127, v24, (__int64)&v127, v25);
        if ( v129 != 0 && v129 != -4096 && v129 != -8192 )
          sub_BD60C0(&v127);
      }
      v28 = 0;
      LOWORD(v133) = 257;
      if ( v23 )
        v28 = *(_QWORD *)(v23 + 24);
      v29 = sub_921880(&v134, v28, v23, (int)&v123, 1, (__int64)&v131, 0);
      v30 = *a1;
      v31 = v29;
      v32 = *(_DWORD *)(*a1 + 48LL);
      v33 = *a1 + 24LL;
      if ( !v32 )
      {
        ++*(_QWORD *)(v30 + 24);
        goto LABEL_97;
      }
      v34 = *(_QWORD *)(v30 + 32);
      v35 = (v32 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v36 = (__int64 *)(v34 + 16LL * v35);
      v37 = *v36;
      if ( v31 != *v36 )
      {
        v66 = 1;
        v67 = 0;
        while ( v37 != -4096 )
        {
          if ( v67 || v37 != -8192 )
            v36 = v67;
          v35 = (v32 - 1) & (v66 + v35);
          v37 = *(_QWORD *)(v34 + 16LL * v35);
          if ( v31 == v37 )
            goto LABEL_34;
          ++v66;
          v67 = v36;
          v36 = (__int64 *)(v34 + 16LL * v35);
        }
        if ( !v67 )
          v67 = v36;
        v68 = *(_DWORD *)(v30 + 40);
        ++*(_QWORD *)(v30 + 24);
        v69 = v68 + 1;
        if ( 4 * v69 >= 3 * v32 )
        {
LABEL_97:
          sub_2A46400(v33, 2 * v32);
          v73 = *(_DWORD *)(v30 + 48);
          if ( !v73 )
            goto LABEL_170;
          v74 = v73 - 1;
          v75 = *(_QWORD *)(v30 + 32);
          LODWORD(v76) = v74 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v69 = *(_DWORD *)(v30 + 40) + 1;
          v67 = (__int64 *)(v75 + 16LL * (unsigned int)v76);
          v77 = *v67;
          if ( v31 != *v67 )
          {
            v78 = 1;
            v79 = 0;
            while ( v77 != -4096 )
            {
              if ( !v79 && v77 == -8192 )
                v79 = v67;
              v76 = v74 & (unsigned int)(v76 + v78);
              v67 = (__int64 *)(v75 + 16 * v76);
              v77 = *v67;
              if ( v31 == *v67 )
                goto LABEL_89;
              ++v78;
            }
            if ( v79 )
              v67 = v79;
          }
        }
        else if ( v32 - *(_DWORD *)(v30 + 44) - v69 <= v32 >> 3 )
        {
          sub_2A46400(v33, v32);
          v84 = *(_DWORD *)(v30 + 48);
          if ( !v84 )
          {
LABEL_170:
            ++*(_DWORD *)(v30 + 40);
            BUG();
          }
          v85 = v84 - 1;
          v86 = *(_QWORD *)(v30 + 32);
          v87 = 0;
          LODWORD(v88) = v85 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v89 = 1;
          v69 = *(_DWORD *)(v30 + 40) + 1;
          v67 = (__int64 *)(v86 + 16LL * (unsigned int)v88);
          v90 = *v67;
          if ( v31 != *v67 )
          {
            while ( v90 != -4096 )
            {
              if ( !v87 && v90 == -8192 )
                v87 = v67;
              v88 = v85 & (unsigned int)(v88 + v89);
              v67 = (__int64 *)(v86 + 16 * v88);
              v90 = *v67;
              if ( v31 == *v67 )
                goto LABEL_89;
              ++v89;
            }
            if ( v87 )
              v67 = v87;
          }
        }
LABEL_89:
        *(_DWORD *)(v30 + 40) = v69;
        if ( *v67 != -4096 )
          --*(_DWORD *)(v30 + 44);
        *v67 = v31;
        v67[1] = v122;
      }
LABEL_34:
      *(_QWORD *)(v12 + 16) = v31;
LABEL_35:
      nullsub_61();
      v149 = &unk_49DA100;
      nullsub_63();
      if ( v134 != (unsigned int *)v136 )
        _libc_free((unsigned __int64)v134);
      v12 += 48;
      v7 = *a3;
      v10 = 48LL * *((unsigned int *)a3 + 2);
    }
    while ( v12 != *a3 + v10 );
  }
  return *(_QWORD *)(v7 + v10 - 32);
}
