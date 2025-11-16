// Function: sub_26DAB40
// Address: 0x26dab40
//
__int64 __fastcall sub_26DAB40(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r15
  unsigned int v4; // esi
  __int64 v5; // rax
  __int64 v6; // r9
  int v7; // ebx
  __int64 *v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // rcx
  __int64 v15; // r8
  int v16; // r11d
  unsigned int v17; // eax
  __int64 *v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // r10
  __int64 *v21; // r12
  __int64 *v22; // r13
  __int64 v23; // r14
  __int64 v24; // rbx
  bool v25; // zf
  __int64 **v26; // rax
  __int64 **v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // r8
  int v30; // r11d
  __int64 *v31; // r10
  unsigned int k; // eax
  __int64 *v33; // rdx
  __int64 v34; // rcx
  char v35; // r12
  __int64 v36; // rsi
  __int64 *v37; // rdi
  __int64 *v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 *v42; // rax
  __int64 *v43; // rax
  __int64 *v44; // r12
  __int64 v45; // r13
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 *v52; // rax
  __int64 *v53; // r13
  __int64 v54; // r14
  __int64 v55; // rbx
  __int64 v56; // r12
  _QWORD *v57; // rax
  _QWORD *v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // r10
  int v61; // r11d
  __int64 *v62; // rdx
  unsigned int j; // eax
  __int64 *v64; // rcx
  __int64 v65; // rdi
  __int64 v66; // r13
  __int64 *v67; // rax
  __int64 v68; // rdx
  __int64 *v69; // rbx
  unsigned __int64 v70; // rbx
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 *v76; // rax
  __int64 v77; // rdi
  unsigned __int64 v78; // rbx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  char v82; // dl
  int v83; // eax
  int v84; // edx
  __int64 *v85; // rax
  unsigned __int64 v86; // rbx
  __int64 v87; // rbx
  __int64 v88; // rax
  int v89; // ecx
  __int64 v90; // rax
  __int64 v91; // rdi
  int v92; // ecx
  __int64 v93; // rax
  unsigned int v94; // eax
  __int64 *v95; // rax
  __int64 *v96; // r12
  __int64 v97; // r13
  __int64 v98; // r15
  __int64 v99; // rax
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // r9
  int v103; // ecx
  int v104; // ecx
  unsigned int v105; // eax
  int v106; // eax
  __int64 v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // rbx
  int v110; // eax
  __int64 v111; // [rsp+8h] [rbp-F8h]
  __int64 v112; // [rsp+18h] [rbp-E8h]
  __int64 v113; // [rsp+20h] [rbp-E0h]
  __int64 v114; // [rsp+28h] [rbp-D8h]
  __int64 v115; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v116; // [rsp+28h] [rbp-D8h]
  __int64 v117; // [rsp+38h] [rbp-C8h]
  unsigned int v118; // [rsp+40h] [rbp-C0h]
  char v119; // [rsp+40h] [rbp-C0h]
  __int64 v120; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v121; // [rsp+48h] [rbp-B8h]
  unsigned __int8 v122; // [rsp+52h] [rbp-AEh]
  int i; // [rsp+54h] [rbp-ACh]
  unsigned int v125; // [rsp+58h] [rbp-A8h]
  __int64 v126; // [rsp+58h] [rbp-A8h]
  unsigned __int8 v127; // [rsp+58h] [rbp-A8h]
  __int64 v128; // [rsp+58h] [rbp-A8h]
  __int64 *v129; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v130; // [rsp+68h] [rbp-98h] BYREF
  __m128i v131; // [rsp+70h] [rbp-90h] BYREF
  __m128i v132; // [rsp+80h] [rbp-80h] BYREF
  __int128 v133; // [rsp+90h] [rbp-70h] BYREF
  __m128i v134; // [rsp+A0h] [rbp-60h] BYREF
  __int64 *v135; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v136; // [rsp+B8h] [rbp-48h]

  v3 = a1;
  v112 = a2 + 72;
  v117 = *(_QWORD *)(a2 + 80);
  if ( v117 == a2 + 72 )
    return 0;
  v122 = 0;
  v111 = a1 + 968;
  v113 = a1 + 104;
  do
  {
    v4 = *(_DWORD *)(v3 + 992);
    v5 = 0;
    if ( v117 )
      v5 = v117 - 24;
    v129 = (__int64 *)v5;
    if ( !v4 )
    {
      ++*(_QWORD *)(v3 + 968);
      v135 = 0;
      goto LABEL_186;
    }
    v6 = *(_QWORD *)(v3 + 976);
    v7 = 1;
    v8 = 0;
    v9 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v5 != *v10 )
    {
      while ( v11 != -4096 )
      {
        if ( v11 == -8192 && !v8 )
          v8 = v10;
        v9 = (v4 - 1) & (v7 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v5 == *v10 )
          goto LABEL_7;
        ++v7;
      }
      v103 = *(_DWORD *)(v3 + 984);
      if ( !v8 )
        v8 = v10;
      ++*(_QWORD *)(v3 + 968);
      v104 = v103 + 1;
      v135 = v8;
      if ( 4 * v104 < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(v3 + 988) - v104 > v4 >> 3 )
        {
LABEL_182:
          *(_DWORD *)(v3 + 984) = v104;
          if ( *v8 != -4096 )
            --*(_DWORD *)(v3 + 988);
          *v8 = v5;
          v12 = 0;
          v8[1] = 0;
          goto LABEL_8;
        }
LABEL_187:
        sub_1059000(v111, v4);
        sub_26CE030(v111, (__int64 *)&v129, &v135);
        v5 = (__int64)v129;
        v104 = *(_DWORD *)(v3 + 984) + 1;
        v8 = v135;
        goto LABEL_182;
      }
LABEL_186:
      v4 *= 2;
      goto LABEL_187;
    }
LABEL_7:
    v12 = v10[1];
LABEL_8:
    v130 = v12;
    for ( i = 0; ; i = 1 )
    {
      v131 = 0;
      v132 = 0;
      v133 = 0;
      if ( !i )
      {
        v52 = sub_26CA180(v3 + 1024, (__int64 *)&v129);
        v53 = (__int64 *)*v52;
        v118 = *((_DWORD *)v52 + 2);
        v54 = *v52 + 8LL * v118;
        if ( *v52 != v54 )
        {
          v125 = 0;
          v121 = 0;
          while ( 1 )
          {
            v55 = *v53;
            v56 = (__int64)v129;
            v25 = *(_QWORD *)(v3 + 960) == 0;
            v135 = (__int64 *)*v53;
            v136 = (__int64)v129;
            if ( v25 )
            {
              v57 = *(_QWORD **)(v3 + 392);
              v58 = &v57[2 * *(unsigned int *)(v3 + 400)];
              if ( v57 == v58 )
                goto LABEL_71;
              while ( v55 != *v57 || v129 != (__int64 *)v57[1] )
              {
                v57 += 2;
                if ( v58 == v57 )
                  goto LABEL_71;
              }
            }
            else
            {
              v57 = sub_26D8C10(v3 + 920, (unsigned __int64 *)&v135);
              v58 = (_QWORD *)(v3 + 928);
            }
            if ( v58 == v57 )
            {
LABEL_71:
              ++v125;
              v131.m128i_i64[0] = v55;
              v131.m128i_i64[1] = v56;
              goto LABEL_72;
            }
            v59 = *(_DWORD *)(v3 + 96);
            if ( !v59 )
            {
              ++*(_QWORD *)(v3 + 72);
              v134.m128i_i64[0] = 0;
LABEL_148:
              v59 *= 2;
              goto LABEL_149;
            }
            v60 = *(_QWORD *)(v3 + 80);
            v61 = 1;
            v62 = 0;
            for ( j = (v59 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4)
                        | ((unsigned __int64)(((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4)))); ; j = (v59 - 1) & v105 )
            {
              v64 = (__int64 *)(v60 + 24LL * j);
              v65 = *v64;
              if ( v55 == *v64 && v56 == v64[1] )
              {
                v121 += v64[2];
                goto LABEL_72;
              }
              if ( v65 == -4096 )
                break;
              if ( v65 == -8192 && v64[1] == -8192 && !v62 )
                v62 = (__int64 *)(v60 + 24LL * j);
LABEL_189:
              v105 = v61 + j;
              ++v61;
            }
            if ( v64[1] != -4096 )
              goto LABEL_189;
            v110 = *(_DWORD *)(v3 + 88);
            if ( !v62 )
              v62 = v64;
            ++*(_QWORD *)(v3 + 72);
            v92 = v110 + 1;
            v134.m128i_i64[0] = (__int64)v62;
            if ( 4 * (v110 + 1) >= 3 * v59 )
              goto LABEL_148;
            v91 = v55;
            if ( v59 - *(_DWORD *)(v3 + 92) - v92 <= v59 >> 3 )
            {
LABEL_149:
              sub_26CC5A0(v3 + 72, v59);
              sub_26C3690(v3 + 72, (__int64 *)&v135, (__int64 **)&v134);
              v91 = (__int64)v135;
              v62 = (__int64 *)v134.m128i_i64[0];
              v92 = *(_DWORD *)(v3 + 88) + 1;
            }
            *(_DWORD *)(v3 + 88) = v92;
            if ( *v62 != -4096 || v62[1] != -4096 )
              --*(_DWORD *)(v3 + 92);
            *v62 = v91;
            v93 = v136;
            v62[2] = 0;
            v62[1] = v93;
LABEL_72:
            if ( v55 == v56 )
            {
              v132.m128i_i64[0] = v55;
              v132.m128i_i64[1] = v55;
            }
            if ( (__int64 *)v54 == ++v53 )
            {
              v35 = a3 & (v121 != 0);
              if ( v118 != 1 )
                goto LABEL_34;
              goto LABEL_76;
            }
          }
        }
        if ( v118 != 1 )
        {
LABEL_167:
          v66 = v3 + 40;
          v35 = 0;
          v121 = 0;
          v69 = sub_26CC460(v3 + 40, &v130);
          goto LABEL_122;
        }
        v125 = 0;
        v35 = 0;
        v121 = 0;
LABEL_76:
        v118 = 1;
        *(_QWORD *)&v133 = *(_QWORD *)*sub_26CA180(v3 + 1024, (__int64 *)&v129);
        *((_QWORD *)&v133 + 1) = v129;
        if ( v125 > 1 )
          goto LABEL_35;
LABEL_77:
        v66 = v3 + 40;
        v67 = sub_26CC460(v3 + 40, &v130);
        v39 = v125;
        v69 = v67;
        if ( !v125 )
        {
LABEL_122:
          if ( *(_BYTE *)(v3 + 132) )
          {
            v85 = *(__int64 **)(v3 + 112);
            v38 = &v85[*(unsigned int *)(v3 + 124)];
            if ( v85 == v38 )
            {
LABEL_156:
              if ( *v69 < v121 )
              {
                v122 = 1;
                *v69 = v121;
              }
              goto LABEL_46;
            }
            while ( v130 != *v85 )
            {
              if ( v38 == ++v85 )
                goto LABEL_156;
            }
          }
          else if ( !sub_C8CA60(v113, v130) )
          {
            goto LABEL_156;
          }
          if ( v118 == 1 )
          {
            v86 = *sub_26CC870(v3 + 72, (__int64 *)&v133);
            if ( v86 < *sub_26CC460(v66, &v130) )
            {
              v87 = *sub_26CC460(v66, &v130);
              v122 = 1;
              *sub_26CC870(v3 + 72, (__int64 *)&v133) = v87;
            }
          }
          goto LABEL_46;
        }
        v127 = sub_B19060(v113, v130, v68, v125);
        if ( !v127 )
          goto LABEL_46;
        v70 = *v69;
        v120 = v3 + 72;
        if ( v70 < v121 )
        {
          *sub_26CC870(v120, v131.m128i_i64) = 0;
          if ( i )
          {
LABEL_81:
            v134.m128i_i64[0] = *sub_26D72C0(v111, &v131.m128i_i64[1]);
            if ( !(unsigned __int8)sub_B19060(v113, v134.m128i_i64[0], v71, v72) )
              goto LABEL_82;
LABEL_197:
            v116 = *sub_26CC870(v120, v131.m128i_i64);
            if ( v116 > *sub_26CC460(v3 + 40, v134.m128i_i64) )
            {
              v109 = *sub_26CC460(v3 + 40, v134.m128i_i64);
              *sub_26CC870(v120, v131.m128i_i64) = v109;
            }
LABEL_82:
            sub_26DA910((__int64)&v135, v3 + 392, &v131, v73, v74, v75);
            v122 = v127;
            if ( !v35 )
              goto LABEL_47;
            goto LABEL_83;
          }
        }
        else
        {
          *sub_26CC870(v3 + 72, v131.m128i_i64) = v70 - v121;
          if ( i )
            goto LABEL_81;
        }
        v134.m128i_i64[0] = *sub_26D72C0(v111, v131.m128i_i64);
        if ( !(unsigned __int8)sub_B19060(v113, v134.m128i_i64[0], v107, v108) )
          goto LABEL_82;
        goto LABEL_197;
      }
      v13 = *(_DWORD *)(v3 + 1080);
      v114 = v3 + 1056;
      if ( !v13 )
      {
        ++*(_QWORD *)(v3 + 1056);
        v135 = 0;
        goto LABEL_164;
      }
      v14 = (__int64)v129;
      v15 = *(_QWORD *)(v3 + 1064);
      v16 = i;
      v17 = (v13 - 1) & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
      v18 = 0;
      v19 = v15 + 88LL * v17;
      v20 = *(_QWORD *)v19;
      if ( v129 != *(__int64 **)v19 )
      {
        while ( v20 != -4096 )
        {
          if ( v20 == -8192 && !v18 )
            v18 = (__int64 *)v19;
          v17 = (v13 - 1) & (v16 + v17);
          v19 = v15 + 88LL * v17;
          v20 = *(_QWORD *)v19;
          if ( v129 == *(__int64 **)v19 )
            goto LABEL_12;
          ++v16;
        }
        v83 = *(_DWORD *)(v3 + 1072);
        if ( !v18 )
          v18 = (__int64 *)v19;
        ++*(_QWORD *)(v3 + 1056);
        v84 = v83 + 1;
        v135 = v18;
        if ( 4 * (v83 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(v3 + 1076) - v84 > v13 >> 3 )
          {
LABEL_119:
            *(_DWORD *)(v3 + 1072) = v84;
            if ( *v18 != -4096 )
              --*(_DWORD *)(v3 + 1076);
            *v18 = v14;
            v66 = v3 + 40;
            v35 = 0;
            v18[1] = (__int64)(v18 + 3);
            v18[2] = 0x800000000LL;
            v118 = 0;
            v121 = 0;
            v69 = sub_26CC460(v3 + 40, &v130);
            goto LABEL_122;
          }
LABEL_165:
          sub_26C9E50(v114, v13);
          sub_26C3200(v114, (__int64 *)&v129, &v135);
          v14 = (__int64)v129;
          v18 = v135;
          v84 = *(_DWORD *)(v3 + 1072) + 1;
          goto LABEL_119;
        }
LABEL_164:
        v13 *= 2;
        goto LABEL_165;
      }
LABEL_12:
      v21 = *(__int64 **)(v19 + 8);
      v22 = &v21[*(unsigned int *)(v19 + 16)];
      v118 = *(_DWORD *)(v19 + 16);
      if ( v22 == v21 )
      {
        if ( v118 == 1 )
        {
          v125 = 0;
          v35 = 0;
          v121 = 0;
          goto LABEL_132;
        }
        goto LABEL_167;
      }
      v125 = 0;
      v121 = 0;
      do
      {
        v23 = *v21;
        v24 = (__int64)v129;
        v25 = *(_QWORD *)(v3 + 960) == 0;
        v135 = v129;
        v136 = v23;
        if ( v25 )
        {
          v26 = *(__int64 ***)(v3 + 392);
          v27 = &v26[2 * *(unsigned int *)(v3 + 400)];
          if ( v26 == v27 )
            goto LABEL_31;
          while ( v129 != *v26 || (__int64 *)v23 != v26[1] )
          {
            v26 += 2;
            if ( v27 == v26 )
              goto LABEL_31;
          }
          if ( v26 == v27 )
          {
LABEL_31:
            ++v125;
            v131.m128i_i64[0] = v24;
            v131.m128i_i64[1] = v23;
            goto LABEL_32;
          }
        }
        else if ( (_QWORD *)(v3 + 928) == sub_26D8C10(v3 + 920, (unsigned __int64 *)&v135) )
        {
          goto LABEL_31;
        }
        v28 = *(_DWORD *)(v3 + 96);
        if ( !v28 )
        {
          ++*(_QWORD *)(v3 + 72);
          v134.m128i_i64[0] = 0;
LABEL_134:
          v28 *= 2;
          goto LABEL_135;
        }
        v29 = *(_QWORD *)(v3 + 80);
        v30 = i;
        v31 = 0;
        for ( k = (v28 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
                    | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)))); ; k = (v28 - 1) & v94 )
        {
          v33 = (__int64 *)(v29 + 24LL * k);
          v34 = *v33;
          if ( v24 == *v33 && v23 == v33[1] )
          {
            v121 += v33[2];
            goto LABEL_32;
          }
          if ( v34 == -4096 )
            break;
          if ( v34 == -8192 && v33[1] == -8192 && !v31 )
            v31 = (__int64 *)(v29 + 24LL * k);
LABEL_161:
          v94 = v30 + k;
          ++v30;
        }
        if ( v33[1] != -4096 )
          goto LABEL_161;
        v106 = *(_DWORD *)(v3 + 88);
        if ( v31 )
          v33 = v31;
        ++*(_QWORD *)(v3 + 72);
        v89 = v106 + 1;
        v134.m128i_i64[0] = (__int64)v33;
        if ( 4 * (v106 + 1) >= 3 * v28 )
          goto LABEL_134;
        if ( v28 - *(_DWORD *)(v3 + 92) - v89 <= v28 >> 3 )
        {
LABEL_135:
          sub_26CC5A0(v3 + 72, v28);
          sub_26C3690(v3 + 72, (__int64 *)&v135, (__int64 **)&v134);
          v24 = (__int64)v135;
          v33 = (__int64 *)v134.m128i_i64[0];
          v89 = *(_DWORD *)(v3 + 88) + 1;
        }
        *(_DWORD *)(v3 + 88) = v89;
        if ( *v33 != -4096 || v33[1] != -4096 )
          --*(_DWORD *)(v3 + 92);
        *v33 = v24;
        v90 = v136;
        v33[2] = 0;
        v33[1] = v90;
LABEL_32:
        ++v21;
      }
      while ( v22 != v21 );
      v35 = a3 & (v121 != 0);
      if ( v118 != 1 )
        goto LABEL_34;
LABEL_132:
      v88 = *(_QWORD *)*sub_26CA180(v114, (__int64 *)&v129);
      *(_QWORD *)&v133 = v129;
      *((_QWORD *)&v133 + 1) = v88;
      v118 = i;
LABEL_34:
      if ( v125 <= 1 )
        goto LABEL_77;
LABEL_35:
      v36 = v130;
      if ( !*(_BYTE *)(v3 + 132) )
      {
        if ( sub_C8CA60(v113, v130) )
        {
LABEL_40:
          if ( *sub_26CC460(v3 + 40, &v130) )
            goto LABEL_96;
          if ( i )
          {
            v43 = sub_26CA180(v3 + 1056, (__int64 *)&v129);
            v39 = *v43;
            v126 = *v43 + 8LL * *((unsigned int *)v43 + 2);
            if ( v126 == *v43 )
              goto LABEL_46;
            v119 = v35;
            v44 = (__int64 *)*v43;
            v115 = v3;
            v45 = v3 + 392;
            v46 = v3 + 72;
            do
            {
              v47 = *v44++;
              v134.m128i_i64[1] = v47;
              v134.m128i_i64[0] = (__int64)v129;
              *sub_26CC870(v46, v134.m128i_i64) = 0;
              sub_26DA910((__int64)&v135, v45, &v134, v48, v49, v50);
            }
            while ( (__int64 *)v126 != v44 );
          }
          else
          {
            v95 = sub_26CA180(v3 + 1024, (__int64 *)&v129);
            v39 = *v95;
            v128 = *v95 + 8LL * *((unsigned int *)v95 + 2);
            if ( v128 == *v95 )
              goto LABEL_46;
            v119 = v35;
            v96 = (__int64 *)*v95;
            v115 = v3;
            v97 = v3 + 392;
            v98 = v3 + 72;
            do
            {
              v99 = *v96++;
              v134.m128i_i64[0] = v99;
              v134.m128i_i64[1] = (__int64)v129;
              *sub_26CC870(v98, v134.m128i_i64) = 0;
              sub_26DA910((__int64)&v135, v97, &v134, v100, v101, v102);
            }
            while ( (__int64 *)v128 != v96 );
          }
          v35 = v119;
          v3 = v115;
          goto LABEL_46;
        }
LABEL_96:
        if ( !v132.m128i_i64[0] )
          goto LABEL_46;
        v36 = v130;
        if ( !*(_BYTE *)(v3 + 132) )
        {
          if ( sub_C8CA60(v113, v130) )
            goto LABEL_99;
          goto LABEL_46;
        }
        v39 = *(_QWORD *)(v3 + 112);
        v38 = (__int64 *)(v39 + 8LL * *(unsigned int *)(v3 + 124));
        v40 = *(unsigned int *)(v3 + 124);
        if ( (__int64 *)v39 != v38 )
        {
          v76 = *(__int64 **)(v3 + 112);
LABEL_105:
          while ( v36 != *(_QWORD *)v39 )
          {
            v39 += 8;
            if ( v38 == (__int64 *)v39 )
              goto LABEL_145;
          }
LABEL_99:
          v77 = v3 + 72;
          v78 = *sub_26CC460(v3 + 40, (__int64 *)&v129);
          if ( v78 < v121 )
            *sub_26CC870(v77, v132.m128i_i64) = 0;
          else
            *sub_26CC870(v77, v132.m128i_i64) = v78 - v121;
          sub_26DA910((__int64)&v135, v3 + 392, &v132, v79, v80, v81);
          v122 = 1;
LABEL_46:
          if ( !v35 )
            goto LABEL_47;
LABEL_83:
          v36 = v130;
          if ( *(_BYTE *)(v3 + 132) )
          {
            v37 = *(__int64 **)(v3 + 112);
            v41 = *(unsigned int *)(v3 + 124);
            goto LABEL_85;
          }
LABEL_107:
          sub_C8CC70(v113, v36, (__int64)v38, v39, v40, v41);
          if ( !v82 )
            goto LABEL_47;
LABEL_91:
          v122 = 1;
          *sub_26CC460(v3 + 40, &v130) = v121;
          goto LABEL_47;
        }
LABEL_144:
        v76 = v38;
LABEL_145:
        if ( !v35 )
          goto LABEL_47;
LABEL_88:
        while ( v76 != v38 )
        {
          if ( v36 == *v76 )
            goto LABEL_47;
          ++v76;
        }
        if ( (unsigned int)v40 >= *(_DWORD *)(v3 + 120) )
          goto LABEL_107;
        *(_DWORD *)(v3 + 124) = v40 + 1;
        *v38 = v36;
        ++*(_QWORD *)(v3 + 104);
        goto LABEL_91;
      }
      v37 = *(__int64 **)(v3 + 112);
      v38 = &v37[*(unsigned int *)(v3 + 124)];
      v39 = (__int64)v37;
      v40 = *(unsigned int *)(v3 + 124);
      v41 = v40;
      if ( v37 == v38 )
      {
        if ( !v132.m128i_i64[0] )
          goto LABEL_142;
        goto LABEL_144;
      }
      v42 = *(__int64 **)(v3 + 112);
      do
      {
        if ( v130 == *v42 )
          goto LABEL_40;
        ++v42;
      }
      while ( v38 != v42 );
      v76 = *(__int64 **)(v3 + 112);
      if ( v132.m128i_i64[0] )
        goto LABEL_105;
LABEL_142:
      if ( v35 )
      {
LABEL_85:
        v76 = v37;
        v40 = (unsigned int)v41;
        v38 = &v37[(unsigned int)v41];
        goto LABEL_88;
      }
LABEL_47:
      if ( i == 1 )
        break;
    }
    v117 = *(_QWORD *)(v117 + 8);
  }
  while ( v112 != v117 );
  return v122;
}
