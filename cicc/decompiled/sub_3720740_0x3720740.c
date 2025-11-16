// Function: sub_3720740
// Address: 0x3720740
//
__int64 __fastcall sub_3720740(_QWORD *a1, __int64 a2, __m128i si128, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rdi
  int v12; // ebx
  __int64 v13; // r14
  unsigned int v14; // eax
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 *v18; // rax
  int v19; // r13d
  __int64 *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 result; // rax
  __int64 v25; // rbx
  __int64 v26; // r9
  unsigned int v27; // edi
  __int64 *v28; // rax
  __int64 v29; // r12
  int v30; // r14d
  unsigned int v31; // esi
  int v32; // eax
  int v33; // edi
  __int64 v34; // rsi
  unsigned int v35; // eax
  __int64 v36; // rbx
  __int64 i; // r12
  _QWORD *v38; // rsi
  _QWORD *j; // rbx
  _QWORD *v40; // rdi
  int v41; // r11d
  int v42; // eax
  int v43; // eax
  int v44; // eax
  __int64 v45; // rdi
  unsigned int v46; // r13d
  int v47; // r10d
  __int64 v48; // rsi
  __int64 v49; // r14
  __int64 v50; // rcx
  __int64 v51; // rdi
  unsigned int v52; // esi
  __int64 v53; // r9
  __int64 v54; // r8
  __int64 *v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 **v60; // rbx
  int v61; // r12d
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rbx
  __int64 v66; // rsi
  unsigned __int8 *v67; // rsi
  int v68; // esi
  int v69; // esi
  __int64 v70; // rdx
  int v71; // eax
  __int64 v72; // rdi
  int v73; // r11d
  __int64 v74; // r10
  unsigned int v75; // edi
  __int64 *v76; // rax
  int v77; // r14d
  __int64 v78; // r12
  unsigned int v79; // esi
  int v80; // eax
  int v81; // esi
  __int64 v82; // rdi
  unsigned int v83; // eax
  int v84; // r11d
  int v85; // eax
  int v86; // eax
  int v87; // eax
  __int64 v88; // rdi
  unsigned int v89; // r13d
  int v90; // r10d
  __int64 v91; // rsi
  int v92; // r13d
  int v93; // eax
  int v94; // esi
  int v95; // esi
  int v96; // r11d
  __int64 v97; // rdx
  __int64 v98; // rdi
  __int64 v99; // rax
  __m128 *v100; // rdx
  __int64 v101; // r12
  _BYTE *v102; // rax
  int v103; // r11d
  __int64 v104; // r10
  int v105; // r13d
  __int64 v106; // r11
  __int128 v107; // [rsp-20h] [rbp-2F0h]
  __int64 v108; // [rsp+8h] [rbp-2C8h]
  __int64 v109; // [rsp+10h] [rbp-2C0h]
  __int64 v110; // [rsp+18h] [rbp-2B8h]
  unsigned int v111; // [rsp+20h] [rbp-2B0h]
  __int64 v112; // [rsp+28h] [rbp-2A8h]
  __int64 v113; // [rsp+30h] [rbp-2A0h]
  __int64 v114; // [rsp+38h] [rbp-298h]
  __int64 v115; // [rsp+38h] [rbp-298h]
  __int64 v116; // [rsp+40h] [rbp-290h]
  __int64 v117; // [rsp+48h] [rbp-288h]
  __int64 v118; // [rsp+48h] [rbp-288h]
  __int64 v119; // [rsp+48h] [rbp-288h]
  __int64 v120; // [rsp+48h] [rbp-288h]
  __int64 v121; // [rsp+48h] [rbp-288h]
  __int64 **v122; // [rsp+58h] [rbp-278h] BYREF
  __int64 v123[2]; // [rsp+60h] [rbp-270h] BYREF
  _QWORD v124[2]; // [rsp+70h] [rbp-260h] BYREF
  int v125; // [rsp+80h] [rbp-250h]
  __int64 *v126; // [rsp+90h] [rbp-240h] BYREF
  __int64 v127; // [rsp+98h] [rbp-238h]
  __int64 v128[2]; // [rsp+A0h] [rbp-230h] BYREF
  char v129; // [rsp+B0h] [rbp-220h]
  char v130; // [rsp+B1h] [rbp-21Fh]

  if ( (_BYTE)qword_5050A28 )
  {
    v99 = sub_C5F790((__int64)a1, a2);
    v100 = *(__m128 **)(v99 + 32);
    v101 = v99;
    if ( *(_QWORD *)(v99 + 24) - (_QWORD)v100 <= 0x16u )
    {
      v101 = sub_CB6200(v99, "IR Module before CSSA:\n", 0x17u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42E2030);
      v100[1].m128_i32[0] = 1397965600;
      v100[1].m128_i16[2] = 14913;
      v100[1].m128_i8[6] = 10;
      *v100 = (__m128)si128;
      *(_QWORD *)(v99 + 32) += 23LL;
    }
    sub_A69980(*(__int64 (__fastcall ***)())(*a1 + 40LL), v101, 0, 0, 0, si128);
    v102 = *(_BYTE **)(v101 + 32);
    if ( *(_BYTE **)(v101 + 24) == v102 )
    {
      sub_CB6200(v101, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v102 = 10;
      ++*(_QWORD *)(v101 + 32);
    }
  }
  v8 = a1[1];
  if ( *(_BYTE *)(v8 + 112) )
  {
    *(_DWORD *)(v8 + 116) = 0;
  }
  else
  {
    HIDWORD(v127) = 32;
    v126 = v128;
    v9 = *(_QWORD *)(v8 + 96);
    if ( v9 )
    {
      v10 = *(_QWORD *)(v9 + 24);
      v11 = v128;
      v12 = 1;
      v13 = v8;
      v128[0] = *(_QWORD *)(v8 + 96);
      v128[1] = v10;
      LODWORD(v127) = 1;
      *(_DWORD *)(v9 + 72) = 0;
      v14 = 1;
      do
      {
        while ( 1 )
        {
          v19 = v12++;
          a5 = (__int64)&v11[2 * v14 - 2];
          v8 = *(_QWORD *)a5;
          v20 = *(__int64 **)(a5 + 8);
          a7 = *(_QWORD *)(*(_QWORD *)a5 + 24LL) + 8LL * *(unsigned int *)(*(_QWORD *)a5 + 32LL);
          if ( v20 != (__int64 *)a7 )
            break;
          --v14;
          *(_DWORD *)(v8 + 76) = v19;
          LODWORD(v127) = v14;
          if ( !v14 )
            goto LABEL_10;
        }
        v15 = *v20;
        *(_QWORD *)(a5 + 8) = v20 + 1;
        v16 = (unsigned int)v127;
        a5 = HIDWORD(v127);
        v8 = *(_QWORD *)(v15 + 24);
        v17 = (unsigned int)v127 + 1LL;
        if ( v17 > HIDWORD(v127) )
        {
          v115 = *(_QWORD *)(v15 + 24);
          sub_C8D5F0((__int64)&v126, v128, v17, 0x10u, v8, a7);
          v11 = v126;
          v16 = (unsigned int)v127;
          v8 = v115;
        }
        v18 = &v11[2 * v16];
        *v18 = v15;
        v18[1] = v8;
        LODWORD(v127) = v127 + 1;
        v14 = v127;
        *(_DWORD *)(v15 + 72) = v19;
        v11 = v126;
      }
      while ( v14 );
LABEL_10:
      *(_DWORD *)(v13 + 116) = 0;
      *(_BYTE *)(v13 + 112) = 1;
      if ( v11 != v128 )
        _libc_free((unsigned __int64)v11);
    }
  }
  sub_F3ABE0(*a1, 0, 0, a5, v8, a7);
  v22 = (__int64)(a1 + 11);
  v110 = (__int64)(a1 + 11);
  v23 = *(_QWORD *)(*a1 + 80LL);
  result = *a1 + 72LL;
  v109 = result;
  v113 = v23;
  if ( v23 == result )
    goto LABEL_34;
  do
  {
    if ( !v113 )
      BUG();
    v25 = *(_QWORD *)(v113 + 32);
    v26 = v113 + 24;
    v116 = v113 + 24;
    if ( v25 == v113 + 24 )
      goto LABEL_29;
    do
    {
      while ( 1 )
      {
        if ( !v25 )
LABEL_186:
          BUG();
        v29 = v25 - 24;
        if ( *(_BYTE *)(v25 - 24) != 84 )
        {
          v114 = v25 - 24;
          if ( *(_QWORD *)(v113 + 32) == v25 )
            goto LABEL_83;
LABEL_53:
          sub_371F160((__int64)v124, (__int64)a1, v113 - 24, v114, 1);
          v49 = *(_QWORD *)(v113 + 32);
          v112 = v124[0];
          if ( v49 == v116 )
            goto LABEL_83;
          v108 = v25;
          while ( 1 )
          {
            if ( !v49 )
              goto LABEL_186;
            if ( *(_BYTE *)(v49 - 24) != 84 )
              goto LABEL_82;
            v60 = *(__int64 ***)(v49 - 16);
            v130 = 1;
            v129 = 3;
            v61 = v125;
            v119 = v124[1];
            v126 = (__int64 *)"pcp";
            v123[0] = v112;
            v62 = sub_ACA8A0(v60);
            v122 = v60;
            v123[1] = v62;
            v63 = sub_371CDC0(0x22D7u, (__int64)&v122, 1, v123, 2, (__int64)&v126, v114);
            v64 = *(_QWORD *)(v49 + 24);
            v65 = v63;
            v126 = (__int64 *)v64;
            if ( !v64 )
              break;
            sub_B96E90((__int64)&v126, v64, 1);
            v51 = v65 + 48;
            if ( (__int64 **)(v65 + 48) == &v126 )
            {
              if ( v126 )
                sub_B91220(v51, (__int64)v126);
LABEL_58:
              v52 = *((_DWORD *)a1 + 28);
              if ( !v52 )
                goto LABEL_74;
              goto LABEL_59;
            }
            v66 = *(_QWORD *)(v65 + 48);
            if ( v66 )
              goto LABEL_71;
LABEL_72:
            v67 = (unsigned __int8 *)v126;
            *(_QWORD *)(v65 + 48) = v126;
            if ( !v67 )
              goto LABEL_58;
            sub_B976B0((__int64)&v126, v67, v51);
            v52 = *((_DWORD *)a1 + 28);
            if ( !v52 )
            {
LABEL_74:
              ++a1[11];
              goto LABEL_75;
            }
LABEL_59:
            v53 = a1[12];
            v54 = (v52 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
            v55 = (__int64 *)(v53 + 16 * v54);
            v56 = *v55;
            if ( v65 != *v55 )
            {
              v92 = 1;
              v50 = 0;
              while ( v56 != -4096 )
              {
                if ( v50 || v56 != -8192 )
                  v55 = (__int64 *)v50;
                v50 = (unsigned int)(v92 + 1);
                v54 = (v52 - 1) & (v92 + (_DWORD)v54);
                v56 = *(_QWORD *)(v53 + 16LL * (unsigned int)v54);
                if ( v65 == v56 )
                  goto LABEL_60;
                ++v92;
                v50 = (__int64)v55;
                v55 = (__int64 *)(v53 + 16LL * (unsigned int)v54);
              }
              if ( !v50 )
                v50 = (__int64)v55;
              v93 = *((_DWORD *)a1 + 26);
              ++a1[11];
              v71 = v93 + 1;
              if ( 4 * v71 >= 3 * v52 )
              {
LABEL_75:
                sub_A41E30(v110, 2 * v52);
                v68 = *((_DWORD *)a1 + 28);
                if ( !v68 )
                  goto LABEL_185;
                v69 = v68 - 1;
                v54 = a1[12];
                LODWORD(v70) = v69 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
                v71 = *((_DWORD *)a1 + 26) + 1;
                v50 = v54 + 16LL * (unsigned int)v70;
                v72 = *(_QWORD *)v50;
                if ( *(_QWORD *)v50 != v65 )
                {
                  v73 = 1;
                  v74 = 0;
                  while ( v72 != -4096 )
                  {
                    if ( v72 == -8192 && !v74 )
                      v74 = v50;
                    v53 = (unsigned int)(v73 + 1);
                    v70 = v69 & (unsigned int)(v70 + v73);
                    v50 = v54 + 16 * v70;
                    v72 = *(_QWORD *)v50;
                    if ( v65 == *(_QWORD *)v50 )
                      goto LABEL_116;
                    ++v73;
                  }
                  goto LABEL_79;
                }
              }
              else
              {
                v54 = v52 >> 3;
                if ( v52 - *((_DWORD *)a1 + 27) - v71 <= (unsigned int)v54 )
                {
                  v111 = ((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4);
                  sub_A41E30(v110, v52);
                  v94 = *((_DWORD *)a1 + 28);
                  if ( !v94 )
                    goto LABEL_185;
                  v95 = v94 - 1;
                  v54 = a1[12];
                  v74 = 0;
                  v96 = 1;
                  LODWORD(v97) = v95 & v111;
                  v71 = *((_DWORD *)a1 + 26) + 1;
                  v50 = v54 + 16LL * (v95 & v111);
                  v98 = *(_QWORD *)v50;
                  if ( v65 != *(_QWORD *)v50 )
                  {
                    while ( v98 != -4096 )
                    {
                      if ( !v74 && v98 == -8192 )
                        v74 = v50;
                      v53 = (unsigned int)(v96 + 1);
                      v97 = v95 & (unsigned int)(v97 + v96);
                      v50 = v54 + 16 * v97;
                      v98 = *(_QWORD *)v50;
                      if ( v65 == *(_QWORD *)v50 )
                        goto LABEL_116;
                      ++v96;
                    }
LABEL_79:
                    if ( v74 )
                      v50 = v74;
                  }
                }
              }
LABEL_116:
              *((_DWORD *)a1 + 26) = v71;
              if ( *(_QWORD *)v50 != -4096 )
                --*((_DWORD *)a1 + 27);
              *(_QWORD *)v50 = v65;
              *(_DWORD *)(v50 + 8) = v61;
            }
LABEL_60:
            LODWORD(v128[0]) = v61;
            v127 = v119;
            *((_QWORD *)&v107 + 1) = v119;
            *(_QWORD *)&v107 = v65;
            v126 = (__int64 *)v65;
            sub_371EDF0((__int64)a1, v65, 1, v50, v54, v53, v107, v128[0]);
            sub_BD84D0(v49 - 24, v65);
            v57 = 32 * (1LL - (*(_DWORD *)(v65 + 4) & 0x7FFFFFF)) + v65;
            if ( *(_QWORD *)v57 )
            {
              v58 = *(_QWORD *)(v57 + 8);
              **(_QWORD **)(v57 + 16) = v58;
              if ( v58 )
                *(_QWORD *)(v58 + 16) = *(_QWORD *)(v57 + 16);
            }
            *(_QWORD *)v57 = v49 - 24;
            v59 = *(_QWORD *)(v49 - 8);
            v22 = v49 - 8;
            *(_QWORD *)(v57 + 8) = v59;
            if ( v59 )
            {
              v23 = v57 + 8;
              *(_QWORD *)(v59 + 16) = v57 + 8;
            }
            *(_QWORD *)(v57 + 16) = v22;
            *(_QWORD *)(v49 - 8) = v57;
            v49 = *(_QWORD *)(v49 + 8);
            if ( v49 == v116 )
            {
LABEL_82:
              v25 = v108;
LABEL_83:
              if ( v25 == v116 )
                goto LABEL_29;
              v26 = v113 + 24;
              while ( 2 )
              {
                v77 = *((_DWORD *)a1 + 30);
                v78 = v25 - 24;
                v79 = *((_DWORD *)a1 + 28);
                if ( !v25 )
                  v78 = 0;
                *((_DWORD *)a1 + 30) = v77 + 1;
                if ( !v79 )
                {
                  ++a1[11];
                  goto LABEL_91;
                }
                v21 = a1[12];
                v75 = (v79 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
                v76 = (__int64 *)(v21 + 16LL * v75);
                v23 = *v76;
                if ( v78 != *v76 )
                {
                  v84 = 1;
                  v22 = 0;
                  while ( v23 != -4096 )
                  {
                    if ( v23 != -8192 || v22 )
                      v76 = (__int64 *)v22;
                    v22 = (unsigned int)(v84 + 1);
                    v75 = (v79 - 1) & (v84 + v75);
                    v23 = *(_QWORD *)(v21 + 16LL * v75);
                    if ( v78 == v23 )
                      goto LABEL_86;
                    ++v84;
                    v22 = (__int64)v76;
                    v76 = (__int64 *)(v21 + 16LL * v75);
                  }
                  if ( !v22 )
                    v22 = (__int64)v76;
                  v85 = *((_DWORD *)a1 + 26);
                  ++a1[11];
                  v23 = (unsigned int)(v85 + 1);
                  if ( 4 * (int)v23 >= 3 * v79 )
                  {
LABEL_91:
                    v120 = v26;
                    sub_A41E30(v110, 2 * v79);
                    v80 = *((_DWORD *)a1 + 28);
                    if ( !v80 )
                      goto LABEL_185;
                    v81 = v80 - 1;
                    v82 = a1[12];
                    v26 = v120;
                    v83 = (v80 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
                    v23 = (unsigned int)(*((_DWORD *)a1 + 26) + 1);
                    v22 = v82 + 16LL * v83;
                    v21 = *(_QWORD *)v22;
                    if ( v78 != *(_QWORD *)v22 )
                    {
                      v105 = 1;
                      v106 = 0;
                      while ( v21 != -4096 )
                      {
                        if ( v21 != -8192 || v106 )
                          v22 = v106;
                        v83 = v81 & (v105 + v83);
                        v21 = *(_QWORD *)(v82 + 16LL * v83);
                        if ( v78 == v21 )
                        {
                          v22 = v82 + 16LL * v83;
                          goto LABEL_93;
                        }
                        ++v105;
                        v106 = v22;
                        v22 = v82 + 16LL * v83;
                      }
                      if ( v106 )
                        v22 = v106;
                    }
                  }
                  else if ( v79 - *((_DWORD *)a1 + 27) - (unsigned int)v23 <= v79 >> 3 )
                  {
                    v121 = v26;
                    sub_A41E30(v110, v79);
                    v86 = *((_DWORD *)a1 + 28);
                    if ( !v86 )
                      goto LABEL_185;
                    v87 = v86 - 1;
                    v88 = a1[12];
                    v21 = 0;
                    v89 = v87 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
                    v26 = v121;
                    v90 = 1;
                    v23 = (unsigned int)(*((_DWORD *)a1 + 26) + 1);
                    v22 = v88 + 16LL * v89;
                    v91 = *(_QWORD *)v22;
                    if ( v78 != *(_QWORD *)v22 )
                    {
                      while ( v91 != -4096 )
                      {
                        if ( v91 == -8192 && !v21 )
                          v21 = v22;
                        v89 = v87 & (v90 + v89);
                        v22 = v88 + 16LL * v89;
                        v91 = *(_QWORD *)v22;
                        if ( v78 == *(_QWORD *)v22 )
                          goto LABEL_93;
                        ++v90;
                      }
                      if ( v21 )
                        v22 = v21;
                    }
                  }
LABEL_93:
                  *((_DWORD *)a1 + 26) = v23;
                  if ( *(_QWORD *)v22 != -4096 )
                    --*((_DWORD *)a1 + 27);
                  *(_QWORD *)v22 = v78;
                  *(_DWORD *)(v22 + 8) = v77;
                }
LABEL_86:
                v25 = *(_QWORD *)(v25 + 8);
                if ( v25 == v26 )
                  goto LABEL_29;
                continue;
              }
            }
          }
          v51 = v63 + 48;
          if ( (__int64 **)(v63 + 48) == &v126 )
            goto LABEL_58;
          v66 = *(_QWORD *)(v63 + 48);
          if ( !v66 )
            goto LABEL_58;
LABEL_71:
          sub_B91220(v51, v66);
          goto LABEL_72;
        }
        v30 = *((_DWORD *)a1 + 30);
        v31 = *((_DWORD *)a1 + 28);
        *((_DWORD *)a1 + 30) = v30 + 1;
        if ( !v31 )
        {
          ++a1[11];
LABEL_22:
          v117 = v26;
          sub_A41E30(v110, 2 * v31);
          v32 = *((_DWORD *)a1 + 28);
          if ( !v32 )
            goto LABEL_185;
          v33 = v32 - 1;
          v34 = a1[12];
          v26 = v117;
          v35 = (v32 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v23 = (unsigned int)(*((_DWORD *)a1 + 26) + 1);
          v22 = v34 + 16LL * v35;
          v21 = *(_QWORD *)v22;
          if ( v29 != *(_QWORD *)v22 )
          {
            v103 = 1;
            v104 = 0;
            while ( v21 != -4096 )
            {
              if ( v21 == -8192 && !v104 )
                v104 = v22;
              v35 = v33 & (v103 + v35);
              v22 = v34 + 16LL * v35;
              v21 = *(_QWORD *)v22;
              if ( v29 == *(_QWORD *)v22 )
                goto LABEL_24;
              ++v103;
            }
            if ( v104 )
              v22 = v104;
          }
          goto LABEL_24;
        }
        v21 = a1[12];
        v27 = (v31 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v28 = (__int64 *)(v21 + 16LL * v27);
        v23 = *v28;
        if ( v29 != *v28 )
          break;
LABEL_17:
        v25 = *(_QWORD *)(v25 + 8);
        if ( v25 == v26 )
          goto LABEL_27;
      }
      v41 = 1;
      v22 = 0;
      while ( v23 != -4096 )
      {
        if ( v22 || v23 != -8192 )
          v28 = (__int64 *)v22;
        v22 = (unsigned int)(v41 + 1);
        v27 = (v31 - 1) & (v41 + v27);
        v23 = *(_QWORD *)(v21 + 16LL * v27);
        if ( v29 == v23 )
          goto LABEL_17;
        ++v41;
        v22 = (__int64)v28;
        v28 = (__int64 *)(v21 + 16LL * v27);
      }
      if ( !v22 )
        v22 = (__int64)v28;
      v42 = *((_DWORD *)a1 + 26);
      ++a1[11];
      v23 = (unsigned int)(v42 + 1);
      if ( 4 * (int)v23 >= 3 * v31 )
        goto LABEL_22;
      if ( v31 - *((_DWORD *)a1 + 27) - (unsigned int)v23 <= v31 >> 3 )
      {
        v118 = v26;
        sub_A41E30(v110, v31);
        v43 = *((_DWORD *)a1 + 28);
        if ( !v43 )
        {
LABEL_185:
          ++*((_DWORD *)a1 + 26);
          BUG();
        }
        v44 = v43 - 1;
        v45 = a1[12];
        v21 = 0;
        v46 = v44 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v26 = v118;
        v47 = 1;
        v23 = (unsigned int)(*((_DWORD *)a1 + 26) + 1);
        v22 = v45 + 16LL * v46;
        v48 = *(_QWORD *)v22;
        if ( v29 != *(_QWORD *)v22 )
        {
          while ( v48 != -4096 )
          {
            if ( !v21 && v48 == -8192 )
              v21 = v22;
            v46 = v44 & (v47 + v46);
            v22 = v45 + 16LL * v46;
            v48 = *(_QWORD *)v22;
            if ( v29 == *(_QWORD *)v22 )
              goto LABEL_24;
            ++v47;
          }
          if ( v21 )
            v22 = v21;
        }
      }
LABEL_24:
      *((_DWORD *)a1 + 26) = v23;
      if ( *(_QWORD *)v22 != -4096 )
        --*((_DWORD *)a1 + 27);
      *(_QWORD *)v22 = v29;
      *(_DWORD *)(v22 + 8) = v30;
      v25 = *(_QWORD *)(v25 + 8);
    }
    while ( v25 != v26 );
LABEL_27:
    if ( v25 != *(_QWORD *)(v113 + 32) )
    {
      v114 = v113;
      goto LABEL_53;
    }
LABEL_29:
    result = *(_QWORD *)(v113 + 8);
    v113 = result;
  }
  while ( v109 != result );
  v36 = *(_QWORD *)(*a1 + 80LL);
  for ( i = *a1 + 72LL; i != v36; v36 = *(_QWORD *)(v36 + 8) )
  {
    v38 = (_QWORD *)(v36 - 24);
    if ( !v36 )
      v38 = 0;
    result = sub_371F790((__int64)a1, v38, v22, v23, v21, v26);
  }
LABEL_34:
  for ( j = (_QWORD *)a1[5]; j; j = (_QWORD *)*j )
  {
    while ( 1 )
    {
      v40 = (_QWORD *)j[1];
      if ( !v40[2] )
        break;
      j = (_QWORD *)*j;
      if ( !j )
        return result;
    }
    result = sub_B43D60(v40);
  }
  return result;
}
