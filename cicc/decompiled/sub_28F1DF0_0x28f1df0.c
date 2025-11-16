// Function: sub_28F1DF0
// Address: 0x28f1df0
//
__int64 __fastcall sub_28F1DF0(unsigned __int8 *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  int v4; // r15d
  __int64 v5; // rax
  unsigned __int8 *v6; // r9
  unsigned int v7; // ebx
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int8 *v10; // r13
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r12
  unsigned __int8 *v15; // r14
  unsigned int v16; // eax
  unsigned __int8 *v17; // rdx
  unsigned __int8 *v18; // rbx
  __int64 v19; // r8
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned __int8 **v22; // rax
  unsigned __int8 *v23; // r8
  unsigned int v24; // ecx
  unsigned __int8 *v25; // rax
  unsigned __int8 *v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // esi
  unsigned __int8 *v31; // rcx
  int v32; // r11d
  unsigned int v33; // edx
  unsigned __int8 *v34; // rax
  unsigned __int8 *v35; // r10
  unsigned __int8 **v36; // rax
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned __int8 *v41; // rcx
  unsigned __int8 *v42; // rdi
  int v43; // r11d
  unsigned int v44; // edx
  unsigned __int8 *v45; // r8
  int v46; // edx
  __int64 *v47; // r12
  _BYTE *v48; // r8
  __int64 *v49; // r14
  __int64 v50; // r13
  unsigned int v51; // ecx
  unsigned __int8 *v52; // rax
  __int64 v53; // r10
  __int64 v54; // r10
  __int64 v55; // rax
  __int64 *v56; // rax
  char v57; // al
  __int64 v59; // rax
  __int64 v60; // r9
  __int64 v61; // r8
  __int64 v62; // rbx
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  __int64 *v65; // rax
  __int64 v66; // rbx
  __int64 v67; // r12
  char v68; // al
  int v69; // eax
  int v70; // eax
  int v71; // ecx
  __int64 *v72; // rax
  char v73; // al
  int v74; // r10d
  unsigned int v75; // eax
  int v76; // r10d
  int v77; // r10d
  unsigned int v78; // eax
  unsigned __int8 *v79; // rbx
  __int64 v80; // r8
  __int64 v81; // r9
  unsigned __int64 v82; // rcx
  unsigned __int64 v83; // rax
  int v84; // edx
  unsigned __int8 **v85; // rax
  __int64 v86; // rdx
  int v87; // eax
  unsigned int v88; // esi
  unsigned __int8 v89; // di
  bool v90; // al
  int v91; // edx
  unsigned __int8 **v92; // rax
  unsigned __int8 *v93; // rdx
  unsigned __int8 v94; // cl
  _BYTE *v95; // rdx
  unsigned __int8 **v96; // rax
  void **v97; // rax
  unsigned __int8 *v98; // rdx
  void **v99; // rcx
  char v100; // r8
  unsigned int v101; // ebx
  unsigned __int8 *v102; // r12
  void **v103; // rax
  void **v104; // rcx
  char v105; // al
  _BYTE *v106; // rcx
  void **v107; // [rsp+8h] [rbp-1F8h]
  char v108; // [rsp+13h] [rbp-1EDh]
  int v109; // [rsp+14h] [rbp-1ECh]
  __int64 v110; // [rsp+18h] [rbp-1E8h]
  unsigned __int8 *v111; // [rsp+18h] [rbp-1E8h]
  __int64 v112; // [rsp+18h] [rbp-1E8h]
  __int64 v113; // [rsp+18h] [rbp-1E8h]
  __int64 v116; // [rsp+30h] [rbp-1D0h]
  unsigned __int8 v117; // [rsp+38h] [rbp-1C8h]
  __int64 v118; // [rsp+38h] [rbp-1C8h]
  _BYTE *v121; // [rsp+58h] [rbp-1A8h]
  _BYTE *v122; // [rsp+60h] [rbp-1A0h]
  __int64 v123; // [rsp+60h] [rbp-1A0h]
  unsigned __int8 *v124; // [rsp+68h] [rbp-198h]
  unsigned __int8 *v125; // [rsp+68h] [rbp-198h]
  unsigned __int8 *v126; // [rsp+68h] [rbp-198h]
  unsigned __int8 *v127; // [rsp+68h] [rbp-198h]
  __int64 v128; // [rsp+68h] [rbp-198h]
  unsigned __int8 *v129; // [rsp+68h] [rbp-198h]
  unsigned __int8 *v130; // [rsp+68h] [rbp-198h]
  void **v131; // [rsp+68h] [rbp-198h]
  unsigned __int8 *v132; // [rsp+78h] [rbp-188h] BYREF
  __int64 v133; // [rsp+80h] [rbp-180h] BYREF
  unsigned __int8 *v134; // [rsp+88h] [rbp-178h]
  __int64 v135; // [rsp+90h] [rbp-170h]
  unsigned int v136; // [rsp+98h] [rbp-168h]
  __m128i v137; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v138; // [rsp+B0h] [rbp-150h]
  __int64 v139; // [rsp+B8h] [rbp-148h]
  __int64 v140; // [rsp+C0h] [rbp-140h]
  __int64 v141; // [rsp+C8h] [rbp-138h]
  __int64 v142; // [rsp+D0h] [rbp-130h]
  __int64 v143; // [rsp+D8h] [rbp-128h]
  __int16 v144; // [rsp+E0h] [rbp-120h]
  __int64 *v145; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v146; // [rsp+F8h] [rbp-108h]
  _BYTE v147[64]; // [rsp+100h] [rbp-100h] BYREF
  _QWORD *v148; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v149; // [rsp+148h] [rbp-B8h]
  _QWORD v150[22]; // [rsp+150h] [rbp-B0h] BYREF

  v4 = *a1 - 29;
  v148 = v150;
  v149 = 0x800000001LL;
  v145 = (__int64 *)v147;
  v150[0] = a1;
  v150[1] = 1;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v146 = 0x800000000LL;
  v5 = sub_B43CC0((__int64)a1);
  v117 = 0;
  v7 = v149;
  v116 = v5;
LABEL_2:
  if ( !v7 )
    goto LABEL_46;
  do
  {
    v8 = v7--;
    v9 = (unsigned __int64)&v148[2 * v8 - 2];
    v10 = *(unsigned __int8 **)v9;
    v11 = *(unsigned __int8 **)(v9 + 8);
    LODWORD(v149) = v7;
    v124 = v11;
    v12 = *v10;
    if ( (unsigned __int8)v12 <= 0x36u )
    {
      v13 = 0x40540000000000LL;
      if ( _bittest64(&v13, v12) )
      {
        *a4 &= sub_B448F0((__int64)v10);
        a4[1] &= sub_B44900((__int64)v10);
      }
    }
    v14 = 0;
    v15 = v124;
    v16 = *((_DWORD *)v10 + 1) & 0x7FFFFFF;
    if ( !v16 )
      goto LABEL_2;
    do
    {
      if ( (v10[7] & 0x40) != 0 )
        v17 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      else
        v17 = &v10[-32 * v16];
      v132 = *(unsigned __int8 **)&v17[32 * v14];
      v18 = sub_28ED300(v132, v4);
      if ( v18 )
      {
        v20 = (unsigned int)v149;
        v21 = (unsigned int)v149 + 1LL;
        if ( v21 > HIDWORD(v149) )
        {
          sub_C8D5F0((__int64)&v148, v150, v21, 0x10u, v19, (__int64)v6);
          v20 = (unsigned int)v149;
        }
        v22 = (unsigned __int8 **)&v148[2 * v20];
        *v22 = v18;
        v22[1] = v15;
        LODWORD(v149) = v149 + 1;
        goto LABEL_13;
      }
      v23 = v132;
      if ( v136 )
      {
        v24 = (v136 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
        v25 = &v134[16 * v24];
        v6 = *(unsigned __int8 **)v25;
        if ( v132 == *(unsigned __int8 **)v25 )
        {
LABEL_18:
          if ( v25 != &v134[16 * v136] )
          {
            v26 = &v15[*((_QWORD *)v25 + 1)];
            *((_QWORD *)v25 + 1) = v26;
            v27 = *((_QWORD *)v23 + 2);
            if ( !v27 || *(_QWORD *)(v27 + 8) )
              goto LABEL_13;
            *(_QWORD *)v25 = -8192;
            v23 = v132;
            v15 = v26;
            LODWORD(v135) = v135 - 1;
            ++HIDWORD(v135);
            goto LABEL_22;
          }
        }
        else
        {
          v37 = 1;
          while ( v6 != (unsigned __int8 *)-4096LL )
          {
            v74 = v37 + 1;
            v24 = (v136 - 1) & (v37 + v24);
            v25 = &v134[16 * v24];
            v6 = *(unsigned __int8 **)v25;
            if ( v132 == *(unsigned __int8 **)v25 )
              goto LABEL_18;
            v37 = v74;
          }
        }
      }
      v38 = *((_QWORD *)v132 + 2);
      if ( !v38 || *(_QWORD *)(v38 + 8) )
      {
        v39 = (unsigned int)v146;
        v40 = (unsigned int)v146 + 1LL;
        if ( v40 > HIDWORD(v146) )
        {
          v126 = v132;
          sub_C8D5F0((__int64)&v145, v147, v40, 8u, (__int64)v132, (__int64)v6);
          v39 = (unsigned int)v146;
          v23 = v126;
        }
        v145[v39] = (__int64)v23;
        LODWORD(v146) = v146 + 1;
        if ( !v136 )
        {
          ++v133;
          goto LABEL_103;
        }
        v41 = v132;
        v6 = v134;
        v42 = 0;
        v43 = 1;
        v44 = (v136 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
        v34 = &v134[16 * v44];
        v45 = *(unsigned __int8 **)v34;
        if ( *(unsigned __int8 **)v34 != v132 )
        {
          while ( v45 != (unsigned __int8 *)-4096LL )
          {
            if ( v45 == (unsigned __int8 *)-8192LL && !v42 )
              v42 = v34;
            v44 = (v136 - 1) & (v43 + v44);
            v34 = &v134[16 * v44];
            v45 = *(unsigned __int8 **)v34;
            if ( v132 == *(unsigned __int8 **)v34 )
              goto LABEL_28;
            ++v43;
          }
          if ( !v42 )
            v42 = v34;
          ++v133;
          v46 = v135 + 1;
          if ( 4 * ((int)v135 + 1) < 3 * v136 )
          {
            if ( v136 - HIDWORD(v135) - v46 <= v136 >> 3 )
            {
              sub_9BBF00((__int64)&v133, v136);
              if ( !v136 )
              {
LABEL_199:
                LODWORD(v135) = v135 + 1;
                BUG();
              }
              v41 = v132;
              v77 = 1;
              v78 = (v136 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
              v46 = v135 + 1;
              v42 = &v134[16 * v78];
              v6 = *(unsigned __int8 **)v42;
              if ( *(unsigned __int8 **)v42 != v132 )
              {
                while ( v6 != (unsigned __int8 *)-4096LL )
                {
                  if ( !v18 && v6 == (unsigned __int8 *)-8192LL )
                    v18 = v42;
                  v78 = (v136 - 1) & (v77 + v78);
                  v42 = &v134[16 * v78];
                  v6 = *(unsigned __int8 **)v42;
                  if ( v132 == *(unsigned __int8 **)v42 )
                    goto LABEL_42;
                  ++v77;
                }
                goto LABEL_107;
              }
            }
            goto LABEL_42;
          }
LABEL_103:
          sub_9BBF00((__int64)&v133, 2 * v136);
          if ( !v136 )
            goto LABEL_199;
          v41 = v132;
          v75 = (v136 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
          v46 = v135 + 1;
          v42 = &v134[16 * v75];
          v6 = *(unsigned __int8 **)v42;
          if ( *(unsigned __int8 **)v42 != v132 )
          {
            v76 = 1;
            while ( v6 != (unsigned __int8 *)-4096LL )
            {
              if ( v6 == (unsigned __int8 *)-8192LL && !v18 )
                v18 = v42;
              v75 = (v136 - 1) & (v76 + v75);
              v42 = &v134[16 * v75];
              v6 = *(unsigned __int8 **)v42;
              if ( v132 == *(unsigned __int8 **)v42 )
                goto LABEL_42;
              ++v76;
            }
LABEL_107:
            if ( v18 )
              v42 = v18;
          }
LABEL_42:
          LODWORD(v135) = v46;
          if ( *(_QWORD *)v42 != -4096 )
            --HIDWORD(v135);
          *(_QWORD *)v42 = v41;
          v36 = (unsigned __int8 **)(v42 + 8);
          *((_QWORD *)v42 + 1) = 0;
LABEL_29:
          *v36 = v15;
          goto LABEL_13;
        }
LABEL_28:
        v36 = (unsigned __int8 **)(v34 + 8);
        goto LABEL_29;
      }
LABEL_22:
      if ( v4 != 17 )
      {
        if ( v4 == 18 )
        {
          v70 = *v23;
          if ( (unsigned __int8)v70 > 0x1Cu )
          {
            v71 = v70 - 29;
            switch ( *v23 )
            {
              case ')':
              case '+':
              case '-':
              case '/':
              case '2':
              case '5':
              case 'J':
              case 'K':
              case 'S':
                goto LABEL_93;
              case 'T':
              case 'U':
              case 'V':
                v86 = *((_QWORD *)v23 + 1);
                v87 = *(unsigned __int8 *)(v86 + 8);
                v88 = v87 - 17;
                v89 = *(_BYTE *)(v86 + 8);
                if ( (unsigned int)(v87 - 17) <= 1 )
                  v89 = *(_BYTE *)(**(_QWORD **)(v86 + 16) + 8LL);
                if ( v89 <= 3u || v89 == 5 || (v89 & 0xFD) == 4 )
                  goto LABEL_94;
                if ( (_BYTE)v87 == 15 )
                {
                  if ( (*(_BYTE *)(v86 + 9) & 4) == 0 )
                    goto LABEL_24;
                  v111 = v23;
                  v128 = *((_QWORD *)v23 + 1);
                  v90 = sub_BCB420(v128);
                  v23 = v111;
                  if ( !v90 )
                    break;
                  v86 = **(_QWORD **)(v128 + 16);
                  v87 = *(unsigned __int8 *)(v86 + 8);
                  v88 = v87 - 17;
                }
                else if ( (_BYTE)v87 == 16 )
                {
                  do
                  {
                    v86 = *(_QWORD *)(v86 + 24);
                    LOBYTE(v87) = *(_BYTE *)(v86 + 8);
                  }
                  while ( (_BYTE)v87 == 16 );
                  v88 = (unsigned __int8)v87 - 17;
                }
                if ( v88 <= 1 )
                  LOBYTE(v87) = *(_BYTE *)(**(_QWORD **)(v86 + 16) + 8LL);
                if ( (unsigned __int8)v87 > 3u && (_BYTE)v87 != 5 && (v87 & 0xFD) != 4 )
                  break;
                v70 = *v23;
                if ( (unsigned __int8)v70 > 0x1Cu )
LABEL_93:
                  v71 = v70 - 29;
                else
                  v71 = *((unsigned __int16 *)v23 + 1);
LABEL_94:
                if ( v71 == 12 )
                  goto LABEL_72;
                if ( v71 != 16 )
                  break;
                if ( (v23[1] & 0x10) == 0 )
                {
                  v137.m128i_i64[0] = 0;
                  v72 = (__int64 *)sub_986520((__int64)v23);
                  v73 = sub_1008640((__int64 **)&v137, *v72);
                  goto LABEL_98;
                }
                v92 = (unsigned __int8 **)sub_986520((__int64)v23);
                v93 = *v92;
                v94 = **v92;
                if ( v94 == 18 )
                {
                  v129 = *v92;
                  if ( *((void **)v129 + 3) == sub_C33340() )
                    v95 = (_BYTE *)*((_QWORD *)v129 + 4);
                  else
                    v95 = v129 + 24;
                  v73 = (v95[20] & 7) == 3;
                  goto LABEL_98;
                }
                v112 = *((_QWORD *)v93 + 1);
                if ( (unsigned int)*(unsigned __int8 *)(v112 + 8) - 17 > 1 || v94 > 0x15u )
                  break;
                v130 = *v92;
                v97 = (void **)sub_AD7630((__int64)v93, 0, (__int64)v93);
                v98 = v130;
                if ( v97 )
                {
                  v131 = v97;
                  if ( *(_BYTE *)v97 == 18 )
                  {
                    if ( v97[3] == sub_C33340() )
                      v99 = (void **)v131[4];
                    else
                      v99 = v131 + 3;
                    v73 = (*((_BYTE *)v99 + 20) & 7) == 3;
LABEL_98:
                    if ( v73 )
                      goto LABEL_72;
                    break;
                  }
                }
                if ( *(_BYTE *)(v112 + 8) == 17 )
                {
                  v109 = *(_DWORD *)(v112 + 32);
                  if ( v109 )
                  {
                    v100 = 0;
                    v113 = v14;
                    v101 = 0;
                    v102 = v98;
                    do
                    {
                      v108 = v100;
                      v103 = (void **)sub_AD69F0(v102, v101);
                      v104 = v103;
                      if ( !v103 )
                      {
LABEL_197:
                        v18 = 0;
                        v14 = v113;
                        goto LABEL_99;
                      }
                      v105 = *(_BYTE *)v103;
                      v107 = v104;
                      v100 = v108;
                      if ( v105 != 13 )
                      {
                        if ( v105 != 18 )
                          goto LABEL_197;
                        v106 = v104[3] == sub_C33340() ? v107[4] : v107 + 3;
                        if ( (v106[20] & 7) != 3 )
                          goto LABEL_197;
                        v100 = 1;
                      }
                      ++v101;
                    }
                    while ( v109 != v101 );
                    v18 = 0;
                    v14 = v113;
                    if ( v100 )
                      goto LABEL_72;
                  }
                }
                break;
              default:
                goto LABEL_24;
            }
LABEL_99:
            v23 = v132;
          }
        }
LABEL_24:
        v28 = (unsigned int)v146;
        v29 = (unsigned int)v146 + 1LL;
        if ( v29 > HIDWORD(v146) )
        {
          v127 = v23;
          sub_C8D5F0((__int64)&v145, v147, v29, 8u, (__int64)v23, (__int64)v6);
          v28 = (unsigned int)v146;
          v23 = v127;
        }
        v145[v28] = (__int64)v23;
        v30 = v136;
        LODWORD(v146) = v146 + 1;
        if ( v136 )
        {
          v31 = v132;
          v32 = 1;
          v33 = (v136 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
          v34 = &v134[16 * v33];
          v35 = *(unsigned __int8 **)v34;
          if ( v132 == *(unsigned __int8 **)v34 )
            goto LABEL_28;
          while ( v35 != (unsigned __int8 *)-4096LL )
          {
            if ( !v18 && v35 == (unsigned __int8 *)-8192LL )
              v18 = v34;
            v6 = (unsigned __int8 *)(unsigned int)(v32 + 1);
            v33 = (v136 - 1) & (v32 + v33);
            v34 = &v134[16 * v33];
            v35 = *(unsigned __int8 **)v34;
            if ( v132 == *(unsigned __int8 **)v34 )
              goto LABEL_28;
            ++v32;
          }
          if ( !v18 )
            v18 = v34;
          ++v133;
          v91 = v135 + 1;
          v137.m128i_i64[0] = (__int64)v18;
          if ( 4 * ((int)v135 + 1) < 3 * v136 )
          {
            if ( v136 - HIDWORD(v135) - v91 > v136 >> 3 )
            {
LABEL_151:
              LODWORD(v135) = v91;
              if ( *(_QWORD *)v18 != -4096 )
                --HIDWORD(v135);
              *(_QWORD *)v18 = v31;
              v36 = (unsigned __int8 **)(v18 + 8);
              *((_QWORD *)v18 + 1) = 0;
              goto LABEL_29;
            }
LABEL_156:
            sub_9BBF00((__int64)&v133, v30);
            sub_28EE430((__int64)&v133, (__int64 *)&v132, &v137);
            v31 = v132;
            v18 = (unsigned __int8 *)v137.m128i_i64[0];
            v91 = v135 + 1;
            goto LABEL_151;
          }
        }
        else
        {
          ++v133;
          v137.m128i_i64[0] = 0;
        }
        v30 = 2 * v136;
        goto LABEL_156;
      }
      v137.m128i_i64[0] = 0;
      if ( *v23 != 44 )
        goto LABEL_24;
      if ( !(unsigned __int8)sub_28EB290((__int64 **)&v137, (__int64)v23) )
        goto LABEL_99;
LABEL_72:
      v23 = v132;
      if ( *v132 <= 0x1Cu )
        goto LABEL_24;
      v125 = v132;
      v59 = sub_28EB420(v132);
      v61 = (__int64)v125;
      v62 = v59;
      v63 = (unsigned int)v149;
      v64 = (unsigned int)v149 + 1LL;
      if ( v64 > HIDWORD(v149) )
      {
        sub_C8D5F0((__int64)&v148, v150, v64, 0x10u, (__int64)v125, v60);
        v63 = (unsigned int)v149;
        v61 = (__int64)v125;
      }
      v65 = &v148[2 * v63];
      *v65 = v62;
      v65[1] = (__int64)v15;
      v66 = *(_QWORD *)(v62 + 16);
      LODWORD(v149) = v149 + 1;
      if ( v66 )
      {
        v118 = v61;
        v110 = v14;
        v67 = v66;
        do
        {
          if ( (unsigned __int8)(**(_BYTE **)(v67 + 24) - 42) <= 0x11u )
          {
            sub_D68D20((__int64)&v137, 0, *(_QWORD *)(v67 + 24));
            sub_28F19A0(a3, &v137);
            sub_D68D70(&v137);
          }
          v67 = *(_QWORD *)(v67 + 8);
        }
        while ( v67 );
        v61 = v118;
        v14 = v110;
      }
      sub_D68D20((__int64)&v137, 0, v61);
      sub_28F19A0(a3, &v137);
      sub_D68D70(&v137);
      v117 = 1;
LABEL_13:
      ++v14;
      v16 = *((_DWORD *)v10 + 1) & 0x7FFFFFF;
    }
    while ( v16 > (unsigned int)v14 );
    v7 = v149;
  }
  while ( (_DWORD)v149 );
LABEL_46:
  v47 = v145;
  if ( v145 != &v145[(unsigned int)v146] )
  {
    v48 = a4;
    v49 = &v145[(unsigned int)v146];
    do
    {
LABEL_48:
      v50 = *v47;
      if ( v136 )
      {
        v51 = (v136 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
        v52 = &v134[16 * v51];
        v53 = *(_QWORD *)v52;
        if ( v50 != *(_QWORD *)v52 )
        {
          v69 = 1;
          while ( v53 != -4096 )
          {
            v6 = (unsigned __int8 *)(unsigned int)(v69 + 1);
            v51 = (v136 - 1) & (v69 + v51);
            v52 = &v134[16 * v51];
            v53 = *(_QWORD *)v52;
            if ( v50 == *(_QWORD *)v52 )
              goto LABEL_50;
            v69 = (int)v6;
          }
          goto LABEL_61;
        }
LABEL_50:
        if ( v52 != &v134[16 * v136] )
        {
          v54 = *((_QWORD *)v52 + 1);
          *((_QWORD *)v52 + 1) = 0;
          v55 = *(unsigned int *)(a2 + 8);
          if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            v121 = v48;
            v123 = v54;
            sub_C8D5F0(a2, (const void *)(a2 + 16), v55 + 1, 0x10u, (__int64)v48, (__int64)v6);
            v55 = *(unsigned int *)(a2 + 8);
            v48 = v121;
            v54 = v123;
          }
          v56 = (__int64 *)(*(_QWORD *)a2 + 16 * v55);
          *v56 = v50;
          v56[1] = v54;
          ++*(_DWORD *)(a2 + 8);
          if ( v4 == 13 )
          {
            if ( v48[2] && v48[1] )
            {
              v122 = v48;
LABEL_85:
              ++v47;
              v137 = (__m128i)(unsigned __int64)v116;
              v138 = 0;
              v139 = 0;
              v140 = 0;
              v141 = 0;
              v142 = 0;
              v143 = 0;
              v144 = 257;
              v68 = sub_9AC470(v50, &v137, 0);
              v48 = v122;
              v122[2] &= v68;
              if ( v49 == v47 )
                break;
              goto LABEL_48;
            }
          }
          else if ( v4 == 17 && v48[3] && (*v48 || v48[1] && v48[2]) )
          {
            v144 = 257;
            v122 = v48;
            v137 = (__m128i)(unsigned __int64)v116;
            v138 = 0;
            v139 = 0;
            v140 = 0;
            v141 = 0;
            v142 = 0;
            v143 = 0;
            v57 = sub_9B6260(v50, &v137, 0);
            v48 = v122;
            v122[3] &= v57;
            if ( v122[1] )
            {
              if ( v122[2] )
                goto LABEL_85;
            }
          }
        }
      }
LABEL_61:
      ++v47;
    }
    while ( v49 != v47 );
  }
  if ( !*(_DWORD *)(a2 + 8) )
  {
    v79 = sub_AD93D0(v4, *((_QWORD *)a1 + 1), 0, 0);
    v82 = *(unsigned int *)(a2 + 12);
    v83 = *(unsigned int *)(a2 + 8);
    v84 = *(_DWORD *)(a2 + 8);
    if ( v83 >= v82 )
    {
      if ( v82 < v83 + 1 )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v83 + 1, 0x10u, v80, v81);
        v83 = *(unsigned int *)(a2 + 8);
      }
      v96 = (unsigned __int8 **)(*(_QWORD *)a2 + 16 * v83);
      *v96 = v79;
      v96[1] = (unsigned __int8 *)1;
      ++*(_DWORD *)(a2 + 8);
    }
    else
    {
      v85 = (unsigned __int8 **)(*(_QWORD *)a2 + 16 * v83);
      if ( v85 )
      {
        *v85 = v79;
        v85[1] = (unsigned __int8 *)1;
        v84 = *(_DWORD *)(a2 + 8);
      }
      *(_DWORD *)(a2 + 8) = v84 + 1;
    }
  }
  if ( v145 != (__int64 *)v147 )
    _libc_free((unsigned __int64)v145);
  sub_C7D6A0((__int64)v134, 16LL * v136, 8);
  if ( v148 != v150 )
    _libc_free((unsigned __int64)v148);
  return v117;
}
