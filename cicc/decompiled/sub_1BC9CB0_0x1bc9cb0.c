// Function: sub_1BC9CB0
// Address: 0x1bc9cb0
//
void __fastcall sub_1BC9CB0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  __int64 *v9; // rsi
  __int64 *v10; // r14
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rbx
  unsigned int v14; // esi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r10
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // rbx
  char v22; // al
  unsigned int v23; // r12d
  unsigned int v24; // r13d
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // esi
  __int64 v30; // r11
  __int64 v31; // r10
  __int64 v32; // rcx
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // r15
  unsigned int j; // edi
  __int64 v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // rax
  _QWORD *v46; // rax
  unsigned int v47; // esi
  __int64 v48; // rdi
  unsigned int v49; // ebx
  unsigned int v50; // ecx
  _QWORD *v51; // rdx
  _QWORD *v52; // r11
  __int64 v53; // rax
  int v54; // edx
  __int64 v55; // rbx
  __int64 v56; // rax
  __int64 v57; // rax
  char v58; // al
  int v59; // r11d
  int v60; // eax
  int v61; // edx
  int v62; // r13d
  int v63; // edi
  int v64; // ecx
  int v65; // edx
  __int64 v66; // rdi
  int v67; // edx
  int v68; // r10d
  unsigned __int64 v69; // rcx
  unsigned __int64 v70; // rcx
  unsigned int i; // eax
  __int64 *v72; // rsi
  unsigned int v73; // eax
  int v74; // eax
  int v75; // ecx
  __int64 v76; // rsi
  unsigned int v77; // eax
  __int64 v78; // rdi
  int v79; // r10d
  int v80; // eax
  int v81; // eax
  __int64 v82; // rsi
  unsigned int v83; // r12d
  __int64 v84; // rdi
  __int64 v85; // rcx
  int v86; // edx
  int v87; // esi
  __int64 v88; // rdi
  __int64 v89; // rdx
  int v90; // r11d
  __int64 v91; // r10
  int v92; // edx
  int v93; // edx
  __int64 v94; // rdi
  __int64 v95; // rbx
  int v96; // r10d
  __int64 v97; // rsi
  unsigned int v98; // edi
  int v99; // edx
  char v100; // al
  char v101; // al
  char v102; // al
  bool v103; // zf
  int v104; // eax
  int v105; // eax
  __int64 v106; // rdx
  __int64 v107; // rdi
  unsigned int v108; // r15d
  __int64 *v109; // rsi
  int v110; // r15d
  __int64 v111; // [rsp+10h] [rbp-140h]
  __int64 v112; // [rsp+18h] [rbp-138h]
  __int64 v113; // [rsp+18h] [rbp-138h]
  __int64 v114; // [rsp+18h] [rbp-138h]
  int v116; // [rsp+28h] [rbp-128h]
  __int64 v117; // [rsp+28h] [rbp-128h]
  __int64 v118; // [rsp+28h] [rbp-128h]
  __int64 v119; // [rsp+28h] [rbp-128h]
  __int64 v120; // [rsp+28h] [rbp-128h]
  __int64 v121; // [rsp+28h] [rbp-128h]
  __int64 *v122; // [rsp+30h] [rbp-120h]
  __int64 v123; // [rsp+38h] [rbp-118h]
  unsigned __int64 v124; // [rsp+40h] [rbp-110h]
  __int64 *v125; // [rsp+48h] [rbp-108h]
  char v127; // [rsp+58h] [rbp-F8h]
  _QWORD *v128; // [rsp+58h] [rbp-F8h]
  _QWORD *v129; // [rsp+58h] [rbp-F8h]
  __m128i v130; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v131; // [rsp+70h] [rbp-E0h]
  __int64 v132; // [rsp+78h] [rbp-D8h]
  __int64 v133; // [rsp+80h] [rbp-D0h]
  __m128i v134; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v135; // [rsp+A0h] [rbp-B0h]
  __int64 v136; // [rsp+A8h] [rbp-A8h]
  __int64 v137; // [rsp+B0h] [rbp-A0h]
  _QWORD *v138; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v139; // [rsp+C8h] [rbp-88h]
  _QWORD v140[16]; // [rsp+D0h] [rbp-80h] BYREF

  v7 = v140;
  v140[0] = a2;
  v138 = v140;
  v111 = a1 + 40;
  v139 = 0xA00000001LL;
  v8 = 1;
  do
  {
    v9 = (__int64 *)v7[v8 - 1];
    LODWORD(v139) = v8 - 1;
    v122 = v9;
    v10 = v9;
    if ( !v9 )
      goto LABEL_28;
    do
    {
      while ( *((_DWORD *)v10 + 22) != -1 )
      {
LABEL_4:
        v10 = (__int64 *)v10[2];
        if ( !v10 )
          goto LABEL_28;
      }
      v11 = v10[1];
      v12 = *((_DWORD *)v10 + 23);
      v10[11] = 0;
      *(_DWORD *)(v11 + 96) -= v12;
      v13 = *v10;
      if ( v10[13] == *v10 )
      {
        v44 = *(_QWORD *)(v13 + 8);
        if ( !v44 )
          goto LABEL_11;
        while ( 2 )
        {
          v46 = sub_1648700(v44);
          if ( *((_BYTE *)v46 + 16) <= 0x17u )
          {
            v45 = v10[1];
            ++*((_DWORD *)v10 + 22);
            ++*((_DWORD *)v10 + 23);
            ++*(_DWORD *)(v45 + 96);
            goto LABEL_60;
          }
          v47 = *(_DWORD *)(a1 + 64);
          if ( v47 )
          {
            LODWORD(a5) = v47 - 1;
            v48 = *(_QWORD *)(a1 + 48);
            v49 = ((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4);
            v50 = (v47 - 1) & v49;
            v51 = (_QWORD *)(v48 + 16LL * v50);
            v52 = (_QWORD *)*v51;
            if ( v46 == (_QWORD *)*v51 )
            {
LABEL_64:
              v53 = v51[1];
              if ( v53 )
              {
                v54 = *(_DWORD *)(a1 + 224);
                if ( *(_DWORD *)(v53 + 80) == v54 && *(_DWORD *)(*(_QWORD *)(v53 + 8) + 80LL) == v54 )
                {
                  ++*((_DWORD *)v10 + 22);
                  v55 = *(_QWORD *)(v53 + 8);
                  if ( !*(_BYTE *)(v55 + 100) )
                  {
                    v56 = v10[1];
                    ++*((_DWORD *)v10 + 23);
                    ++*(_DWORD *)(v56 + 96);
                  }
                  if ( *(_DWORD *)(v55 + 88) == -1 )
                  {
                    v57 = (unsigned int)v139;
                    if ( (unsigned int)v139 >= HIDWORD(v139) )
                    {
                      sub_16CD150((__int64)&v138, v140, 0, 8, a5, a6);
                      v57 = (unsigned int)v139;
                    }
                    v138[v57] = v55;
                    LODWORD(v139) = v139 + 1;
                  }
                }
              }
              goto LABEL_60;
            }
            v62 = 1;
            a6 = 0;
            while ( v52 != (_QWORD *)-8LL )
            {
              if ( !a6 && v52 == (_QWORD *)-16LL )
                a6 = (__int64)v51;
              v50 = a5 & (v62 + v50);
              v51 = (_QWORD *)(v48 + 16LL * v50);
              v52 = (_QWORD *)*v51;
              if ( v46 == (_QWORD *)*v51 )
                goto LABEL_64;
              ++v62;
            }
            v63 = *(_DWORD *)(a1 + 56);
            if ( !a6 )
              a6 = (__int64)v51;
            ++*(_QWORD *)(a1 + 40);
            v64 = v63 + 1;
            if ( 4 * (v63 + 1) < 3 * v47 )
            {
              if ( v47 - *(_DWORD *)(a1 + 60) - v64 <= v47 >> 3 )
              {
                v129 = v46;
                sub_1BC8C30(v111, v47);
                v92 = *(_DWORD *)(a1 + 64);
                if ( !v92 )
                {
LABEL_215:
                  ++*(_DWORD *)(a1 + 56);
                  BUG();
                }
                v93 = v92 - 1;
                v94 = *(_QWORD *)(a1 + 48);
                a5 = 0;
                LODWORD(v95) = v93 & v49;
                v96 = 1;
                v64 = *(_DWORD *)(a1 + 56) + 1;
                v46 = v129;
                a6 = v94 + 16LL * (unsigned int)v95;
                v97 = *(_QWORD *)a6;
                if ( v129 != *(_QWORD **)a6 )
                {
                  while ( v97 != -8 )
                  {
                    if ( v97 == -16 && !a5 )
                      a5 = a6;
                    v95 = v93 & (unsigned int)(v95 + v96);
                    a6 = v94 + 16 * v95;
                    v97 = *(_QWORD *)a6;
                    if ( v129 == *(_QWORD **)a6 )
                      goto LABEL_96;
                    ++v96;
                  }
                  if ( a5 )
                    a6 = a5;
                }
              }
              goto LABEL_96;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 40);
          }
          v128 = v46;
          sub_1BC8C30(v111, 2 * v47);
          v86 = *(_DWORD *)(a1 + 64);
          if ( !v86 )
            goto LABEL_215;
          v46 = v128;
          v87 = v86 - 1;
          v88 = *(_QWORD *)(a1 + 48);
          LODWORD(v89) = (v86 - 1) & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
          v64 = *(_DWORD *)(a1 + 56) + 1;
          a6 = v88 + 16LL * (unsigned int)v89;
          a5 = *(_QWORD *)a6;
          if ( v128 != *(_QWORD **)a6 )
          {
            v90 = 1;
            v91 = 0;
            while ( a5 != -8 )
            {
              if ( a5 == -16 && !v91 )
                v91 = a6;
              v89 = v87 & (unsigned int)(v89 + v90);
              a6 = v88 + 16 * v89;
              a5 = *(_QWORD *)a6;
              if ( v128 == *(_QWORD **)a6 )
                goto LABEL_96;
              ++v90;
            }
            if ( v91 )
              a6 = v91;
          }
LABEL_96:
          *(_DWORD *)(a1 + 56) = v64;
          if ( *(_QWORD *)a6 != -8 )
            --*(_DWORD *)(a1 + 60);
          *(_QWORD *)a6 = v46;
          *(_QWORD *)(a6 + 8) = 0;
LABEL_60:
          v44 = *(_QWORD *)(v44 + 8);
          if ( !v44 )
            goto LABEL_11;
          continue;
        }
      }
      v14 = *(_DWORD *)(a1 + 64);
      if ( v14 )
      {
        v15 = *(_QWORD *)(a1 + 48);
        v16 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v13 == *v17 )
        {
LABEL_9:
          v19 = v17[1];
          if ( v19 )
          {
            v20 = *(_DWORD *)(a1 + 224);
            if ( *(_DWORD *)(v19 + 80) == v20 && *(_DWORD *)(*(_QWORD *)(v19 + 8) + 80LL) == v20 )
            {
              ++*((_DWORD *)v10 + 22);
              v41 = *(_QWORD *)(v19 + 8);
              if ( !*(_BYTE *)(v41 + 100) )
              {
                v42 = v10[1];
                ++*((_DWORD *)v10 + 23);
                ++*(_DWORD *)(v42 + 96);
              }
              if ( *(_DWORD *)(v41 + 88) == -1 )
              {
                v43 = (unsigned int)v139;
                if ( (unsigned int)v139 >= HIDWORD(v139) )
                {
                  sub_16CD150((__int64)&v138, v140, 0, 8, a5, a6);
                  v43 = (unsigned int)v139;
                }
                v138[v43] = v41;
                LODWORD(v139) = v139 + 1;
              }
            }
          }
          goto LABEL_11;
        }
        v59 = 1;
        a5 = 0;
        while ( v18 != -8 )
        {
          if ( !a5 && v18 == -16 )
            a5 = (__int64)v17;
          LODWORD(a6) = v59 + 1;
          v16 = (v14 - 1) & (v59 + v16);
          v17 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( v13 == *v17 )
            goto LABEL_9;
          ++v59;
        }
        if ( !a5 )
          a5 = (__int64)v17;
        v60 = *(_DWORD *)(a1 + 56);
        ++*(_QWORD *)(a1 + 40);
        v61 = v60 + 1;
        if ( 4 * (v60 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 60) - v61 <= v14 >> 3 )
          {
            sub_1BC8C30(v111, v14);
            v80 = *(_DWORD *)(a1 + 64);
            if ( !v80 )
            {
LABEL_214:
              ++*(_DWORD *)(a1 + 56);
              BUG();
            }
            v81 = v80 - 1;
            v82 = *(_QWORD *)(a1 + 48);
            LODWORD(a6) = 1;
            v83 = v81 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v61 = *(_DWORD *)(a1 + 56) + 1;
            v84 = 0;
            a5 = v82 + 16LL * v83;
            v85 = *(_QWORD *)a5;
            if ( v13 != *(_QWORD *)a5 )
            {
              while ( v85 != -8 )
              {
                if ( !v84 && v85 == -16 )
                  v84 = a5;
                v83 = v81 & (a6 + v83);
                a5 = v82 + 16LL * v83;
                v85 = *(_QWORD *)a5;
                if ( v13 == *(_QWORD *)a5 )
                  goto LABEL_87;
                LODWORD(a6) = a6 + 1;
              }
              if ( v84 )
                a5 = v84;
            }
          }
          goto LABEL_87;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 40);
      }
      sub_1BC8C30(v111, 2 * v14);
      v74 = *(_DWORD *)(a1 + 64);
      if ( !v74 )
        goto LABEL_214;
      v75 = v74 - 1;
      v76 = *(_QWORD *)(a1 + 48);
      v77 = (v74 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v61 = *(_DWORD *)(a1 + 56) + 1;
      a5 = v76 + 16LL * v77;
      v78 = *(_QWORD *)a5;
      if ( v13 != *(_QWORD *)a5 )
      {
        v79 = 1;
        a6 = 0;
        while ( v78 != -8 )
        {
          if ( v78 == -16 && !a6 )
            a6 = a5;
          v77 = v75 & (v79 + v77);
          a5 = v76 + 16LL * v77;
          v78 = *(_QWORD *)a5;
          if ( v13 == *(_QWORD *)a5 )
            goto LABEL_87;
          ++v79;
        }
        if ( a6 )
          a5 = a6;
      }
LABEL_87:
      *(_DWORD *)(a1 + 56) = v61;
      if ( *(_QWORD *)a5 != -8 )
        --*(_DWORD *)(a1 + 60);
      *(_QWORD *)a5 = v13;
      *(_QWORD *)(a5 + 8) = 0;
LABEL_11:
      v21 = v10[3];
      if ( !v21 )
        goto LABEL_4;
      v22 = *(_BYTE *)(*v10 + 16);
      v125 = (__int64 *)*v10;
      if ( v22 == 55 )
      {
        sub_141EDF0(&v130, *v10);
      }
      else if ( v22 == 54 )
      {
        sub_141EB40(&v130, v125);
      }
      else
      {
        v130.m128i_i64[0] = 0;
        v130.m128i_i64[1] = -1;
        v131 = 0;
        v132 = 0;
        v133 = 0;
      }
      v23 = 1;
      v123 = a1;
      v127 = sub_15F3040(*v10);
      v24 = 0;
      v124 = (unsigned __int64)(((unsigned int)v125 >> 9) ^ ((unsigned int)v125 >> 4)) << 32;
      while ( 1 )
      {
LABEL_23:
        if ( v23 > 0x9F )
          goto LABEL_16;
        if ( !v127 && !(unsigned __int8)sub_15F3040(*(_QWORD *)v21) )
          break;
        if ( v24 > 9 )
          goto LABEL_16;
        a5 = *(_QWORD *)v21;
        v29 = *(_DWORD *)(a4 + 296);
        v30 = a4 + 272;
        if ( !v29 )
        {
          ++*(_QWORD *)(a4 + 272);
LABEL_100:
          v117 = a5;
          sub_1BC8970(v30, 2 * v29);
          v65 = *(_DWORD *)(a4 + 296);
          if ( !v65 )
            goto LABEL_216;
          a5 = v117;
          v66 = *(_QWORD *)(a4 + 280);
          v67 = v65 - 1;
          a6 = 0;
          v68 = 1;
          v69 = (((((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4) | v124)
                - 1
                - ((unsigned __int64)(((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4)) << 32)) >> 22)
              ^ ((((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4) | v124)
               - 1
               - ((unsigned __int64)(((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4)) << 32));
          v70 = ((9 * (((v69 - 1 - (v69 << 13)) >> 8) ^ (v69 - 1 - (v69 << 13)))) >> 15)
              ^ (9 * (((v69 - 1 - (v69 << 13)) >> 8) ^ (v69 - 1 - (v69 << 13))));
          for ( i = v67 & (((v70 - 1 - (v70 << 27)) >> 31) ^ (v70 - 1 - ((_DWORD)v70 << 27))); ; i = v67 & v73 )
          {
            v32 = v66 + 24LL * i;
            v72 = *(__int64 **)v32;
            if ( v125 == *(__int64 **)v32 && v117 == *(_QWORD *)(v32 + 8) )
              break;
            if ( v72 == (__int64 *)-8LL )
            {
              if ( *(_QWORD *)(v32 + 8) == -8 )
              {
                if ( a6 )
                  v32 = a6;
                v99 = *(_DWORD *)(a4 + 288) + 1;
                goto LABEL_141;
              }
            }
            else if ( v72 == (__int64 *)-16LL && *(_QWORD *)(v32 + 8) == -16 && !a6 )
            {
              a6 = v66 + 24LL * i;
            }
            v73 = v68 + i;
            ++v68;
          }
LABEL_140:
          v99 = *(_DWORD *)(a4 + 288) + 1;
          goto LABEL_141;
        }
        v31 = *(_QWORD *)(a4 + 280);
        LODWORD(a6) = v29 - 1;
        v116 = 1;
        v32 = 0;
        v33 = (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4) | v124)
            - 1
            - ((unsigned __int64)(((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4)) << 32);
        v34 = (v33 ^ (v33 >> 22)) - 1 - ((v33 ^ (v33 >> 22)) << 13);
        v35 = (9 * ((v34 >> 8) ^ v34)) ^ ((9 * ((v34 >> 8) ^ v34)) >> 15);
        v36 = ((v35 - 1 - (v35 << 27)) >> 31) ^ (v35 - 1 - (v35 << 27));
        for ( j = v36 & (v29 - 1); ; j = a6 & v98 )
        {
          v38 = v31 + 24LL * j;
          v39 = *(__int64 **)v38;
          if ( v125 == *(__int64 **)v38 && a5 == *(_QWORD *)(v38 + 8) )
          {
            if ( *(_BYTE *)(v38 + 17) )
            {
              v58 = *(_BYTE *)(v38 + 16);
              goto LABEL_76;
            }
            v32 = v31 + 24LL * j;
            goto LABEL_144;
          }
          if ( v39 == (__int64 *)-8LL )
            break;
          if ( v39 == (__int64 *)-16LL && *(_QWORD *)(v38 + 8) == -16 && !v32 )
            v32 = v31 + 24LL * j;
LABEL_138:
          v98 = v116 + j;
          ++v116;
        }
        if ( *(_QWORD *)(v38 + 8) != -8 )
          goto LABEL_138;
        if ( !v32 )
          v32 = v31 + 24LL * j;
        ++*(_QWORD *)(a4 + 272);
        v99 = *(_DWORD *)(a4 + 288) + 1;
        if ( 4 * v99 >= 3 * v29 )
          goto LABEL_100;
        if ( v29 - *(_DWORD *)(a4 + 292) - v99 <= v29 >> 3 )
        {
          v121 = a5;
          sub_1BC8970(v30, v29);
          v104 = *(_DWORD *)(a4 + 296);
          if ( v104 )
          {
            v105 = v104 - 1;
            a5 = v121;
            v106 = *(_QWORD *)(a4 + 280);
            v107 = 0;
            v108 = v105 & v36;
            LODWORD(a6) = 1;
            while ( 1 )
            {
              v32 = v106 + 24LL * v108;
              v109 = *(__int64 **)v32;
              if ( v125 == *(__int64 **)v32 && v121 == *(_QWORD *)(v32 + 8) )
                goto LABEL_140;
              if ( v109 == (__int64 *)-8LL )
              {
                if ( *(_QWORD *)(v32 + 8) == -8 )
                {
                  if ( v107 )
                    v32 = v107;
                  v99 = *(_DWORD *)(a4 + 288) + 1;
                  goto LABEL_141;
                }
              }
              else if ( v109 == (__int64 *)-16LL && *(_QWORD *)(v32 + 8) == -16 && !v107 )
              {
                v107 = v106 + 24LL * v108;
              }
              v110 = a6 + v108;
              LODWORD(a6) = a6 + 1;
              v108 = v105 & v110;
            }
          }
LABEL_216:
          ++*(_DWORD *)(a4 + 288);
          BUG();
        }
LABEL_141:
        *(_DWORD *)(a4 + 288) = v99;
        if ( *(_QWORD *)v32 != -8 || *(_QWORD *)(v32 + 8) != -8 )
          --*(_DWORD *)(a4 + 292);
        *(_QWORD *)(v32 + 8) = a5;
        *(_BYTE *)(v32 + 17) = 0;
        *(_QWORD *)v32 = v125;
LABEL_144:
        v100 = *(_BYTE *)(a5 + 16);
        if ( v100 != 55 )
        {
          if ( v100 == 54 )
          {
            v114 = v32;
            v120 = a5;
            sub_141EB40(&v134, (__int64 *)a5);
            a5 = v120;
            v32 = v114;
            goto LABEL_146;
          }
          v134.m128i_i64[0] = 0;
          v134.m128i_i64[1] = -1;
          v135 = 0;
          v136 = 0;
          v137 = 0;
LABEL_149:
          if ( *(_BYTE *)(v32 + 17) )
            *(_BYTE *)(v32 + 16) = 1;
          else
            *(_WORD *)(v32 + 16) = 257;
          goto LABEL_16;
        }
        v112 = v32;
        v118 = a5;
        sub_141EDF0(&v134, a5);
        a5 = v118;
        v32 = v112;
LABEL_146:
        if ( !v130.m128i_i64[0] )
          goto LABEL_149;
        if ( !v134.m128i_i64[0] )
          goto LABEL_149;
        v113 = v32;
        v119 = a5;
        v101 = sub_1BBA4E0((__int64)v125);
        LODWORD(a5) = v119;
        v32 = v113;
        if ( !v101 )
          goto LABEL_149;
        v102 = sub_1BBA4E0(v119);
        v32 = v113;
        if ( !v102 )
          goto LABEL_149;
        v58 = (unsigned __int8)sub_134CB50(*(_QWORD *)(a4 + 1336), (__int64)&v130, (__int64)&v134) != 0;
        v103 = *(_BYTE *)(v113 + 17) == 0;
        *(_BYTE *)(v113 + 16) = v58;
        if ( v103 )
          *(_BYTE *)(v113 + 17) = 1;
LABEL_76:
        if ( !v58 )
          break;
LABEL_16:
        v25 = *(unsigned int *)(v21 + 40);
        ++v24;
        if ( (unsigned int)v25 >= *(_DWORD *)(v21 + 44) )
        {
          sub_16CD150(v21 + 32, (const void *)(v21 + 48), 0, 8, a5, a6);
          v25 = *(unsigned int *)(v21 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(v21 + 32) + 8 * v25) = v10;
        ++*(_DWORD *)(v21 + 40);
        ++*((_DWORD *)v10 + 22);
        v26 = *(_QWORD *)(v21 + 8);
        if ( !*(_BYTE *)(v26 + 100) )
        {
          v27 = v10[1];
          ++*((_DWORD *)v10 + 23);
          ++*(_DWORD *)(v27 + 96);
        }
        if ( *(_DWORD *)(v26 + 88) == -1 )
        {
          v40 = (unsigned int)v139;
          if ( (unsigned int)v139 >= HIDWORD(v139) )
          {
            sub_16CD150((__int64)&v138, v140, 0, 8, a5, a6);
            v40 = (unsigned int)v139;
          }
          v138[v40] = v26;
          LODWORD(v139) = v139 + 1;
        }
        v21 = *(_QWORD *)(v21 + 24);
        if ( v23 != 320 )
        {
          ++v23;
          if ( v21 )
            continue;
        }
        goto LABEL_27;
      }
      v21 = *(_QWORD *)(v21 + 24);
      ++v23;
      if ( v21 )
        goto LABEL_23;
LABEL_27:
      v10 = (__int64 *)v10[2];
      a1 = v123;
    }
    while ( v10 );
LABEL_28:
    if ( a3 && !*((_DWORD *)v122 + 24) && !*((_BYTE *)v122 + 100) )
    {
      v28 = *(unsigned int *)(a1 + 112);
      if ( (unsigned int)v28 >= *(_DWORD *)(a1 + 116) )
      {
        sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 8, a5, a6);
        v28 = *(unsigned int *)(a1 + 112);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v28) = v122;
      ++*(_DWORD *)(a1 + 112);
    }
    v8 = v139;
    v7 = v138;
  }
  while ( (_DWORD)v139 );
  if ( v138 != v140 )
    _libc_free((unsigned __int64)v138);
}
