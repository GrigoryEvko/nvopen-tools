// Function: sub_19AC5F0
// Address: 0x19ac5f0
//
__int64 __fastcall sub_19AC5F0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi
  __int64 v3; // rsi
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  char v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r13
  int v11; // r9d
  int v12; // r11d
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rbx
  __int64 *v16; // rdx
  unsigned int i; // r8d
  __int64 *v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rdi
  int v21; // edi
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // rsi
  __int64 *v26; // r13
  __int64 *v27; // r15
  __int64 v28; // rcx
  unsigned __int64 v29; // r13
  __int64 v30; // rbx
  __int64 *v31; // rax
  unsigned int v32; // edx
  __int64 v33; // rsi
  unsigned __int64 v34; // rdi
  unsigned int v35; // edx
  _QWORD *v36; // rax
  __int64 v37; // r14
  __int64 v38; // r12
  unsigned __int64 v39; // r14
  __int64 v40; // rbx
  __int64 *v41; // rax
  unsigned int v42; // edx
  __int64 v43; // rsi
  unsigned __int64 v44; // rdi
  unsigned int v45; // edx
  _QWORD *v46; // rax
  __int64 v47; // r15
  __int64 v48; // r12
  bool v49; // al
  int v50; // eax
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  int v55; // r8d
  int v56; // r9d
  __int64 *v57; // rsi
  __int64 *v58; // rsi
  unsigned int v59; // ecx
  _QWORD *v60; // rdi
  unsigned int v61; // eax
  int v62; // eax
  unsigned __int64 v63; // r12
  unsigned int v64; // eax
  _QWORD *v65; // rax
  _QWORD *k; // rdx
  int v67; // edi
  __int64 *v68; // rcx
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rdx
  unsigned int v71; // eax
  __int64 v72; // r8
  unsigned int v73; // eax
  int v74; // edi
  int v75; // edi
  unsigned int v76; // r8d
  int v77; // esi
  unsigned int j; // ebx
  __int64 *v79; // rax
  __int64 v80; // r8
  unsigned int v81; // ebx
  _QWORD *v82; // rax
  _QWORD *v83; // [rsp-20h] [rbp-2C0h]
  _QWORD *v84; // [rsp-20h] [rbp-2C0h]
  __int64 v85; // [rsp-18h] [rbp-2B8h]
  __int64 v86; // [rsp-18h] [rbp-2B8h]
  __int64 v87; // [rsp+8h] [rbp-298h]
  __int64 v88; // [rsp+10h] [rbp-290h]
  __int64 v89; // [rsp+18h] [rbp-288h]
  __int64 *v90; // [rsp+20h] [rbp-280h]
  __int64 v91; // [rsp+28h] [rbp-278h]
  __int64 v93; // [rsp+38h] [rbp-268h]
  __int64 v94; // [rsp+40h] [rbp-260h]
  unsigned int v95; // [rsp+48h] [rbp-258h]
  unsigned __int64 v96; // [rsp+48h] [rbp-258h]
  __int64 v97; // [rsp+50h] [rbp-250h]
  __int64 v98; // [rsp+58h] [rbp-248h]
  __int64 *v99; // [rsp+60h] [rbp-240h]
  unsigned int v100; // [rsp+60h] [rbp-240h]
  __int64 *v101; // [rsp+70h] [rbp-230h]
  __int64 v102; // [rsp+78h] [rbp-228h]
  __int64 v103; // [rsp+80h] [rbp-220h]
  unsigned int v104; // [rsp+8Ch] [rbp-214h]
  __int64 v105; // [rsp+90h] [rbp-210h] BYREF
  _QWORD *v106; // [rsp+98h] [rbp-208h]
  __int64 v107; // [rsp+A0h] [rbp-200h]
  unsigned int v108; // [rsp+A8h] [rbp-1F8h]
  __int64 v109; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v110; // [rsp+B8h] [rbp-1E8h]
  __int64 v111; // [rsp+C0h] [rbp-1E0h]
  __int64 v112; // [rsp+C8h] [rbp-1D8h]
  _QWORD v113[4]; // [rsp+D0h] [rbp-1D0h] BYREF
  _QWORD v114[4]; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 v115; // [rsp+110h] [rbp-190h] BYREF
  _BYTE *v116; // [rsp+118h] [rbp-188h]
  _BYTE *v117; // [rsp+120h] [rbp-180h]
  __int64 v118; // [rsp+128h] [rbp-178h]
  int v119; // [rsp+130h] [rbp-170h]
  _BYTE v120[136]; // [rsp+138h] [rbp-168h] BYREF
  __int64 v121; // [rsp+1C0h] [rbp-E0h] BYREF
  __int64 *v122; // [rsp+1C8h] [rbp-D8h]
  __int64 *v123; // [rsp+1D0h] [rbp-D0h]
  __int64 v124; // [rsp+1D8h] [rbp-C8h]
  char *v125; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v126; // [rsp+1E8h] [rbp-B8h] BYREF
  _BYTE v127[32]; // [rsp+1F0h] [rbp-B0h] BYREF
  __int64 v128; // [rsp+210h] [rbp-90h]
  __int64 v129; // [rsp+218h] [rbp-88h]

  result = *(unsigned int *)(a1 + 376);
  v2 = *(_QWORD *)(a1 + 368);
  v87 = result;
  v3 = v2 + 1984 * result;
  if ( v2 == v3 )
    return result;
  result = v2;
  v4 = 1;
  while ( 1 )
  {
    v5 = *(unsigned int *)(result + 752);
    if ( v5 > 0xFFFE )
      break;
    v4 *= v5;
    if ( v4 > 0x3FFFB )
      break;
    result += 1984;
    if ( v3 == result )
    {
      if ( v4 <= 0xFFFE )
        return result;
      break;
    }
  }
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v115 = 0;
  v116 = v120;
  v117 = v120;
  v118 = 16;
  v119 = 0;
  if ( !v87 )
  {
    v20 = 0;
    goto LABEL_22;
  }
  v88 = 0;
  while ( 2 )
  {
    v6 = 0;
    v98 = 0;
    v94 = v2 + 1984 * v88;
    v93 = *(unsigned int *)(v94 + 752);
    if ( !*(_DWORD *)(v94 + 752) )
      goto LABEL_35;
    do
    {
      v7 = *(_QWORD *)(v94 + 744);
      v8 = v7 + 96 * v98;
      v9 = *(_QWORD *)(v8 + 80);
      v97 = v8;
      if ( !v9 )
        goto LABEL_31;
      v10 = *(_QWORD *)(v8 + 24);
      if ( !v108 )
      {
        ++v105;
        goto LABEL_107;
      }
      v11 = v108 - 1;
      v12 = 1;
      v13 = ((((unsigned int)(37 * v10) | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v10) << 32)) >> 22)
          ^ (((unsigned int)(37 * v10) | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v10) << 32));
      v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
          ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
      v15 = ((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - (v14 << 27));
      v16 = 0;
      for ( i = v15 & (v108 - 1); ; i = v11 & v76 )
      {
        v18 = &v106[3 * i];
        v19 = *v18;
        if ( v9 == *v18 && v10 == v18[1] )
        {
          v25 = v7 + 96 * v18[2];
          v26 = *(__int64 **)(v25 + 32);
          v89 = v25;
          v27 = *(__int64 **)(v97 + 32);
          v28 = (__int64)&v27[*(unsigned int *)(v97 + 40)];
          v99 = (__int64 *)v28;
          v101 = &v26[*(unsigned int *)(v25 + 40)];
          if ( v27 == (__int64 *)v28 )
          {
            if ( v26 == v101 )
              goto LABEL_85;
            v96 = 0;
            v102 = *(_QWORD *)(a1 + 32136);
            v104 = *(_DWORD *)(a1 + 32152);
            v91 = v104;
            v103 = v87 + 1;
          }
          else
          {
            v90 = *(__int64 **)(v25 + 32);
            v102 = *(_QWORD *)(a1 + 32136);
            v104 = *(_DWORD *)(a1 + 32152);
            v103 = v87 + 1;
            v91 = v104;
            v29 = 0;
            v95 = v104 - 1;
            do
            {
              v31 = (__int64 *)(v102 + 16LL * v104);
              if ( v104 )
              {
                v28 = *v27;
                v32 = v95 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
                v31 = (__int64 *)(v102 + 16LL * v32);
                v33 = *v31;
                if ( *v27 != *v31 )
                {
                  v51 = 1;
                  while ( v33 != -8 )
                  {
                    v32 = v95 & (v51 + v32);
                    v75 = v51 + 1;
                    v31 = (__int64 *)(v102 + 16LL * v32);
                    v33 = *v31;
                    if ( v28 == *v31 )
                      goto LABEL_50;
                    v51 = v75;
                  }
                  v31 = (__int64 *)(v102 + 16LL * v104);
                }
              }
LABEL_50:
              v34 = v31[1];
              if ( (v34 & 1) != 0 )
              {
                v30 = (int)sub_39FAC40(~(-1LL << (v34 >> 58)) & (v34 >> 1));
              }
              else
              {
                v35 = (unsigned int)(*(_DWORD *)(v34 + 16) + 63) >> 6;
                if ( v35 )
                {
                  v36 = *(_QWORD **)v34;
                  LODWORD(v30) = 0;
                  v37 = *(_QWORD *)v34 + 8LL;
                  v38 = v37 + 8LL * (v35 - 1);
                  while ( 1 )
                  {
                    v30 = (unsigned int)sub_39FAC40(*v36) + (unsigned int)v30;
                    v36 = (_QWORD *)v37;
                    if ( v37 == v38 )
                      break;
                    v37 += 8;
                  }
                }
                else
                {
                  v30 = 0;
                }
              }
              ++v27;
              v29 += v103 - v30;
            }
            while ( v99 != v27 );
            v96 = v29;
            v26 = v90;
            if ( v90 == v101 )
            {
              v39 = 0;
              goto LABEL_69;
            }
          }
          v39 = 0;
          v100 = v104 - 1;
          do
          {
            v28 = v104;
            v41 = (__int64 *)(v102 + 16 * v91);
            if ( v104 )
            {
              v28 = *v26;
              v42 = v100 & (((unsigned int)*v26 >> 9) ^ ((unsigned int)*v26 >> 4));
              v41 = (__int64 *)(v102 + 16LL * v42);
              v43 = *v41;
              if ( *v26 != *v41 )
              {
                v50 = 1;
                while ( v43 != -8 )
                {
                  v42 = v100 & (v50 + v42);
                  v74 = v50 + 1;
                  v41 = (__int64 *)(v102 + 16LL * v42);
                  v43 = *v41;
                  if ( v28 == *v41 )
                    goto LABEL_62;
                  v50 = v74;
                }
                v41 = (__int64 *)(v102 + 16 * v91);
              }
            }
LABEL_62:
            v44 = v41[1];
            if ( (v44 & 1) != 0 )
            {
              v40 = (int)sub_39FAC40(~(-1LL << (v44 >> 58)) & (v44 >> 1));
            }
            else
            {
              v45 = (unsigned int)(*(_DWORD *)(v44 + 16) + 63) >> 6;
              if ( v45 )
              {
                v46 = *(_QWORD **)v44;
                LODWORD(v40) = 0;
                v47 = *(_QWORD *)v44 + 8LL;
                v48 = v47 + 8LL * (v45 - 1);
                while ( 1 )
                {
                  v40 = (unsigned int)sub_39FAC40(*v46) + (unsigned int)v40;
                  v46 = (_QWORD *)v47;
                  if ( v48 == v47 )
                    break;
                  v47 += 8;
                }
              }
              else
              {
                v40 = 0;
              }
            }
            ++v26;
            v39 += v103 - v40;
          }
          while ( v26 != v101 );
LABEL_69:
          if ( v96 != v39 )
          {
            v49 = v96 < v39;
LABEL_71:
            if ( v49 )
            {
              v121 = *(_QWORD *)v97;
              v122 = *(__int64 **)(v97 + 8);
              LOBYTE(v123) = *(_BYTE *)(v97 + 16);
              v52 = *(_QWORD *)(v97 + 24);
              v125 = v127;
              v124 = v52;
              v126 = 0x400000000LL;
              v53 = *(unsigned int *)(v97 + 40);
              if ( (_DWORD)v53 )
                sub_19931B0((__int64)&v125, (char **)(v97 + 32), v53, v28, i, v11);
              v128 = *(_QWORD *)(v97 + 80);
              v129 = *(_QWORD *)(v97 + 88);
              *(_QWORD *)v97 = *(_QWORD *)v89;
              *(_QWORD *)(v97 + 8) = *(_QWORD *)(v89 + 8);
              *(_BYTE *)(v97 + 16) = *(_BYTE *)(v89 + 16);
              *(_QWORD *)(v97 + 24) = *(_QWORD *)(v89 + 24);
              sub_19931B0(v97 + 32, (char **)(v89 + 32), v53, v97, i, v11);
              *(_QWORD *)(v97 + 80) = *(_QWORD *)(v89 + 80);
              *(_QWORD *)(v97 + 88) = *(_QWORD *)(v89 + 88);
              *(_QWORD *)v89 = v121;
              *(_QWORD *)(v89 + 8) = v122;
              *(_BYTE *)(v89 + 16) = (_BYTE)v123;
              *(_QWORD *)(v89 + 24) = v124;
              sub_19931B0(v89 + 32, &v125, v54, v97, v55, v56);
              *(_QWORD *)(v89 + 80) = v128;
              *(_QWORD *)(v89 + 88) = v129;
              if ( v125 != v127 )
                _libc_free((unsigned __int64)v125);
            }
            v6 = 1;
            sub_1994A60(v94, (__int64 *)v97);
            --v93;
            goto LABEL_32;
          }
LABEL_85:
          memset(v113, 0, sizeof(v113));
          memset(v114, 0, sizeof(v114));
          sub_18CE100((__int64)&v115);
          v121 = 0;
          v122 = &v126;
          v57 = *(__int64 **)(a1 + 32);
          v85 = *(_QWORD *)(a1 + 8);
          v83 = *(_QWORD **)(a1 + 40);
          v123 = &v126;
          v124 = 16;
          LODWORD(v125) = 0;
          sub_199D0A0((__int64)v113, v57, v97, (__int64)&v115, (__int64)&v121, (__int64)&v109, v83, v85, v94, 0);
          if ( v123 != v122 )
            _libc_free((unsigned __int64)v123);
          sub_18CE100((__int64)&v115);
          v122 = &v126;
          v123 = &v126;
          v58 = *(__int64 **)(a1 + 32);
          v86 = *(_QWORD *)(a1 + 8);
          v84 = *(_QWORD **)(a1 + 40);
          v121 = 0;
          v124 = 16;
          LODWORD(v125) = 0;
          sub_199D0A0((__int64)v114, v58, v89, (__int64)&v115, (__int64)&v121, (__int64)&v109, v84, v86, v94, 0);
          if ( v123 != v122 )
            _libc_free((unsigned __int64)v123);
          v49 = sub_1992A80(v113, v114, *(_BYTE *)(a1 + 49));
          goto LABEL_71;
        }
        if ( v19 == -8 )
          break;
        if ( v19 == -16 && v18[1] == 0x7FFFFFFFFFFFFFFELL && !v16 )
          v16 = &v106[3 * i];
LABEL_123:
        v76 = v12 + i;
        ++v12;
      }
      if ( v18[1] != 0x7FFFFFFFFFFFFFFFLL )
        goto LABEL_123;
      if ( !v16 )
        v16 = &v106[3 * i];
      ++v105;
      v21 = v107 + 1;
      if ( 4 * ((int)v107 + 1) < 3 * v108 )
      {
        if ( v108 - HIDWORD(v107) - v21 > v108 >> 3 )
          goto LABEL_28;
        sub_19AC310((__int64)&v105, v108);
        if ( v108 )
        {
          v77 = 1;
          v16 = 0;
          for ( j = (v108 - 1) & v15; ; j = (v108 - 1) & v81 )
          {
            v79 = &v106[3 * j];
            v80 = *v79;
            if ( v9 == *v79 && v10 == v79[1] )
            {
              v16 = &v106[3 * j];
              v21 = v107 + 1;
              goto LABEL_28;
            }
            if ( v80 == -8 )
            {
              if ( v79[1] == 0x7FFFFFFFFFFFFFFFLL )
              {
                if ( !v16 )
                  v16 = &v106[3 * j];
                v21 = v107 + 1;
                goto LABEL_28;
              }
            }
            else if ( v80 == -16 && v79[1] == 0x7FFFFFFFFFFFFFFELL && !v16 )
            {
              v16 = &v106[3 * j];
            }
            v81 = v77 + j;
            ++v77;
          }
        }
LABEL_156:
        LODWORD(v107) = v107 + 1;
        BUG();
      }
LABEL_107:
      sub_19AC310((__int64)&v105, 2 * v108);
      if ( !v108 )
        goto LABEL_156;
      v67 = 1;
      v68 = 0;
      v69 = ((((unsigned int)(37 * v10) | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v10) << 32)) >> 22)
          ^ (((unsigned int)(37 * v10) | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v10) << 32));
      v70 = ((9 * (((v69 - 1 - (v69 << 13)) >> 8) ^ (v69 - 1 - (v69 << 13)))) >> 15)
          ^ (9 * (((v69 - 1 - (v69 << 13)) >> 8) ^ (v69 - 1 - (v69 << 13))));
      v71 = (v108 - 1) & (((v70 - 1 - (v70 << 27)) >> 31) ^ (v70 - 1 - ((_DWORD)v70 << 27)));
      while ( 2 )
      {
        v16 = &v106[3 * v71];
        v72 = *v16;
        if ( v9 == *v16 && v10 == v16[1] )
        {
          v21 = v107 + 1;
          goto LABEL_28;
        }
        if ( v72 != -8 )
        {
          if ( v72 == -16 && v16[1] == 0x7FFFFFFFFFFFFFFELL && !v68 )
            v68 = &v106[3 * v71];
          goto LABEL_115;
        }
        if ( v16[1] != 0x7FFFFFFFFFFFFFFFLL )
        {
LABEL_115:
          v73 = v67 + v71;
          ++v67;
          v71 = (v108 - 1) & v73;
          continue;
        }
        break;
      }
      if ( v68 )
        v16 = v68;
      v21 = v107 + 1;
LABEL_28:
      LODWORD(v107) = v21;
      if ( *v16 != -8 || v16[1] != 0x7FFFFFFFFFFFFFFFLL )
        --HIDWORD(v107);
      *v16 = v9;
      v16[1] = v10;
      v16[2] = v98;
LABEL_31:
      ++v98;
LABEL_32:
      ;
    }
    while ( v93 != v98 );
    if ( v6 )
      sub_1996C50(v94, v88, a1 + 32128);
LABEL_35:
    ++v105;
    if ( (_DWORD)v107 )
    {
      v59 = 4 * v107;
      v22 = v108;
      if ( (unsigned int)(4 * v107) < 0x40 )
        v59 = 64;
      if ( v108 <= v59 )
      {
LABEL_38:
        v23 = v106;
        v24 = &v106[3 * v22];
        if ( v106 != v24 )
        {
          do
          {
            *v23 = -8;
            v23 += 3;
            *(v23 - 2) = 0x7FFFFFFFFFFFFFFFLL;
          }
          while ( v24 != v23 );
        }
        goto LABEL_40;
      }
      v60 = v106;
      if ( (_DWORD)v107 == 1 )
      {
        v63 = 86;
      }
      else
      {
        _BitScanReverse(&v61, v107 - 1);
        v62 = 1 << (33 - (v61 ^ 0x1F));
        if ( v62 < 64 )
          v62 = 64;
        if ( v108 == v62 )
        {
          v107 = 0;
          v82 = &v106[3 * v108];
          do
          {
            if ( v60 )
            {
              *v60 = -8;
              v60[1] = 0x7FFFFFFFFFFFFFFFLL;
            }
            v60 += 3;
          }
          while ( v82 != v60 );
          goto LABEL_41;
        }
        v63 = 4 * v62 / 3u + 1;
      }
      j___libc_free_0(v106);
      v64 = sub_1454B60(v63);
      v108 = v64;
      if ( !v64 )
        goto LABEL_125;
      v65 = (_QWORD *)sub_22077B0(24LL * v64);
      v107 = 0;
      v106 = v65;
      for ( k = &v65[3 * v108]; k != v65; v65 += 3 )
      {
        if ( v65 )
        {
          *v65 = -8;
          v65[1] = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
    }
    else if ( HIDWORD(v107) )
    {
      v22 = v108;
      if ( v108 <= 0x40 )
        goto LABEL_38;
      j___libc_free_0(v106);
      v108 = 0;
LABEL_125:
      v106 = 0;
LABEL_40:
      v107 = 0;
    }
LABEL_41:
    if ( ++v88 != v87 )
    {
      v2 = *(_QWORD *)(a1 + 368);
      continue;
    }
    break;
  }
  if ( v117 != v116 )
    _libc_free((unsigned __int64)v117);
  v20 = v110;
LABEL_22:
  j___libc_free_0(v20);
  return j___libc_free_0(v106);
}
