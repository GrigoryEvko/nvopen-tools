// Function: sub_1927200
// Address: 0x1927200
//
__int64 __fastcall sub_1927200(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 *v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  __int64 *v20; // rcx
  __int64 *v21; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 *v26; // rax
  char v27; // r8
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  char v35; // si
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rax
  __int64 *v38; // r12
  __int64 v39; // r14
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned int v42; // edx
  __int64 *v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r13
  __int64 j; // r12
  int v47; // edx
  int v48; // ecx
  __int64 v49; // r10
  int v50; // r9d
  int v51; // r11d
  unsigned __int64 v52; // r8
  unsigned __int64 v53; // r8
  unsigned int k; // eax
  __int64 v55; // rbx
  int v56; // r8d
  __int64 *v57; // rax
  char v58; // dl
  __int64 v59; // r13
  __int64 *v60; // rax
  __int64 *v61; // rcx
  __int64 *v62; // rsi
  unsigned __int64 v63; // rdx
  char v64; // cl
  char v65; // si
  __int64 v66; // rbx
  __int64 v67; // r12
  unsigned __int64 v68; // rdi
  __int64 v69; // rax
  _QWORD *v70; // rax
  __int64 v71; // rdx
  int i; // ecx
  int v73; // ecx
  int v74; // edx
  int v75; // r9d
  unsigned __int64 v76; // r8
  unsigned __int64 v77; // r8
  int v78; // eax
  __int64 v79; // r8
  unsigned int v80; // eax
  int v81; // r10d
  unsigned int v82; // eax
  int v83; // edx
  int v84; // eax
  int v85; // ecx
  int v86; // edx
  int v87; // r9d
  unsigned __int64 v88; // r8
  unsigned __int64 v89; // r8
  unsigned int m; // eax
  int *v91; // r8
  int v92; // r10d
  unsigned int v93; // esi
  unsigned int v94; // r9d
  unsigned int v95; // eax
  unsigned int v96; // eax
  int v97; // r9d
  __int64 v101; // [rsp+30h] [rbp-350h] BYREF
  __int64 v102; // [rsp+38h] [rbp-348h]
  __int64 v103; // [rsp+40h] [rbp-340h]
  unsigned int v104; // [rsp+48h] [rbp-338h]
  _QWORD v105[16]; // [rsp+50h] [rbp-330h] BYREF
  __int64 v106; // [rsp+D0h] [rbp-2B0h] BYREF
  _QWORD *v107; // [rsp+D8h] [rbp-2A8h]
  _QWORD *v108; // [rsp+E0h] [rbp-2A0h]
  __int64 v109; // [rsp+E8h] [rbp-298h]
  int v110; // [rsp+F0h] [rbp-290h]
  _QWORD v111[8]; // [rsp+F8h] [rbp-288h] BYREF
  unsigned __int64 v112; // [rsp+138h] [rbp-248h] BYREF
  unsigned __int64 v113; // [rsp+140h] [rbp-240h]
  unsigned __int64 v114; // [rsp+148h] [rbp-238h]
  __int64 v115; // [rsp+150h] [rbp-230h] BYREF
  __int64 *v116; // [rsp+158h] [rbp-228h]
  __int64 *v117; // [rsp+160h] [rbp-220h]
  unsigned int v118; // [rsp+168h] [rbp-218h]
  unsigned int v119; // [rsp+16Ch] [rbp-214h]
  int v120; // [rsp+170h] [rbp-210h]
  _BYTE v121[64]; // [rsp+178h] [rbp-208h] BYREF
  unsigned __int64 v122; // [rsp+1B8h] [rbp-1C8h] BYREF
  unsigned __int64 v123; // [rsp+1C0h] [rbp-1C0h]
  unsigned __int64 v124; // [rsp+1C8h] [rbp-1B8h]
  __int64 v125; // [rsp+1D0h] [rbp-1B0h] BYREF
  __int64 v126; // [rsp+1D8h] [rbp-1A8h]
  unsigned __int64 v127; // [rsp+1E0h] [rbp-1A0h]
  _BYTE v128[64]; // [rsp+1F8h] [rbp-188h] BYREF
  unsigned __int64 v129; // [rsp+238h] [rbp-148h]
  unsigned __int64 v130; // [rsp+240h] [rbp-140h]
  unsigned __int64 v131; // [rsp+248h] [rbp-138h]
  _QWORD v132[2]; // [rsp+250h] [rbp-130h] BYREF
  unsigned __int64 v133; // [rsp+260h] [rbp-120h]
  char v134[64]; // [rsp+278h] [rbp-108h] BYREF
  __int64 *v135; // [rsp+2B8h] [rbp-C8h]
  __int64 *v136; // [rsp+2C0h] [rbp-C0h]
  unsigned __int64 v137; // [rsp+2C8h] [rbp-B8h]
  _QWORD v138[2]; // [rsp+2D0h] [rbp-B0h] BYREF
  unsigned __int64 v139; // [rsp+2E0h] [rbp-A0h]
  char v140[64]; // [rsp+2F8h] [rbp-88h] BYREF
  unsigned __int64 v141; // [rsp+338h] [rbp-48h]
  unsigned __int64 v142; // [rsp+340h] [rbp-40h]
  unsigned __int64 v143; // [rsp+348h] [rbp-38h]

  result = *(_QWORD *)(a1 + 224);
  v4 = *(unsigned int *)(result + 72);
  if ( !(_DWORD)v4 )
    return result;
  v5 = *(__int64 **)(result + 56);
  v6 = *v5;
  result = (__int64)v5;
  if ( *v5 )
  {
    result = 1;
    v93 = 0;
    while ( v6 != -8 )
    {
      v94 = result + 1;
      v93 = (v4 - 1) & (result + v93);
      result = (__int64)&v5[2 * v93];
      v6 = *(_QWORD *)result;
      if ( !*(_QWORD *)result )
        goto LABEL_3;
      result = v94;
    }
    return result;
  }
LABEL_3:
  if ( (__int64 *)result == &v5[2 * v4] )
    return result;
  v7 = *(_QWORD *)(result + 8);
  if ( !v7 )
    return result;
  v111[0] = *(_QWORD *)(result + 8);
  memset(v105, 0, sizeof(v105));
  v105[1] = &v105[5];
  v105[2] = &v105[5];
  v109 = 0x100000008LL;
  v107 = v111;
  v108 = v111;
  v132[0] = v7;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  LODWORD(v105[3]) = 8;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v110 = 0;
  v106 = 1;
  LOBYTE(v133) = 0;
  sub_13B8390(&v112, (__int64)v132);
  sub_16CCEE0(&v125, (__int64)v128, 8, (__int64)v105);
  v8 = v105[13];
  memset(&v105[13], 0, 24);
  v129 = v8;
  v130 = v105[14];
  v131 = v105[15];
  sub_16CCEE0(&v115, (__int64)v121, 8, (__int64)&v106);
  v9 = v112;
  v112 = 0;
  v122 = v9;
  v10 = v113;
  v113 = 0;
  v123 = v10;
  v11 = v114;
  v114 = 0;
  v124 = v11;
  sub_16CCEE0(v132, (__int64)v134, 8, (__int64)&v115);
  v12 = v122;
  v122 = 0;
  v135 = (__int64 *)v12;
  v13 = v123;
  v123 = 0;
  v136 = (__int64 *)v13;
  v14 = v124;
  v124 = 0;
  v137 = v14;
  sub_16CCEE0(v138, (__int64)v140, 8, (__int64)&v125);
  v15 = v129;
  v129 = 0;
  v141 = v15;
  v16 = v130;
  v130 = 0;
  v142 = v16;
  v17 = v131;
  v131 = 0;
  v143 = v17;
  if ( v122 )
    j_j___libc_free_0(v122, v124 - v122);
  if ( v117 != v116 )
    _libc_free((unsigned __int64)v117);
  if ( v129 )
    j_j___libc_free_0(v129, v131 - v129);
  if ( v127 != v126 )
    _libc_free(v127);
  if ( v112 )
    j_j___libc_free_0(v112, v114 - v112);
  if ( v108 != v107 )
    _libc_free((unsigned __int64)v108);
  if ( v105[13] )
    j_j___libc_free_0(v105[13], v105[15] - v105[13]);
  if ( v105[2] != v105[1] )
    _libc_free(v105[2]);
  v18 = v121;
  sub_16CCCB0(&v115, (__int64)v121, (__int64)v132);
  v20 = v136;
  v21 = v135;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v22 = (char *)v136 - (char *)v135;
  if ( v136 == v135 )
  {
    v22 = 0;
    v24 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_169;
    v23 = sub_22077B0((char *)v136 - (char *)v135);
    v20 = v136;
    v21 = v135;
    v24 = v23;
  }
  v122 = v24;
  v123 = v24;
  v124 = v24 + v22;
  if ( v20 != v21 )
  {
    v25 = v24;
    v26 = v21;
    do
    {
      if ( v25 )
      {
        *(_QWORD *)v25 = *v26;
        v27 = *((_BYTE *)v26 + 16);
        *(_BYTE *)(v25 + 16) = v27;
        if ( v27 )
          *(_QWORD *)(v25 + 8) = v26[1];
      }
      v26 += 3;
      v25 += 24LL;
    }
    while ( v20 != v26 );
    v24 += 8 * ((unsigned __int64)((char *)(v20 - 3) - (char *)v21) >> 3) + 24;
  }
  v123 = v24;
  v21 = &v125;
  v18 = v128;
  sub_16CCCB0(&v125, (__int64)v128, (__int64)v138);
  v28 = v142;
  v29 = v141;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v30 = v142 - v141;
  if ( v142 != v141 )
  {
    if ( v30 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v31 = sub_22077B0(v142 - v141);
      v28 = v142;
      v29 = v141;
      v32 = v31;
      goto LABEL_34;
    }
LABEL_169:
    sub_4261EA(v21, v18, v19);
  }
  v32 = 0;
LABEL_34:
  v129 = v32;
  v33 = v32;
  v130 = v32;
  v131 = v32 + v30;
  if ( v28 != v29 )
  {
    v34 = v29;
    do
    {
      if ( v33 )
      {
        *(_QWORD *)v33 = *(_QWORD *)v34;
        v35 = *(_BYTE *)(v34 + 16);
        *(_BYTE *)(v33 + 16) = v35;
        if ( v35 )
          *(_QWORD *)(v33 + 8) = *(_QWORD *)(v34 + 8);
      }
      v34 += 24LL;
      v33 += 24LL;
    }
    while ( v28 != v34 );
    v33 = v32 + 8 * ((v28 - 24 - v29) >> 3) + 24;
  }
  v36 = v123;
  v37 = v122;
  v130 = v33;
  if ( v123 - v122 == v33 - v32 )
    goto LABEL_75;
  do
  {
LABEL_42:
    v38 = *(__int64 **)(v36 - 24);
    v39 = *v38;
    if ( *v38 )
    {
      v40 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v40 )
        goto LABEL_116;
      v41 = *(_QWORD *)(a2 + 8);
      v42 = (v40 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v43 = (__int64 *)(v41 + 56LL * v42);
      v44 = *v43;
      if ( v39 != *v43 )
      {
        for ( i = 1; ; i = v97 )
        {
          if ( v44 == -8 )
            goto LABEL_116;
          v97 = i + 1;
          v42 = (v40 - 1) & (i + v42);
          v43 = (__int64 *)(v41 + 56LL * v42);
          v44 = *v43;
          if ( v39 == *v43 )
            break;
        }
      }
      if ( v43 != (__int64 *)(v41 + 56 * v40) )
      {
        v45 = v43[1];
        for ( j = v45 + 16LL * *((unsigned int *)v43 + 4); v45 != j; ++*(_DWORD *)(v55 + 16) )
        {
          if ( !v104 )
          {
            ++v101;
            goto LABEL_122;
          }
          v47 = *(_DWORD *)(j - 16);
          v48 = *(_DWORD *)(j - 12);
          v49 = 0;
          v50 = v104 - 1;
          v51 = 1;
          v52 = ((((unsigned int)(37 * v48) | ((unsigned __int64)(unsigned int)(37 * v47) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v48) << 32)) >> 22)
              ^ (((unsigned int)(37 * v48) | ((unsigned __int64)(unsigned int)(37 * v47) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v48) << 32));
          v53 = ((9 * (((v52 - 1 - (v52 << 13)) >> 8) ^ (v52 - 1 - (v52 << 13)))) >> 15)
              ^ (9 * (((v52 - 1 - (v52 << 13)) >> 8) ^ (v52 - 1 - (v52 << 13))));
          for ( k = (v104 - 1) & (((v53 - 1 - (v53 << 27)) >> 31) ^ (v53 - 1 - ((_DWORD)v53 << 27))); ; k = v50 & v82 )
          {
            v55 = v102 + 40LL * k;
            v56 = *(_DWORD *)v55;
            if ( *(_DWORD *)v55 == v47 && *(_DWORD *)(v55 + 4) == v48 )
            {
              v69 = *(unsigned int *)(v55 + 16);
              if ( (unsigned int)v69 >= *(_DWORD *)(v55 + 20) )
              {
                sub_16CD150(v55 + 8, (const void *)(v55 + 24), 0, 8, v56, v50);
                v70 = (_QWORD *)(*(_QWORD *)(v55 + 8) + 8LL * *(unsigned int *)(v55 + 16));
              }
              else
              {
                v70 = (_QWORD *)(*(_QWORD *)(v55 + 8) + 8 * v69);
              }
              goto LABEL_115;
            }
            if ( v56 == -1 )
              break;
            if ( v56 == -2 && *(_DWORD *)(v55 + 4) == -2 && !v49 )
              v49 = v102 + 40LL * k;
LABEL_135:
            v82 = v51 + k;
            ++v51;
          }
          if ( *(_DWORD *)(v55 + 4) != -1 )
            goto LABEL_135;
          if ( v49 )
            v55 = v49;
          ++v101;
          v83 = v103 + 1;
          if ( 4 * ((int)v103 + 1) < 3 * v104 )
          {
            if ( v104 - HIDWORD(v103) - v83 > v104 >> 3 )
              goto LABEL_139;
            sub_1926F00((__int64)&v101, v104);
            if ( v104 )
            {
              v85 = *(_DWORD *)(j - 16);
              v86 = *(_DWORD *)(j - 12);
              v55 = 0;
              v87 = 1;
              v88 = ((((unsigned int)(37 * v86) | ((unsigned __int64)(unsigned int)(37 * v85) << 32))
                    - 1
                    - ((unsigned __int64)(unsigned int)(37 * v86) << 32)) >> 22)
                  ^ (((unsigned int)(37 * v86) | ((unsigned __int64)(unsigned int)(37 * v85) << 32))
                   - 1
                   - ((unsigned __int64)(unsigned int)(37 * v86) << 32));
              v89 = ((9 * (((v88 - 1 - (v88 << 13)) >> 8) ^ (v88 - 1 - (v88 << 13)))) >> 15)
                  ^ (9 * (((v88 - 1 - (v88 << 13)) >> 8) ^ (v88 - 1 - (v88 << 13))));
              for ( m = (v104 - 1) & (((v89 - 1 - (v89 << 27)) >> 31) ^ (v89 - 1 - ((_DWORD)v89 << 27)));
                    ;
                    m = (v104 - 1) & v95 )
              {
                v91 = (int *)(v102 + 40LL * m);
                v92 = *v91;
                if ( *v91 == v85 && v91[1] == v86 )
                {
                  v55 = v102 + 40LL * m;
                  v83 = v103 + 1;
                  goto LABEL_139;
                }
                if ( v92 == -1 )
                {
                  if ( v91[1] == -1 )
                  {
                    if ( !v55 )
                      v55 = v102 + 40LL * m;
                    v83 = v103 + 1;
                    goto LABEL_139;
                  }
                }
                else if ( v92 == -2 && v91[1] == -2 && !v55 )
                {
                  v55 = v102 + 40LL * m;
                }
                v95 = v87 + m;
                ++v87;
              }
            }
LABEL_178:
            LODWORD(v103) = v103 + 1;
            BUG();
          }
LABEL_122:
          sub_1926F00((__int64)&v101, 2 * v104);
          if ( !v104 )
            goto LABEL_178;
          v73 = *(_DWORD *)(j - 16);
          v74 = *(_DWORD *)(j - 12);
          v75 = 1;
          v76 = ((((unsigned int)(37 * v74) | ((unsigned __int64)(unsigned int)(37 * v73) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v74) << 32)) >> 22)
              ^ (((unsigned int)(37 * v74) | ((unsigned __int64)(unsigned int)(37 * v73) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v74) << 32));
          v77 = ((9 * (((v76 - 1 - (v76 << 13)) >> 8) ^ (v76 - 1 - (v76 << 13)))) >> 15)
              ^ (9 * (((v76 - 1 - (v76 << 13)) >> 8) ^ (v76 - 1 - (v76 << 13))));
          v78 = ((v77 - 1 - (v77 << 27)) >> 31) ^ (v77 - 1 - ((_DWORD)v77 << 27));
          v79 = 0;
          v80 = (v104 - 1) & v78;
          while ( 2 )
          {
            v55 = v102 + 40LL * v80;
            v81 = *(_DWORD *)v55;
            if ( *(_DWORD *)v55 == v73 && *(_DWORD *)(v55 + 4) == v74 )
            {
              v83 = v103 + 1;
              goto LABEL_139;
            }
            if ( v81 != -1 )
            {
              if ( v81 == -2 && *(_DWORD *)(v55 + 4) == -2 && !v79 )
                v79 = v102 + 40LL * v80;
              goto LABEL_168;
            }
            if ( *(_DWORD *)(v55 + 4) != -1 )
            {
LABEL_168:
              v96 = v75 + v80;
              ++v75;
              v80 = (v104 - 1) & v96;
              continue;
            }
            break;
          }
          if ( v79 )
            v55 = v79;
          v83 = v103 + 1;
LABEL_139:
          LODWORD(v103) = v83;
          if ( *(_DWORD *)v55 != -1 || *(_DWORD *)(v55 + 4) != -1 )
            --HIDWORD(v103);
          *(_DWORD *)v55 = *(_DWORD *)(j - 16);
          v84 = *(_DWORD *)(j - 12);
          *(_QWORD *)(v55 + 16) = 0x200000000LL;
          *(_DWORD *)(v55 + 4) = v84;
          v70 = (_QWORD *)(v55 + 24);
          *(_QWORD *)(v55 + 8) = v55 + 24;
LABEL_115:
          v71 = *(_QWORD *)(j - 8);
          j -= 16;
          *v70 = v71;
        }
      }
LABEL_116:
      sub_1921490(a1, v39, a3, (__int64)&v101);
      v36 = v123;
      v38 = *(__int64 **)(v123 - 24);
    }
LABEL_60:
    if ( !*(_BYTE *)(v36 - 8) )
    {
      v57 = (__int64 *)v38[3];
      *(_BYTE *)(v36 - 8) = 1;
      *(_QWORD *)(v36 - 16) = v57;
      goto LABEL_64;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v57 = *(__int64 **)(v36 - 16);
LABEL_64:
        if ( v57 == (__int64 *)v38[4] )
        {
          v123 -= 24LL;
          v37 = v122;
          v36 = v123;
          if ( v123 == v122 )
            goto LABEL_74;
          v38 = *(__int64 **)(v123 - 24);
          goto LABEL_60;
        }
        *(_QWORD *)(v36 - 16) = v57 + 1;
        v59 = *v57;
        v60 = v116;
        if ( v117 == v116 )
          break;
LABEL_62:
        sub_16CCBA0((__int64)&v115, v59);
        if ( v58 )
          goto LABEL_73;
      }
      v61 = &v116[v119];
      if ( v116 == v61 )
        break;
      v62 = 0;
      while ( v59 != *v60 )
      {
        if ( *v60 == -2 )
        {
          v62 = v60;
          if ( v61 == v60 + 1 )
            goto LABEL_72;
          ++v60;
        }
        else if ( v61 == ++v60 )
        {
          if ( !v62 )
            goto LABEL_110;
LABEL_72:
          *v62 = v59;
          --v120;
          ++v115;
          goto LABEL_73;
        }
      }
    }
LABEL_110:
    if ( v119 >= v118 )
      goto LABEL_62;
    ++v119;
    *v61 = v59;
    ++v115;
LABEL_73:
    v106 = v59;
    LOBYTE(v108) = 0;
    sub_13B8390(&v122, (__int64)&v106);
    v37 = v122;
    v36 = v123;
LABEL_74:
    v32 = v129;
  }
  while ( v36 - v37 != v130 - v129 );
LABEL_75:
  if ( v36 != v37 )
  {
    v63 = v32;
    while ( *(_QWORD *)v37 == *(_QWORD *)v63 )
    {
      v64 = *(_BYTE *)(v37 + 16);
      v65 = *(_BYTE *)(v63 + 16);
      if ( v64 && v65 )
      {
        if ( *(_QWORD *)(v37 + 8) != *(_QWORD *)(v63 + 8) )
          goto LABEL_42;
        v37 += 24LL;
        v63 += 24LL;
        if ( v36 == v37 )
          goto LABEL_82;
      }
      else
      {
        if ( v65 != v64 )
          goto LABEL_42;
        v37 += 24LL;
        v63 += 24LL;
        if ( v36 == v37 )
          goto LABEL_82;
      }
    }
    goto LABEL_42;
  }
LABEL_82:
  if ( v32 )
    j_j___libc_free_0(v32, v131 - v32);
  if ( v127 != v126 )
    _libc_free(v127);
  if ( v122 )
    j_j___libc_free_0(v122, v124 - v122);
  if ( v117 != v116 )
    _libc_free((unsigned __int64)v117);
  if ( v141 )
    j_j___libc_free_0(v141, v143 - v141);
  if ( v139 != v138[1] )
    _libc_free(v139);
  if ( v135 )
    j_j___libc_free_0(v135, v137 - (_QWORD)v135);
  if ( v133 != v132[1] )
    _libc_free(v133);
  if ( v104 )
  {
    v66 = v102;
    v67 = v102 + 40LL * v104;
    do
    {
      if ( *(_DWORD *)v66 == -1 )
      {
        if ( *(_DWORD *)(v66 + 4) != -1 )
          goto LABEL_101;
      }
      else if ( *(_DWORD *)v66 != -2 || *(_DWORD *)(v66 + 4) != -2 )
      {
LABEL_101:
        v68 = *(_QWORD *)(v66 + 8);
        if ( v68 != v66 + 24 )
          _libc_free(v68);
      }
      v66 += 40;
    }
    while ( v67 != v66 );
  }
  return j___libc_free_0(v102);
}
