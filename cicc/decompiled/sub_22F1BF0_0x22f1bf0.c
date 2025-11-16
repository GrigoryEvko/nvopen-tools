// Function: sub_22F1BF0
// Address: 0x22f1bf0
//
void __fastcall sub_22F1BF0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned int v8; // eax
  _BYTE *v9; // r14
  _BYTE *v10; // r15
  __int64 v11; // r14
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 *v19; // rbx
  __int64 v20; // r9
  __int64 v21; // r8
  int v22; // r11d
  unsigned int v23; // edx
  __int64 **v24; // rcx
  __int64 **v25; // rax
  __int64 *v26; // rdi
  int v27; // edi
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r9
  _QWORD *v32; // rdx
  __int64 v33; // r15
  unsigned int v34; // eax
  int v35; // ecx
  unsigned int v36; // edx
  __int64 v37; // rcx
  __int64 v38; // rdi
  __int64 v39; // rsi
  int v40; // edx
  __int64 v41; // rdx
  __int64 v42; // rcx
  unsigned int v43; // eax
  __int64 v44; // rax
  unsigned int v45; // ecx
  unsigned int v46; // edx
  _QWORD *v47; // rdi
  unsigned int v48; // esi
  __int64 *v49; // r13
  __int64 v50; // rax
  unsigned __int64 v51; // r8
  __int64 v52; // r14
  __int64 v53; // r14
  __int64 v54; // r12
  __int64 v55; // r13
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  _QWORD *v58; // rcx
  unsigned __int64 i; // rax
  __int64 *v60; // rbx
  __int64 *v61; // r12
  __int64 v62; // r8
  _QWORD *v63; // r10
  int v64; // r11d
  unsigned int v65; // eax
  _QWORD *v66; // rdi
  __int64 v67; // rcx
  unsigned int v68; // esi
  int v69; // ecx
  int v70; // ecx
  __int64 v71; // rdx
  __int64 v72; // rdi
  int v73; // eax
  __int64 v74; // r14
  __int64 v75; // rax
  unsigned __int64 v76; // rax
  __int64 v77; // r12
  int v78; // ebx
  unsigned int v79; // r14d
  int v80; // edx
  unsigned int v81; // eax
  __int64 v82; // rsi
  __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // r13
  int v86; // eax
  __int64 v87; // r8
  __int64 v88; // rax
  unsigned __int64 v89; // rdx
  int v90; // eax
  int v91; // ecx
  int v92; // ecx
  int v93; // r14d
  __int64 v94; // rdx
  __int64 v95; // rdi
  int v96; // edi
  __int64 v97; // rsi
  int v98; // r10d
  int v99; // r8d
  unsigned int v100; // ecx
  __int64 *v101; // r11
  int v102; // edi
  __int64 **v103; // rsi
  int v104; // r8d
  int v105; // ecx
  unsigned int v106; // r12d
  __int64 **v107; // rdi
  __int64 *v108; // rsi
  int v109; // r14d
  _BYTE *v110; // [rsp+0h] [rbp-1F0h]
  const void *v111; // [rsp+10h] [rbp-1E0h]
  __int64 *v112; // [rsp+38h] [rbp-1B8h]
  _BYTE *v113; // [rsp+40h] [rbp-1B0h]
  __int64 *v114; // [rsp+48h] [rbp-1A8h]
  _QWORD *v115; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v116; // [rsp+58h] [rbp-198h]
  _QWORD v117[4]; // [rsp+60h] [rbp-190h] BYREF
  _BYTE *v118; // [rsp+80h] [rbp-170h] BYREF
  __int64 v119; // [rsp+88h] [rbp-168h]
  _BYTE v120[64]; // [rsp+90h] [rbp-160h] BYREF
  __int64 v121; // [rsp+D0h] [rbp-120h]
  __int64 v122; // [rsp+D8h] [rbp-118h]
  __int64 v123; // [rsp+E0h] [rbp-110h]
  __int64 v124; // [rsp+E8h] [rbp-108h]
  char *v125; // [rsp+F0h] [rbp-100h]
  __int64 v126; // [rsp+F8h] [rbp-F8h]
  char v127; // [rsp+100h] [rbp-F0h] BYREF
  __int64 *v128; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v129; // [rsp+128h] [rbp-C8h]
  _BYTE v130[64]; // [rsp+130h] [rbp-C0h] BYREF
  _QWORD *v131; // [rsp+170h] [rbp-80h] BYREF
  __int64 v132; // [rsp+178h] [rbp-78h]
  _QWORD v133[14]; // [rsp+180h] [rbp-70h] BYREF

  v3 = a2 + 72;
  *a1 = a3;
  v5 = *(_QWORD *)(a2 + 80);
  if ( v5 != a2 + 72 )
  {
    do
    {
      while ( 1 )
      {
        if ( v5 )
        {
          v6 = v5 - 24;
          v7 = (unsigned int)(*(_DWORD *)(v5 + 20) + 1);
          v8 = *(_DWORD *)(v5 + 20) + 1;
        }
        else
        {
          v6 = 0;
          v7 = 0;
          v8 = 0;
        }
        if ( v8 >= *(_DWORD *)(a3 + 32) || !*(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v7) )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 == v3 )
          goto LABEL_9;
      }
      v131 = (_QWORD *)v6;
      sub_22EE9A0((__int64)(a1 + 1), (__int64 *)&v131);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v5 != v3 );
  }
LABEL_9:
  v118 = v120;
  v119 = 0x800000000LL;
  sub_22ED620((__int64)&v118, a2);
  v9 = &v118[8 * (unsigned int)v119];
  v113 = v118;
  if ( v118 != v9 )
  {
    v10 = &v118[8 * (unsigned int)v119];
    v11 = (__int64)a1;
    while ( 1 )
    {
      v12 = *((_QWORD *)v10 - 1);
      v13 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v13 == v12 + 48 || !v13 || (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
        BUG();
      if ( *(_BYTE *)(v13 - 24) != 31 )
        goto LABEL_25;
      if ( (*(_DWORD *)(v13 - 20) & 0x7FFFFFF) != 3 )
        goto LABEL_25;
      v14 = *(_QWORD *)(v13 - 120);
      if ( *(_BYTE *)v14 > 0x15u || *(_QWORD *)(v13 - 88) == *(_QWORD *)(v13 - 56) || *(_BYTE *)v14 != 17 )
        goto LABEL_25;
      if ( *(_DWORD *)(v14 + 32) <= 0x40u )
        v15 = *(_QWORD *)(v14 + 24);
      else
        v15 = **(_QWORD **)(v14 + 24);
      v16 = v13 - 120;
      v17 = v15 == 0 ? 64LL : 32LL;
      if ( (*(_BYTE *)(v13 - 17) & 0x40) != 0 )
        v16 = *(_QWORD *)(v13 - 32);
      v18 = *(_DWORD *)(v11 + 80);
      v19 = (__int64 *)(v17 + v16);
      v111 = (const void *)(v11 + 56);
      if ( !v18 )
        break;
      v20 = v18 - 1;
      v21 = *(_QWORD *)(v11 + 64);
      v22 = 1;
      v23 = v20 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v24 = (__int64 **)(v21 + 8LL * v23);
      v25 = 0;
      v26 = *v24;
      if ( v19 == *v24 )
      {
LABEL_25:
        v10 -= 8;
        if ( v113 == v10 )
          goto LABEL_26;
      }
      else
      {
        while ( v26 != (__int64 *)-4096LL )
        {
          if ( v26 == (__int64 *)-8192LL && !v25 )
            v25 = v24;
          v23 = v20 & (v22 + v23);
          v24 = (__int64 **)(v21 + 8LL * v23);
          v26 = *v24;
          if ( v19 == *v24 )
            goto LABEL_25;
          ++v22;
        }
        v27 = *(_DWORD *)(v11 + 72);
        if ( !v25 )
          v25 = v24;
        ++*(_QWORD *)(v11 + 56);
        v28 = v27 + 1;
        if ( 4 * (v27 + 1) < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(v11 + 76) - v28 <= v18 >> 3 )
          {
            sub_22F1A20((__int64)v111, v18);
            v104 = *(_DWORD *)(v11 + 80);
            if ( !v104 )
            {
LABEL_165:
              ++*(_DWORD *)(v11 + 72);
              BUG();
            }
            v21 = (unsigned int)(v104 - 1);
            v20 = *(_QWORD *)(v11 + 64);
            v105 = 1;
            v106 = v21 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
            v28 = *(_DWORD *)(v11 + 72) + 1;
            v107 = 0;
            v25 = (__int64 **)(v20 + 8LL * v106);
            v108 = *v25;
            if ( v19 != *v25 )
            {
              while ( v108 != (__int64 *)-4096LL )
              {
                if ( !v107 && v108 == (__int64 *)-8192LL )
                  v107 = v25;
                v106 = v21 & (v105 + v106);
                v25 = (__int64 **)(v20 + 8LL * v106);
                v108 = *v25;
                if ( v19 == *v25 )
                  goto LABEL_42;
                ++v105;
              }
              if ( v107 )
                v25 = v107;
            }
          }
          goto LABEL_42;
        }
LABEL_129:
        sub_22F1A20((__int64)v111, 2 * v18);
        v99 = *(_DWORD *)(v11 + 80);
        if ( !v99 )
          goto LABEL_165;
        v21 = (unsigned int)(v99 - 1);
        v20 = *(_QWORD *)(v11 + 64);
        v100 = v21 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v28 = *(_DWORD *)(v11 + 72) + 1;
        v25 = (__int64 **)(v20 + 8LL * v100);
        v101 = *v25;
        if ( v19 != *v25 )
        {
          v102 = 1;
          v103 = 0;
          while ( v101 != (__int64 *)-4096LL )
          {
            if ( !v103 && v101 == (__int64 *)-8192LL )
              v103 = v25;
            v100 = v21 & (v102 + v100);
            v25 = (__int64 **)(v20 + 8LL * v100);
            v101 = *v25;
            if ( v19 == *v25 )
              goto LABEL_42;
            ++v102;
          }
          if ( v103 )
            v25 = v103;
        }
LABEL_42:
        *(_DWORD *)(v11 + 72) = v28;
        if ( *v25 != (__int64 *)-4096LL )
          --*(_DWORD *)(v11 + 76);
        *v25 = v19;
        v29 = *(unsigned int *)(v11 + 96);
        if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 100) )
        {
          sub_C8D5F0(v11 + 88, (const void *)(v11 + 104), v29 + 1, 8u, v21, v20);
          v29 = *(unsigned int *)(v11 + 96);
        }
        *(_QWORD *)(*(_QWORD *)(v11 + 88) + 8 * v29) = v19;
        ++*(_DWORD *)(v11 + 96);
        v30 = *v19;
        if ( (unsigned __int8)sub_22ECC70(v11, v30) )
          goto LABEL_25;
        v32 = v117;
        v110 = v10;
        v33 = v11;
        v125 = &v127;
        v126 = 0x400000000LL;
        v116 = 0x400000001LL;
        v34 = 1;
        v115 = v117;
        v121 = 0;
        v122 = 0;
        v123 = 0;
        v124 = 0;
        v117[0] = v30;
        while ( 1 )
        {
          v37 = v34--;
          v38 = *(_QWORD *)(v33 + 16);
          v39 = v32[v37 - 1];
          v40 = *(_DWORD *)(v33 + 32);
          LODWORD(v116) = v34;
          if ( v40 )
          {
            v35 = v40 - 1;
            v36 = (v40 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            v31 = *(_QWORD *)(v38 + 8LL * v36);
            if ( v39 == v31 )
              goto LABEL_49;
            v98 = 1;
            if ( v31 != -4096 )
            {
              while ( 1 )
              {
                v36 = v35 & (v98 + v36);
                v31 = *(_QWORD *)(v38 + 8LL * v36);
                if ( v39 == v31 )
                  break;
                ++v98;
                if ( v31 == -4096 )
                  goto LABEL_52;
              }
LABEL_49:
              if ( !v34 )
                break;
              goto LABEL_50;
            }
          }
LABEL_52:
          v41 = *(_QWORD *)v33;
          v42 = 0;
          v128 = (__int64 *)v130;
          v129 = 0x800000000LL;
          v43 = 0;
          if ( v39 )
          {
            v42 = (unsigned int)(*(_DWORD *)(v39 + 44) + 1);
            v43 = *(_DWORD *)(v39 + 44) + 1;
          }
          if ( v43 < *(_DWORD *)(v41 + 32) )
          {
            v44 = *(_QWORD *)(v41 + 24);
            if ( *(_QWORD *)(v44 + 8 * v42) )
            {
              v133[0] = *(_QWORD *)(v44 + 8 * v42);
              v45 = 8;
              v46 = 0;
              v131 = v133;
              v47 = v133;
              v48 = 1;
              v132 = 0x800000001LL;
              while ( 1 )
              {
                v49 = (__int64 *)v47[v48 - 1];
                v50 = v46;
                LODWORD(v132) = v48 - 1;
                v51 = v46 + 1LL;
                v52 = *v49;
                if ( v51 > v45 )
                {
                  sub_C8D5F0((__int64)&v128, v130, v46 + 1LL, 8u, v51, v31);
                  v50 = (unsigned int)v129;
                }
                v128[v50] = v52;
                v53 = v49[3];
                v54 = *((unsigned int *)v49 + 8);
                v55 = 8 * v54;
                v56 = (unsigned int)v132;
                LODWORD(v129) = v129 + 1;
                v57 = v54 + (unsigned int)v132;
                if ( v57 > HIDWORD(v132) )
                {
                  sub_C8D5F0((__int64)&v131, v133, v57, 8u, v51, v31);
                  v56 = (unsigned int)v132;
                }
                v47 = v131;
                v58 = &v131[v56];
                if ( v55 )
                {
                  for ( i = 0; i != v55; i += 8LL )
                    v58[i / 8] = *(_QWORD *)(v53 + i);
                  v47 = v131;
                  LODWORD(v56) = v132;
                }
                LODWORD(v132) = v54 + v56;
                v48 = v54 + v56;
                if ( !((_DWORD)v54 + (_DWORD)v56) )
                  break;
                v46 = v129;
                v45 = HIDWORD(v129);
              }
              if ( v47 != v133 )
                _libc_free((unsigned __int64)v47);
              v60 = v128;
              v61 = &v128[(unsigned int)v129];
              v112 = v61;
              if ( v128 != v61 )
              {
                while ( 1 )
                {
                  v68 = *(_DWORD *)(v33 + 32);
                  if ( !v68 )
                    break;
                  v31 = v68 - 1;
                  v62 = *(_QWORD *)(v33 + 16);
                  v63 = 0;
                  v64 = 1;
                  v65 = v31 & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
                  v66 = (_QWORD *)(v62 + 8LL * v65);
                  v67 = *v66;
                  if ( *v66 == *v60 )
                  {
LABEL_80:
                    if ( v61 == ++v60 )
                      goto LABEL_90;
                  }
                  else
                  {
                    while ( v67 != -4096 )
                    {
                      if ( v63 || v67 != -8192 )
                        v66 = v63;
                      v65 = v31 & (v64 + v65);
                      v67 = *(_QWORD *)(v62 + 8LL * v65);
                      if ( *v60 == v67 )
                        goto LABEL_80;
                      ++v64;
                      v63 = v66;
                      v66 = (_QWORD *)(v62 + 8LL * v65);
                    }
                    v90 = *(_DWORD *)(v33 + 24);
                    if ( !v63 )
                      v63 = v66;
                    ++*(_QWORD *)(v33 + 8);
                    v73 = v90 + 1;
                    if ( 4 * v73 < 3 * v68 )
                    {
                      if ( v68 - *(_DWORD *)(v33 + 28) - v73 > v68 >> 3 )
                        goto LABEL_85;
                      sub_E3B4A0(v33 + 8, v68);
                      v91 = *(_DWORD *)(v33 + 32);
                      if ( !v91 )
                      {
LABEL_167:
                        ++*(_DWORD *)(v33 + 24);
                        BUG();
                      }
                      v92 = v91 - 1;
                      v31 = *(_QWORD *)(v33 + 16);
                      v93 = 1;
                      v62 = 0;
                      LODWORD(v94) = v92 & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
                      v63 = (_QWORD *)(v31 + 8LL * (unsigned int)v94);
                      v95 = *v63;
                      v73 = *(_DWORD *)(v33 + 24) + 1;
                      if ( *v63 == *v60 )
                        goto LABEL_85;
                      while ( v95 != -4096 )
                      {
                        if ( !v62 && v95 == -8192 )
                          v62 = (__int64)v63;
                        v94 = v92 & (unsigned int)(v94 + v93);
                        v63 = (_QWORD *)(v31 + 8 * v94);
                        v95 = *v63;
                        if ( *v60 == *v63 )
                          goto LABEL_85;
                        ++v93;
                      }
                      goto LABEL_116;
                    }
LABEL_83:
                    sub_E3B4A0(v33 + 8, 2 * v68);
                    v69 = *(_DWORD *)(v33 + 32);
                    if ( !v69 )
                      goto LABEL_167;
                    v70 = v69 - 1;
                    v31 = *(_QWORD *)(v33 + 16);
                    LODWORD(v71) = v70 & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
                    v63 = (_QWORD *)(v31 + 8LL * (unsigned int)v71);
                    v72 = *v63;
                    v73 = *(_DWORD *)(v33 + 24) + 1;
                    if ( *v60 == *v63 )
                      goto LABEL_85;
                    v109 = 1;
                    v62 = 0;
                    while ( v72 != -4096 )
                    {
                      if ( !v62 && v72 == -8192 )
                        v62 = (__int64)v63;
                      v71 = v70 & (unsigned int)(v71 + v109);
                      v63 = (_QWORD *)(v31 + 8 * v71);
                      v72 = *v63;
                      if ( *v60 == *v63 )
                        goto LABEL_85;
                      ++v109;
                    }
LABEL_116:
                    if ( v62 )
                      v63 = (_QWORD *)v62;
LABEL_85:
                    *(_DWORD *)(v33 + 24) = v73;
                    if ( *v63 != -4096 )
                      --*(_DWORD *)(v33 + 28);
                    v74 = *v60;
                    *v63 = *v60;
                    v75 = *(unsigned int *)(v33 + 48);
                    if ( v75 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 52) )
                    {
                      sub_C8D5F0(v33 + 40, v111, v75 + 1, 8u, v62, v31);
                      v75 = *(unsigned int *)(v33 + 48);
                    }
                    ++v60;
                    *(_QWORD *)(*(_QWORD *)(v33 + 40) + 8 * v75) = v74;
                    ++*(_DWORD *)(v33 + 48);
                    if ( v61 == v60 )
                    {
LABEL_90:
                      v112 = &v128[(unsigned int)v129];
                      if ( v128 == v112 )
                        goto LABEL_69;
                      v114 = v128;
                      while ( 1 )
                      {
                        v76 = *(_QWORD *)(*v114 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                        if ( v76 != *v114 + 48 )
                        {
                          if ( !v76 )
                            BUG();
                          v77 = v76 - 24;
                          if ( (unsigned int)*(unsigned __int8 *)(v76 - 24) - 30 <= 0xA )
                          {
                            v78 = sub_B46E30(v77);
                            if ( v78 )
                              break;
                          }
                        }
LABEL_67:
                        if ( v112 == ++v114 )
                        {
                          v112 = v128;
                          goto LABEL_69;
                        }
                      }
                      v79 = 0;
                      while ( 2 )
                      {
                        v83 = sub_B46EC0(v77, v79);
                        v84 = *(_QWORD *)(v33 + 16);
                        v85 = v83;
                        v86 = *(_DWORD *)(v33 + 32);
                        if ( v86 )
                        {
                          v80 = v86 - 1;
                          v81 = (v86 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                          v82 = *(_QWORD *)(v84 + 8LL * v81);
                          if ( v82 != v85 )
                          {
                            v96 = 1;
                            if ( v82 == -4096 )
                              goto LABEL_100;
                            while ( 1 )
                            {
                              v81 = v80 & (v96 + v81);
                              v97 = *(_QWORD *)(v84 + 8LL * v81);
                              if ( v85 == v97 )
                                break;
                              ++v96;
                              if ( v97 == -4096 )
                                goto LABEL_100;
                            }
                          }
                        }
                        else
                        {
LABEL_100:
                          if ( !(unsigned __int8)sub_22ECC70(v33, v85) )
                          {
                            v88 = (unsigned int)v116;
                            v89 = (unsigned int)v116 + 1LL;
                            if ( v89 > HIDWORD(v116) )
                            {
                              sub_C8D5F0((__int64)&v115, v117, v89, 8u, v87, v31);
                              v88 = (unsigned int)v116;
                            }
                            v115[v88] = v85;
                            LODWORD(v116) = v116 + 1;
                          }
                        }
                        if ( v78 == ++v79 )
                          goto LABEL_67;
                        continue;
                      }
                    }
                  }
                }
                ++*(_QWORD *)(v33 + 8);
                goto LABEL_83;
              }
LABEL_69:
              if ( v112 != (__int64 *)v130 )
                _libc_free((unsigned __int64)v112);
            }
          }
          v34 = v116;
          if ( !(_DWORD)v116 )
            break;
LABEL_50:
          v32 = v115;
        }
        v11 = v33;
        v10 = v110;
        sub_C7D6A0(0, 0, 8);
        if ( v115 == v117 )
          goto LABEL_25;
        _libc_free((unsigned __int64)v115);
        v10 = v110 - 8;
        if ( v113 == v110 - 8 )
        {
LABEL_26:
          v9 = v118;
          goto LABEL_27;
        }
      }
    }
    ++*(_QWORD *)(v11 + 56);
    goto LABEL_129;
  }
LABEL_27:
  if ( v9 != v120 )
    _libc_free((unsigned __int64)v9);
}
