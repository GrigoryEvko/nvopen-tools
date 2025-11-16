// Function: sub_357B6D0
// Address: 0x357b6d0
//
__int64 __fastcall sub_357B6D0(__int64 a1)
{
  size_t v2; // rax
  __int64 v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // r12
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // eax
  _QWORD *v15; // rax
  _QWORD *v16; // rcx
  bool v17; // zf
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // r13
  __int64 v25; // rsi
  __int64 *v26; // r12
  __int64 v28; // r13
  int v29; // ecx
  _QWORD *v30; // rdi
  _QWORD *v31; // rsi
  __int64 v32; // r8
  __int64 v33; // rbx
  unsigned int v34; // r9d
  _QWORD *v35; // rdi
  _QWORD *v36; // rsi
  int v37; // edx
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 *v40; // rdi
  int v41; // eax
  unsigned int v42; // r10d
  __int64 *v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // r9
  __int64 v46; // r8
  char v47; // al
  _QWORD *v48; // rdi
  _QWORD *v49; // rsi
  int v50; // eax
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 *v53; // rdi
  int v54; // eax
  __int64 *v55; // rcx
  __int64 v56; // rsi
  char v57; // al
  __int64 v58; // r12
  char v59; // al
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  __int64 *v62; // rax
  char *v63; // r14
  __int64 v64; // rbx
  char *v65; // r13
  unsigned __int64 v66; // rax
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 **v69; // r14
  __int64 v70; // r12
  __int64 v71; // rbx
  __int64 **v72; // r13
  __int64 v73; // rax
  __int64 v74; // r12
  __int64 **v75; // r12
  __int64 v76; // rax
  __int64 v77; // r14
  __int64 i; // r13
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rcx
  __int64 v84; // rdx
  __int64 *v85; // r8
  int v86; // eax
  unsigned int v87; // esi
  __int64 *v88; // rcx
  __int64 v89; // rdi
  char v90; // al
  int v91; // ecx
  int v92; // r10d
  __int64 v93; // rcx
  int v94; // ecx
  int v95; // r11d
  __int64 v96; // rcx
  __int64 *v97; // rax
  char *v98; // rbx
  __int64 v99; // rdx
  __int64 v100; // rcx
  char *v101; // rax
  char *v102; // rsi
  __int64 v103; // rax
  unsigned __int64 v104; // rcx
  __int64 v105; // rdx
  __int64 v106; // r12
  __int64 j; // r14
  __int64 v108; // rbx
  __int64 k; // r13
  int v110; // eax
  size_t v111; // r14
  unsigned int *v112; // rax
  int v113; // r13d
  unsigned int *v114; // rbx
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  __int64 v118; // r9
  unsigned int *v119; // r12
  unsigned int v120; // esi
  char *v121; // rsi
  int v122; // ecx
  int v123; // r9d
  const void *v124; // [rsp+8h] [rbp-E8h]
  __int64 v125; // [rsp+18h] [rbp-D8h]
  __int64 v126; // [rsp+20h] [rbp-D0h]
  __int64 v127; // [rsp+28h] [rbp-C8h]
  _QWORD *v128; // [rsp+30h] [rbp-C0h]
  __int64 v129; // [rsp+38h] [rbp-B8h]
  unsigned int v130; // [rsp+4Ch] [rbp-A4h]
  __int64 *v131; // [rsp+50h] [rbp-A0h]
  __int64 v132; // [rsp+58h] [rbp-98h]
  __int64 *v133; // [rsp+60h] [rbp-90h]
  char *v134; // [rsp+60h] [rbp-90h]
  __int64 v135; // [rsp+68h] [rbp-88h]
  __int64 *v136; // [rsp+68h] [rbp-88h]
  __int64 v137; // [rsp+78h] [rbp-78h] BYREF
  void *src; // [rsp+80h] [rbp-70h] BYREF
  __int64 v139; // [rsp+88h] [rbp-68h]
  _BYTE v140[96]; // [rsp+90h] [rbp-60h] BYREF

  sub_C7D6A0(0, 0, 4);
  v2 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v2 )
  {
    v111 = v2;
    v126 = 4 * v2;
    v112 = (unsigned int *)sub_C7D670(4 * v2, 4);
    v113 = *(_DWORD *)(a1 + 256);
    v114 = v112;
    v125 = (__int64)v112;
    memcpy(v112, *(const void **)(a1 + 248), v111 * 4);
    v119 = &v114[v111];
    if ( v113 )
    {
      while ( *v114 > 0xFFFFFFFD )
      {
        if ( v119 == ++v114 )
          goto LABEL_3;
      }
      while ( v119 != v114 )
      {
        v120 = *v114++;
        sub_3578510(a1, v120, v115, v116, v117, v118);
        if ( v114 == v119 )
          break;
        while ( *v114 > 0xFFFFFFFD )
        {
          if ( v119 == ++v114 )
            goto LABEL_3;
        }
      }
    }
  }
  else
  {
    v125 = 0;
    v126 = 0;
  }
LABEL_3:
  v3 = *(_QWORD *)(a1 + 568);
  v127 = a1 + 816;
  v124 = (const void *)(a1 + 768);
  while ( v3 != *(_QWORD *)(a1 + 560) )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v3 - 8);
      *(_QWORD *)(a1 + 568) = v3 - 8;
      sub_3574BC0((_QWORD *)a1, v8);
      v9 = *(_DWORD *)(v8 + 44);
      if ( (v9 & 4) == 0 && (v9 & 8) != 0 )
        break;
      if ( (*(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL) & 0x200LL) == 0 )
        goto LABEL_7;
LABEL_11:
      v10 = *(_QWORD *)(v8 + 24);
      if ( *(_BYTE *)(a1 + 300) )
      {
        v11 = *(__int64 **)(a1 + 280);
        v5 = *(unsigned int *)(a1 + 292);
        v4 = &v11[v5];
        if ( v11 != v4 )
        {
          while ( v10 != *v11 )
          {
            if ( v4 == ++v11 )
              goto LABEL_33;
          }
LABEL_16:
          v12 = *(_QWORD *)(a1 + 584);
          if ( !v10 )
            goto LABEL_35;
          goto LABEL_17;
        }
LABEL_33:
        if ( (unsigned int)v5 < *(_DWORD *)(a1 + 288) )
        {
          *(_DWORD *)(a1 + 292) = v5 + 1;
          *v4 = v10;
          ++*(_QWORD *)(a1 + 272);
          goto LABEL_16;
        }
      }
      sub_C8CC70(a1 + 272, v10, (__int64)v4, v5, v6, v7);
      v12 = *(_QWORD *)(a1 + 584);
      if ( !v10 )
      {
LABEL_35:
        v13 = 0;
        v14 = 0;
        goto LABEL_18;
      }
LABEL_17:
      v13 = (unsigned int)(*(_DWORD *)(v10 + 24) + 1);
      v14 = *(_DWORD *)(v10 + 24) + 1;
LABEL_18:
      if ( v14 >= *(_DWORD *)(v12 + 32) || !*(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v13) )
        goto LABEL_8;
      v15 = sub_357B0F0(v127, v10);
      src = v140;
      v139 = 0x600000000LL;
      v16 = v15;
      v17 = *((_BYTE *)v15 + 28) == 0;
      v128 = v15;
      v18 = (__int64 *)v15[1];
      if ( v17 )
        v19 = *((unsigned int *)v128 + 4);
      else
        v19 = *((unsigned int *)v16 + 5);
      v133 = &v18[v19];
      if ( v18 != v133 )
      {
        while ( 1 )
        {
          v20 = *v18;
          if ( (unsigned __int64)*v18 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v133 == ++v18 )
            goto LABEL_25;
        }
        v131 = v18;
        if ( v133 != v18 )
        {
          v130 = ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4);
          while ( 1 )
          {
            v28 = sub_2E5E6D0(*(_QWORD *)(a1 + 224), v20);
            v135 = *(_QWORD *)(a1 + 584);
            if ( !v28 )
              goto LABEL_91;
            v29 = *(_DWORD *)(v28 + 72);
            v137 = v10;
            if ( v29 )
            {
              v83 = *(unsigned int *)(v28 + 80);
              v84 = *(_QWORD *)(v28 + 64);
              v85 = (__int64 *)(v84 + 8 * v83);
              if ( !(_DWORD)v83 )
                goto LABEL_41;
              v86 = v83 - 1;
              v87 = (v83 - 1) & v130;
              v88 = (__int64 *)(v84 + 8LL * v87);
              v89 = *v88;
              if ( v10 != *v88 )
              {
                v122 = 1;
                while ( v89 != -4096 )
                {
                  v123 = v122 + 1;
                  v87 = v86 & (v122 + v87);
                  v88 = (__int64 *)(v84 + 8LL * v87);
                  v89 = *v88;
                  if ( v10 == *v88 )
                    goto LABEL_103;
                  v122 = v123;
                }
LABEL_41:
                v32 = *(_QWORD *)v28;
                v33 = v28;
                if ( *(_QWORD *)v28 )
                {
                  v34 = v130;
                  while ( 1 )
                  {
                    v37 = *(_DWORD *)(v32 + 72);
                    v137 = v10;
                    if ( v37 )
                    {
                      v38 = *(unsigned int *)(v32 + 80);
                      v39 = *(_QWORD *)(v32 + 64);
                      v40 = (__int64 *)(v39 + 8 * v38);
                      if ( (_DWORD)v38 )
                      {
                        v41 = v38 - 1;
                        v42 = (v38 - 1) & v34;
                        v43 = (__int64 *)(v39 + 8LL * v42);
                        v44 = *v43;
                        if ( v10 == *v43 )
                        {
LABEL_49:
                          if ( v43 != v40 )
                            break;
                        }
                        else
                        {
                          v94 = 1;
                          while ( v44 != -4096 )
                          {
                            v95 = v94 + 1;
                            v96 = v41 & (v42 + v94);
                            v42 = v96;
                            v43 = (__int64 *)(v39 + 8 * v96);
                            v44 = *v43;
                            if ( v10 == *v43 )
                              goto LABEL_49;
                            v94 = v95;
                          }
                        }
                      }
                    }
                    else
                    {
                      v35 = *(_QWORD **)(v32 + 88);
                      v36 = &v35[*(unsigned int *)(v32 + 96)];
                      if ( v36 != sub_3574250(v35, (__int64)v36, &v137) )
                        break;
                    }
                    v33 = v32;
                    if ( !*(_QWORD *)v32 )
                      break;
                    v32 = *(_QWORD *)v32;
                  }
                }
                if ( *(_DWORD *)(v33 + 16) != 1 )
                {
                  sub_2E6D2E0(v135, v10, v20);
                  v46 = v130;
                  if ( !v47 )
                    goto LABEL_55;
                  goto LABEL_68;
                }
                goto LABEL_104;
              }
LABEL_103:
              if ( v85 == v88 )
                goto LABEL_41;
            }
            else
            {
              v30 = *(_QWORD **)(v28 + 88);
              v31 = &v30[*(unsigned int *)(v28 + 96)];
              if ( v31 == sub_3574250(v30, (__int64)v31, &v137) )
                goto LABEL_41;
            }
LABEL_104:
            sub_2E6D2E0(v135, v10, v20);
            if ( v90 )
              goto LABEL_91;
            v46 = v130;
            v33 = 0;
            while ( 1 )
            {
LABEL_55:
              v50 = *(_DWORD *)(v28 + 72);
              v137 = v10;
              if ( !v50 )
              {
                v48 = *(_QWORD **)(v28 + 88);
                v49 = &v48[*(unsigned int *)(v28 + 96)];
                if ( v49 != sub_3574250(v48, (__int64)v49, &v137) )
                  break;
                goto LABEL_54;
              }
              v51 = *(unsigned int *)(v28 + 80);
              v52 = *(_QWORD *)(v28 + 64);
              v53 = (__int64 *)(v52 + 8 * v51);
              if ( (_DWORD)v51 )
              {
                v54 = v51 - 1;
                v45 = ((_DWORD)v51 - 1) & (unsigned int)v46;
                v55 = (__int64 *)(v52 + 8 * v45);
                v56 = *v55;
                if ( v10 != *v55 )
                {
                  v91 = 1;
                  while ( v56 != -4096 )
                  {
                    v92 = v91 + 1;
                    v93 = v54 & (unsigned int)(v45 + v91);
                    v45 = (unsigned int)v93;
                    v55 = (__int64 *)(v52 + 8 * v93);
                    v56 = *v55;
                    if ( v10 == *v55 )
                      goto LABEL_58;
                    v91 = v92;
                  }
                  goto LABEL_54;
                }
LABEL_58:
                if ( v55 != v53 )
                  break;
              }
LABEL_54:
              v28 = *(_QWORD *)v28;
              if ( !v28 )
                goto LABEL_90;
            }
            if ( *(_DWORD *)(v28 + 16) == 1 || (sub_2E6D2E0(v135, **(_QWORD **)(v28 + 8), v20), v57) )
            {
LABEL_90:
              if ( v33 )
                goto LABEL_68;
LABEL_91:
              v76 = sub_2E311E0(v20);
              v77 = *(_QWORD *)(v20 + 56);
              for ( i = v76; v77 != i; v77 = *(_QWORD *)(v77 + 8) )
              {
                if ( !sub_2EE71D0(v77) )
                  sub_3577FF0(a1, v77, v79, v80, v81, v82);
                if ( !v77 )
                  BUG();
                if ( (*(_BYTE *)v77 & 4) == 0 && (*(_BYTE *)(v77 + 44) & 8) != 0 )
                {
                  do
                    v77 = *(_QWORD *)(v77 + 8);
                  while ( (*(_BYTE *)(v77 + 44) & 8) != 0 );
                }
              }
              goto LABEL_70;
            }
            if ( !*(_QWORD *)v28 )
              goto LABEL_67;
            v129 = v10;
            v58 = *(_QWORD *)v28;
            while ( 2 )
            {
              sub_2E6D2E0(v135, **(_QWORD **)(v58 + 8), v20);
              if ( v59 )
              {
                v10 = v129;
LABEL_67:
                v33 = v28;
LABEL_68:
                v60 = (unsigned int)v139;
                v61 = (unsigned int)v139 + 1LL;
                if ( v61 > HIDWORD(v139) )
                  goto LABEL_107;
                goto LABEL_69;
              }
              v28 = v58;
              if ( *(_QWORD *)v58 )
              {
                v58 = *(_QWORD *)v58;
                continue;
              }
              break;
            }
            v60 = (unsigned int)v139;
            v33 = v58;
            v10 = v129;
            v61 = (unsigned int)v139 + 1LL;
            if ( v61 > HIDWORD(v139) )
            {
LABEL_107:
              sub_C8D5F0((__int64)&src, v140, v61, 8u, v46, v45);
              v60 = (unsigned int)v139;
            }
LABEL_69:
            *((_QWORD *)src + v60) = v33;
            LODWORD(v139) = v139 + 1;
LABEL_70:
            v62 = v131 + 1;
            if ( v131 + 1 != v133 )
            {
              while ( 1 )
              {
                v20 = *v62;
                if ( (unsigned __int64)*v62 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v133 == ++v62 )
                  goto LABEL_73;
              }
              v131 = v62;
              if ( v62 != v133 )
                continue;
            }
LABEL_73:
            v63 = (char *)src;
            v64 = 8LL * (unsigned int)v139;
            v65 = (char *)src + v64;
            if ( src == (char *)src + v64 )
              break;
            _BitScanReverse64(&v66, v64 >> 3);
            sub_3578F70((char *)src, (__int64 *)((char *)src + v64), 2LL * (int)(63 - (v66 ^ 0x3F)));
            if ( (unsigned __int64)v64 > 0x80 )
            {
              v98 = v63 + 128;
              sub_3574180(v63, v63 + 128);
              if ( v63 + 128 != v65 )
              {
                do
                {
                  while ( 1 )
                  {
                    v99 = *((_QWORD *)v98 - 1);
                    v100 = *(_QWORD *)v98;
                    v101 = v98 - 8;
                    if ( *(_DWORD *)(*(_QWORD *)v98 + 168LL) > *(_DWORD *)(v99 + 168) )
                      break;
                    v121 = v98;
                    v98 += 8;
                    *(_QWORD *)v121 = v100;
                    if ( v98 == v65 )
                      goto LABEL_76;
                  }
                  do
                  {
                    *((_QWORD *)v101 + 1) = v99;
                    v102 = v101;
                    v99 = *((_QWORD *)v101 - 1);
                    v101 -= 8;
                  }
                  while ( *(_DWORD *)(v100 + 168) > *(_DWORD *)(v99 + 168) );
                  v98 += 8;
                  *(_QWORD *)v102 = v100;
                }
                while ( v98 != v65 );
              }
            }
            else
            {
              sub_3574180(v63, v65);
            }
LABEL_76:
            v134 = (char *)src + 8 * (unsigned int)v139;
            if ( src == v134 )
              break;
            v136 = (__int64 *)src;
            v132 = v10;
LABEL_78:
            v69 = *(__int64 ***)(a1 + 752);
            v70 = 8LL * *(unsigned int *)(a1 + 760);
            v71 = *v136;
            v72 = &v69[(unsigned __int64)v70 / 8];
            v73 = v70 >> 3;
            v74 = v70 >> 5;
            if ( v74 )
            {
              v75 = &v69[4 * v74];
              while ( !(unsigned __int8)sub_2E5E7F0(*v69, (__int64 *)v71) )
              {
                if ( (unsigned __int8)sub_2E5E7F0(v69[1], (__int64 *)v71) )
                {
                  ++v69;
                  break;
                }
                if ( (unsigned __int8)sub_2E5E7F0(v69[2], (__int64 *)v71) )
                {
                  v69 += 2;
                  break;
                }
                if ( (unsigned __int8)sub_2E5E7F0(v69[3], (__int64 *)v71) )
                {
                  v69 += 3;
                  break;
                }
                v69 += 4;
                if ( v69 == v75 )
                {
                  v73 = v72 - v69;
                  goto LABEL_132;
                }
              }
LABEL_85:
              if ( v72 != v69 )
              {
LABEL_86:
                if ( v134 == (char *)++v136 )
                {
                  v10 = v132;
                  break;
                }
                goto LABEL_78;
              }
LABEL_136:
              v103 = *(unsigned int *)(a1 + 760);
              v104 = *(unsigned int *)(a1 + 764);
              if ( v103 + 1 > v104 )
              {
                sub_C8D5F0(a1 + 752, v124, v103 + 1, 8u, v67, v68);
                v103 = *(unsigned int *)(a1 + 760);
              }
              v105 = *(_QWORD *)(a1 + 752);
              *(_QWORD *)(v105 + 8 * v103) = v71;
              ++*(_DWORD *)(a1 + 760);
              v106 = *(_QWORD *)(v71 + 88);
              for ( j = v106 + 8LL * *(unsigned int *)(v71 + 96); j != v106; v106 += 8 )
              {
                v108 = *(_QWORD *)(*(_QWORD *)v106 + 56LL);
                for ( k = *(_QWORD *)v106 + 48LL; k != v108; v108 = *(_QWORD *)(v108 + 8) )
                {
                  v110 = *(_DWORD *)(v108 + 44);
                  if ( (v110 & 4) != 0 || (v110 & 8) == 0 )
                  {
                    if ( (*(_QWORD *)(*(_QWORD *)(v108 + 16) + 24LL) & 0x200LL) != 0 )
                      break;
                  }
                  else if ( sub_2E88A90(v108, 512, 1) )
                  {
                    break;
                  }
                  sub_3577FF0(a1, v108, v105, v104, v67, v68);
                }
              }
              goto LABEL_86;
            }
LABEL_132:
            if ( v73 != 2 )
            {
              if ( v73 != 3 )
              {
                if ( v73 != 1 )
                  goto LABEL_136;
LABEL_135:
                if ( (unsigned __int8)sub_2E5E7F0(*v69, (__int64 *)v71) )
                  goto LABEL_85;
                goto LABEL_136;
              }
              if ( (unsigned __int8)sub_2E5E7F0(*v69, (__int64 *)v71) )
                goto LABEL_85;
              ++v69;
            }
            if ( (unsigned __int8)sub_2E5E7F0(*v69, (__int64 *)v71) )
              goto LABEL_85;
            ++v69;
            goto LABEL_135;
          }
        }
      }
LABEL_25:
      v21 = sub_2E5E6D0(*(_QWORD *)(a1 + 224), v10);
      v22 = (__int64 *)v128[9];
      if ( *((_BYTE *)v128 + 92) )
        v23 = *((unsigned int *)v128 + 21);
      else
        v23 = *((unsigned int *)v128 + 20);
      v24 = &v22[v23];
      if ( v22 != v24 )
      {
        while ( 1 )
        {
          v25 = *v22;
          v26 = v22;
          if ( (unsigned __int64)*v22 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v24 == ++v22 )
            goto LABEL_30;
        }
        if ( v24 != v22 )
        {
          do
          {
            sub_3578DD0(a1, v25, v21);
            v97 = v26 + 1;
            if ( v26 + 1 == v24 )
              break;
            v25 = *v97;
            for ( ++v26; (unsigned __int64)*v97 >= 0xFFFFFFFFFFFFFFFELL; v26 = v97 )
            {
              if ( v24 == ++v97 )
                goto LABEL_30;
              v25 = *v97;
            }
          }
          while ( v26 != v24 );
        }
      }
LABEL_30:
      if ( src == v140 )
        goto LABEL_8;
      _libc_free((unsigned __int64)src);
      v3 = *(_QWORD *)(a1 + 568);
      if ( v3 == *(_QWORD *)(a1 + 560) )
        return sub_C7D6A0(v125, v126, 4);
    }
    if ( sub_2E88A90(v8, 512, 1) )
      goto LABEL_11;
LABEL_7:
    sub_3578640(a1, v8);
LABEL_8:
    v3 = *(_QWORD *)(a1 + 568);
  }
  return sub_C7D6A0(v125, v126, 4);
}
