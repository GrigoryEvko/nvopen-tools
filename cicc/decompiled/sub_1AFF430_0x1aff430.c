// Function: sub_1AFF430
// Address: 0x1aff430
//
__int64 __fastcall sub_1AFF430(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v13; // r13
  __int64 v15; // rax
  __int64 v16; // r12
  unsigned __int64 v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // r15
  _QWORD *v21; // r14
  __int64 v22; // rsi
  unsigned __int64 *v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 *v25; // r14
  double v26; // xmm4_8
  double v27; // xmm5_8
  double v28; // xmm4_8
  double v29; // xmm5_8
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rcx
  int v35; // esi
  unsigned int v36; // r8d
  __int64 *v37; // rdx
  __int64 v38; // rdi
  __int64 *v39; // rax
  __int64 v40; // rdi
  unsigned int v41; // r8d
  __int64 *v42; // rdx
  __int64 v43; // r10
  _QWORD *v44; // r11
  _BYTE *v45; // r9
  _BYTE *v46; // r8
  size_t v47; // r10
  __int64 v48; // r15
  __int64 *v49; // r14
  int v50; // eax
  const char **v51; // rdi
  __int64 *v52; // rax
  _QWORD *v53; // r15
  __int64 *v54; // r12
  unsigned int v55; // esi
  __int64 v56; // r13
  __int64 v57; // r8
  unsigned int v58; // edi
  _QWORD *v59; // rax
  __int64 v60; // rcx
  __int64 v61; // r13
  __int64 v62; // rax
  _BYTE *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  int v66; // r8d
  int v67; // r9d
  _BYTE *v68; // rsi
  __int64 v70; // rax
  __int64 v71; // rcx
  unsigned int v72; // edx
  __int64 v73; // rsi
  __int64 *v74; // r15
  const char ***v75; // rax
  __int64 v76; // rax
  char *v77; // rax
  char *v78; // rdx
  char *v79; // rsi
  __int64 v80; // rbx
  const char ***v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // rcx
  int v84; // r11d
  _QWORD *v85; // r10
  int v86; // ecx
  int v87; // ecx
  int v88; // r9d
  int v89; // r9d
  __int64 v90; // rdi
  unsigned int v91; // edx
  __int64 v92; // rsi
  int v93; // r11d
  _QWORD *v94; // r10
  int v95; // esi
  int v96; // esi
  __int64 v97; // rdi
  int v98; // r11d
  __int64 v99; // r9
  __int64 v100; // rdx
  __int64 v101; // rax
  int v102; // esi
  __int64 v103; // rcx
  unsigned int v104; // edi
  __int64 *v105; // rdx
  __int64 v106; // r8
  __int64 v107; // rax
  __int64 v108; // r8
  _BYTE *v109; // rax
  __int64 v110; // r8
  int v111; // eax
  int v112; // edi
  unsigned int v113; // edx
  __int64 *v114; // r14
  __int64 v115; // rax
  __int64 v116; // r15
  __int64 v117; // rdi
  int v118; // edx
  int v119; // r9d
  int v120; // edx
  int v121; // edx
  int v122; // r9d
  int v123; // r9d
  _QWORD *v124; // [rsp+0h] [rbp-E0h]
  __int64 v125; // [rsp+8h] [rbp-D8h]
  size_t v126; // [rsp+8h] [rbp-D8h]
  __int64 v127; // [rsp+10h] [rbp-D0h]
  _BYTE *v128; // [rsp+10h] [rbp-D0h]
  _QWORD *v129; // [rsp+18h] [rbp-C8h]
  _BYTE *v130; // [rsp+18h] [rbp-C8h]
  unsigned int v131; // [rsp+20h] [rbp-C0h]
  int v132; // [rsp+24h] [rbp-BCh]
  _QWORD *desta; // [rsp+28h] [rbp-B8h]
  __int64 *dest; // [rsp+28h] [rbp-B8h]
  _QWORD *v137; // [rsp+38h] [rbp-A8h]
  __int64 v138; // [rsp+48h] [rbp-98h] BYREF
  const char *v139; // [rsp+50h] [rbp-90h] BYREF
  __int64 v140; // [rsp+58h] [rbp-88h]
  const char **v141; // [rsp+60h] [rbp-80h] BYREF
  __int64 v142; // [rsp+68h] [rbp-78h]
  _WORD v143[56]; // [rsp+70h] [rbp-70h] BYREF

  v13 = a1;
  v15 = sub_157F0B0(a1);
  if ( !v15 )
    return 0;
  v16 = v15;
  v17 = sub_157EBA0(v15);
  v132 = sub_15F4D60(v17);
  if ( v132 != 1 )
    return 0;
  v20 = v16 + 40;
  sub_1AA62D0(a1, 0, a3, a4, a5, a6, v18, v19, a9, a10);
  v21 = (_QWORD *)(*(_QWORD *)(v16 + 40) & 0xFFFFFFFFFFFFFFF8LL);
  v22 = (__int64)(v21 - 3);
  desta = v21 - 3;
  sub_157EA20(v16 + 40, (__int64)(v21 - 3));
  v23 = (unsigned __int64 *)v21[1];
  v24 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
  *v23 = v24 | *v23 & 7;
  *(_QWORD *)(v24 + 8) = v23;
  *v21 &= 7uLL;
  v21[1] = 0;
  v25 = (__int64 *)(a1 + 40);
  sub_164BEC0((__int64)desta, v22, v24, (__int64)v23, a3, a4, a5, a6, v26, v27, a9, a10);
  sub_164D160(a1, v16, a3, a4, a5, a6, v28, v29, a9, a10);
  if ( a1 + 40 != (*(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL) && (__int64 *)v20 != v25 )
  {
    dest = *(__int64 **)(a1 + 48);
    sub_157EA80(v16 + 40, a1 + 40, (__int64)dest, a1 + 40);
    if ( v25 != dest )
    {
      v30 = *(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*dest & 0xFFFFFFFFFFFFFFF8LL) + 8) = v25;
      *(_QWORD *)(a1 + 40) = *(_QWORD *)(a1 + 40) & 7LL | *dest & 0xFFFFFFFFFFFFFFF8LL;
      v31 = *(_QWORD *)(v16 + 40);
      *(_QWORD *)(v30 + 8) = v20;
      v31 &= 0xFFFFFFFFFFFFFFF8LL;
      *dest = v31 | *dest & 7;
      *(_QWORD *)(v31 + 8) = dest;
      *(_QWORD *)(v16 + 40) = v30 | *(_QWORD *)(v16 + 40) & 7LL;
    }
  }
  v139 = sub_1649960(a1);
  v140 = v32;
  if ( !a12 )
    goto LABEL_37;
  v33 = *(unsigned int *)(a12 + 48);
  if ( !(_DWORD)v33 )
    goto LABEL_37;
  v34 = *(_QWORD *)(a12 + 32);
  v131 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v35 = v33 - 1;
  v36 = (v33 - 1) & v131;
  v37 = (__int64 *)(v34 + 16LL * v36);
  v38 = *v37;
  if ( v13 != *v37 )
  {
    v118 = 1;
    while ( v38 != -8 )
    {
      v119 = v118 + 1;
      v36 = v35 & (v118 + v36);
      v37 = (__int64 *)(v34 + 16LL * v36);
      v38 = *v37;
      if ( v13 == *v37 )
        goto LABEL_10;
      v118 = v119;
    }
    goto LABEL_37;
  }
LABEL_10:
  v39 = (__int64 *)(v34 + 16 * v33);
  if ( v39 != v37 )
  {
    v40 = v37[1];
    if ( v40 )
    {
      v41 = v35 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v42 = (__int64 *)(v34 + 16LL * v41);
      v43 = *v42;
      if ( v16 == *v42 )
      {
LABEL_13:
        if ( v39 != v42 )
        {
          v44 = (_QWORD *)v42[1];
          goto LABEL_15;
        }
      }
      else
      {
        v120 = 1;
        while ( v43 != -8 )
        {
          v123 = v120 + 1;
          v41 = v35 & (v120 + v41);
          v42 = (__int64 *)(v34 + 16LL * v41);
          v43 = *v42;
          if ( v16 == *v42 )
            goto LABEL_13;
          v120 = v123;
        }
      }
      v44 = 0;
LABEL_15:
      v45 = *(_BYTE **)(v40 + 32);
      v46 = *(_BYTE **)(v40 + 24);
      v141 = (const char **)v143;
      v47 = v45 - v46;
      v142 = 0x800000000LL;
      v48 = (v45 - v46) >> 3;
      if ( (unsigned __int64)(v45 - v46) > 0x40 )
      {
        v124 = v44;
        v126 = v45 - v46;
        v128 = v45;
        v130 = v46;
        sub_16CD150((__int64)&v141, v143, (v45 - v46) >> 3, 8, (int)v46, (int)v45);
        v49 = (__int64 *)v141;
        v50 = v142;
        v46 = v130;
        v45 = v128;
        v47 = v126;
        v44 = v124;
        v51 = &v141[(unsigned int)v142];
      }
      else
      {
        v49 = (__int64 *)v143;
        v50 = 0;
        v51 = (const char **)v143;
      }
      if ( v46 != v45 )
      {
        v129 = v44;
        memmove(v51, v46, v47);
        v49 = (__int64 *)v141;
        v50 = v142;
        v44 = v129;
      }
      LODWORD(v142) = v48 + v50;
      v52 = &v49[(unsigned int)(v48 + v50)];
      if ( v52 != v49 )
      {
        v127 = v16;
        v53 = v44;
        v54 = v52;
        v125 = v13;
        while ( 1 )
        {
          while ( 1 )
          {
            v61 = *v49;
            if ( a13 )
              break;
            *(_BYTE *)(a12 + 72) = 0;
            v62 = *(_QWORD *)(v61 + 8);
            if ( v53 == (_QWORD *)v62 )
              goto LABEL_24;
            v138 = v61;
            v63 = sub_1AFB990(*(_QWORD **)(v62 + 24), *(_QWORD *)(v62 + 32), &v138);
            sub_15CDF70(*(_QWORD *)(v61 + 8) + 24LL, v63);
            *(_QWORD *)(v61 + 8) = v53;
            v138 = v61;
            v68 = (_BYTE *)v53[4];
            if ( v68 == (_BYTE *)v53[5] )
            {
              sub_15CE310((__int64)(v53 + 3), v68, &v138);
            }
            else
            {
              if ( v68 )
              {
                *(_QWORD *)v68 = v61;
                v68 = (_BYTE *)v53[4];
              }
              v68 += 8;
              v53[4] = v68;
            }
            if ( *(_DWORD *)(v61 + 16) == *(_DWORD *)(*(_QWORD *)(v61 + 8) + 16LL) + 1 )
              goto LABEL_24;
            ++v49;
            sub_1AFB750(v61, (__int64)v68, v64, v65, v66, v67);
            if ( v54 == v49 )
            {
LABEL_33:
              v16 = v127;
              v13 = v125;
              goto LABEL_34;
            }
          }
          v55 = *(_DWORD *)(a13 + 24);
          v56 = *(_QWORD *)v61;
          if ( !v55 )
            break;
          v57 = *(_QWORD *)(a13 + 8);
          v58 = (v55 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
          v59 = (_QWORD *)(v57 + 16LL * v58);
          v60 = *v59;
          if ( v56 != *v59 )
          {
            v84 = 1;
            v85 = 0;
            while ( v60 != -8 )
            {
              if ( !v85 && v60 == -16 )
                v85 = v59;
              v58 = (v55 - 1) & (v84 + v58);
              v59 = (_QWORD *)(v57 + 16LL * v58);
              v60 = *v59;
              if ( v56 == *v59 )
                goto LABEL_23;
              ++v84;
            }
            v86 = *(_DWORD *)(a13 + 16);
            if ( v85 )
              v59 = v85;
            ++*(_QWORD *)a13;
            v87 = v86 + 1;
            if ( 4 * v87 < 3 * v55 )
            {
              if ( v55 - *(_DWORD *)(a13 + 20) - v87 <= v55 >> 3 )
              {
                sub_1447B20(a13, v55);
                v95 = *(_DWORD *)(a13 + 24);
                if ( !v95 )
                {
LABEL_139:
                  ++*(_DWORD *)(a13 + 16);
                  BUG();
                }
                v96 = v95 - 1;
                v97 = *(_QWORD *)(a13 + 8);
                v94 = 0;
                v98 = 1;
                v87 = *(_DWORD *)(a13 + 16) + 1;
                v99 = v96 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
                v59 = (_QWORD *)(v97 + 16 * v99);
                v100 = *v59;
                if ( v56 != *v59 )
                {
                  while ( v100 != -8 )
                  {
                    if ( v100 == -16 && !v94 )
                      v94 = v59;
                    LODWORD(v99) = v96 & (v98 + v99);
                    v59 = (_QWORD *)(v97 + 16LL * (unsigned int)v99);
                    v100 = *v59;
                    if ( v56 == *v59 )
                      goto LABEL_79;
                    ++v98;
                  }
                  goto LABEL_89;
                }
              }
              goto LABEL_79;
            }
LABEL_85:
            sub_1447B20(a13, 2 * v55);
            v88 = *(_DWORD *)(a13 + 24);
            if ( !v88 )
              goto LABEL_139;
            v89 = v88 - 1;
            v90 = *(_QWORD *)(a13 + 8);
            v91 = v89 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
            v87 = *(_DWORD *)(a13 + 16) + 1;
            v59 = (_QWORD *)(v90 + 16LL * v91);
            v92 = *v59;
            if ( v56 != *v59 )
            {
              v93 = 1;
              v94 = 0;
              while ( v92 != -8 )
              {
                if ( !v94 && v92 == -16 )
                  v94 = v59;
                v91 = v89 & (v93 + v91);
                v59 = (_QWORD *)(v90 + 16LL * v91);
                v92 = *v59;
                if ( v56 == *v59 )
                  goto LABEL_79;
                ++v93;
              }
LABEL_89:
              if ( v94 )
                v59 = v94;
            }
LABEL_79:
            *(_DWORD *)(a13 + 16) = v87;
            if ( *v59 != -8 )
              --*(_DWORD *)(a13 + 20);
            *v59 = v56;
            v59[1] = 0;
          }
LABEL_23:
          v59[1] = *v53;
LABEL_24:
          if ( v54 == ++v49 )
            goto LABEL_33;
        }
        ++*(_QWORD *)a13;
        goto LABEL_85;
      }
LABEL_34:
      if ( a13 )
        goto LABEL_35;
      v101 = *(unsigned int *)(a12 + 48);
      if ( (_DWORD)v101 )
      {
        v102 = v101 - 1;
        v103 = *(_QWORD *)(a12 + 32);
        v104 = (v101 - 1) & v131;
        v105 = (__int64 *)(v103 + 16LL * v104);
        v106 = *v105;
        if ( v13 == *v105 )
        {
LABEL_102:
          if ( v105 != (__int64 *)(v103 + 16 * v101) )
          {
            v138 = v105[1];
            v107 = v138;
            *(_BYTE *)(a12 + 72) = 0;
            v108 = *(_QWORD *)(v107 + 8);
            if ( v108 )
            {
              v109 = sub_1AFB990(*(_QWORD **)(v108 + 24), *(_QWORD *)(v108 + 32), &v138);
              sub_15CDF70(v110 + 24, v109);
              v111 = *(_DWORD *)(a12 + 48);
              v103 = *(_QWORD *)(a12 + 32);
              if ( !v111 )
              {
LABEL_35:
                if ( v141 != (const char **)v143 )
                  _libc_free((unsigned __int64)v141);
                goto LABEL_37;
              }
              v102 = v111 - 1;
            }
            v112 = 1;
            v113 = v102 & v131;
            v114 = (__int64 *)(v103 + 16LL * (v102 & v131));
            v115 = *v114;
            if ( v13 == *v114 )
            {
LABEL_107:
              v116 = v114[1];
              if ( v116 )
              {
                v117 = *(_QWORD *)(v116 + 24);
                if ( v117 )
                  j_j___libc_free_0(v117, *(_QWORD *)(v116 + 40) - v117);
                j_j___libc_free_0(v116, 56);
              }
              *v114 = -16;
              --*(_DWORD *)(a12 + 40);
              ++*(_DWORD *)(a12 + 44);
            }
            else
            {
              while ( v115 != -8 )
              {
                v113 = v102 & (v112 + v113);
                v114 = (__int64 *)(v103 + 16LL * v113);
                v115 = *v114;
                if ( v13 == *v114 )
                  goto LABEL_107;
                ++v112;
              }
            }
            goto LABEL_35;
          }
        }
        else
        {
          v121 = 1;
          while ( v106 != -8 )
          {
            v122 = v121 + 1;
            v104 = v102 & (v121 + v104);
            v105 = (__int64 *)(v103 + 16LL * v104);
            v106 = *v105;
            if ( v13 == *v105 )
              goto LABEL_102;
            v121 = v122;
          }
        }
      }
      v138 = 0;
      *(_BYTE *)(a12 + 72) = 0;
      BUG();
    }
  }
LABEL_37:
  if ( a13 )
  {
    if ( v140 && (*(_BYTE *)(v16 + 23) & 0x20) == 0 )
    {
      v143[0] = 261;
      v141 = &v139;
      sub_164B780(v16, (__int64 *)&v141);
    }
    return v16;
  }
  v70 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v70 )
  {
    v71 = *(_QWORD *)(a2 + 8);
    v72 = (v70 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v73 = *(_QWORD *)(v71 + 16LL * v72);
    v137 = (_QWORD *)(v71 + 16LL * v72);
    if ( v13 != v73 )
    {
      while ( v73 != -8 )
      {
        v72 = (v70 - 1) & (v132 + v72);
        v73 = *(_QWORD *)(v71 + 16LL * v72);
        v137 = (_QWORD *)(v71 + 16LL * v72);
        if ( v13 == v73 )
          goto LABEL_45;
        ++v132;
      }
      goto LABEL_65;
    }
LABEL_45:
    if ( v137 != (_QWORD *)(v71 + 16 * v70) )
    {
      v74 = (__int64 *)v137[1];
      if ( !v74 )
      {
LABEL_64:
        *v137 = -16;
        --*(_DWORD *)(a2 + 16);
        ++*(_DWORD *)(a2 + 20);
        goto LABEL_65;
      }
      while ( 1 )
      {
        v141 = (const char **)v13;
        v77 = (char *)sub_1AFBA50((_QWORD *)v74[4], v74[5], (__int64 *)&v141);
        v78 = (char *)v74[5];
        v79 = v77 + 8;
        if ( v78 != v77 + 8 )
        {
          memmove(v77, v79, v78 - v79);
          v79 = (char *)v74[5];
        }
        v75 = (const char ***)v74[8];
        v74[5] = (__int64)(v79 - 8);
        v80 = (__int64)v141;
        if ( (const char ***)v74[9] == v75 )
        {
          v81 = &v75[*((unsigned int *)v74 + 21)];
          if ( v75 == v81 )
          {
LABEL_63:
            v75 = v81;
          }
          else
          {
            while ( v141 != *v75 )
            {
              if ( v81 == ++v75 )
                goto LABEL_63;
            }
          }
          goto LABEL_58;
        }
        v75 = (const char ***)sub_16CC9F0((__int64)(v74 + 7), (__int64)v141);
        if ( (const char **)v80 == *v75 )
          break;
        v76 = v74[9];
        if ( v76 == v74[8] )
        {
          v75 = (const char ***)(v76 + 8LL * *((unsigned int *)v74 + 21));
          v81 = v75;
          goto LABEL_58;
        }
LABEL_50:
        v74 = (__int64 *)*v74;
        if ( !v74 )
          goto LABEL_64;
      }
      v82 = v74[9];
      if ( v82 == v74[8] )
        v83 = *((unsigned int *)v74 + 21);
      else
        v83 = *((unsigned int *)v74 + 20);
      v81 = (const char ***)(v82 + 8 * v83);
LABEL_58:
      if ( v75 != v81 )
      {
        *v75 = (const char **)-2LL;
        ++*((_DWORD *)v74 + 22);
      }
      goto LABEL_50;
    }
  }
LABEL_65:
  if ( v140 && (*(_BYTE *)(v16 + 23) & 0x20) == 0 )
  {
    v143[0] = 261;
    v141 = &v139;
    sub_164B780(v16, (__int64 *)&v141);
  }
  sub_157F980(v13);
  return v16;
}
