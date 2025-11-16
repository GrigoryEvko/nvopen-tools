// Function: sub_BBF560
// Address: 0xbbf560
//
__int64 __fastcall sub_BBF560(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9)
{
  __int64 result; // rax
  unsigned __int64 v10; // rdx
  __int64 *v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rdi
  int v17; // esi
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // r10
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  char v24; // cl
  __int64 v25; // rdi
  int v26; // esi
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // r10
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // r9
  char v33; // cl
  __int64 v34; // rdi
  int v35; // esi
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // r10
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // r9
  char v42; // cl
  __int64 v43; // rdi
  int v44; // esi
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // r10
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // r9
  char v51; // cl
  __int64 v52; // rsi
  __int64 v53; // rsi
  __int64 v54; // rsi
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rsi
  int v59; // r11d
  unsigned int i; // eax
  _QWORD *v61; // rcx
  unsigned int v62; // eax
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rsi
  int v66; // r11d
  unsigned int j; // eax
  _QWORD *v68; // rcx
  unsigned int v69; // eax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rsi
  int v73; // r11d
  unsigned int k; // eax
  _QWORD *v75; // rcx
  unsigned int v76; // eax
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rsi
  int v80; // r11d
  unsigned int m; // eax
  _QWORD *v82; // rcx
  unsigned int v83; // eax
  __int64 v84; // rsi
  __int64 v85; // rsi
  __int64 v86; // rsi
  __int64 v87; // rsi
  int v88; // eax
  int v89; // eax
  int v90; // eax
  int v91; // eax
  __int64 *v92; // r14
  __int64 v93; // r9
  __int64 *v94; // r8
  __int64 v95; // rdi
  int v96; // esi
  unsigned int v97; // edx
  __int64 v98; // rax
  __int64 v99; // r10
  __int64 v100; // rdx
  char v101; // al
  __int64 v102; // r12
  __int64 v103; // r13
  char v104; // cl
  __int64 v105; // rsi
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rsi
  int v109; // r11d
  unsigned int n; // eax
  _QWORD *v111; // rcx
  unsigned int v112; // eax
  __int64 v113; // r12
  __int64 v114; // r13
  __int64 v115; // rsi
  int v116; // eax
  int v117; // r11d
  int v118; // r11d
  int v119; // r11d
  int v120; // r11d
  int v121; // r11d
  __int64 *src; // [rsp+0h] [rbp-A0h]
  __int64 v124; // [rsp+10h] [rbp-90h]
  __int64 v125; // [rsp+10h] [rbp-90h]
  __int64 v126; // [rsp+10h] [rbp-90h]
  __int64 v127; // [rsp+10h] [rbp-90h]
  __int64 v128; // [rsp+18h] [rbp-88h]
  __int64 v129; // [rsp+18h] [rbp-88h]
  __int64 v130; // [rsp+18h] [rbp-88h]
  __int64 v131; // [rsp+18h] [rbp-88h]
  __int64 *dest; // [rsp+20h] [rbp-80h]
  __int64 *desta; // [rsp+20h] [rbp-80h]
  __int64 v134; // [rsp+28h] [rbp-78h]
  __int64 v135; // [rsp+28h] [rbp-78h]
  __int64 v136; // [rsp+30h] [rbp-70h] BYREF
  char v137[8]; // [rsp+38h] [rbp-68h] BYREF
  _BYTE v138[16]; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v139; // [rsp+50h] [rbp-50h]

  result = *a1;
  v10 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*a1 & 4) == 0 )
  {
    v11 = a1;
    if ( v10 )
    {
      src = a1 + 1;
      dest = a1 + 1;
      goto LABEL_97;
    }
    dest = a1;
    goto LABEL_4;
  }
  v11 = *(__int64 **)v10;
  v13 = 8LL * *(unsigned int *)(v10 + 8);
  dest = (__int64 *)(*(_QWORD *)v10 + v13);
  v14 = v13 >> 3;
  v15 = v13 >> 5;
  if ( !v15 )
    goto LABEL_166;
  v134 = ((unsigned int)a8 >> 9) ^ ((unsigned int)a8 >> 4);
  do
  {
    v49 = *a7;
    v50 = *v11;
    v51 = *(_BYTE *)(*a7 + 8) & 1;
    if ( v51 )
    {
      v16 = v49 + 16;
      v17 = 7;
    }
    else
    {
      v52 = *(unsigned int *)(v49 + 24);
      v16 = *(_QWORD *)(v49 + 16);
      if ( !(_DWORD)v52 )
        goto LABEL_69;
      v17 = v52 - 1;
    }
    v18 = v17 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
    v19 = (__int64 *)(v16 + 16LL * v18);
    v20 = *v19;
    if ( v50 == *v19 )
      goto LABEL_13;
    v88 = 1;
    while ( v20 != -4096 )
    {
      v117 = v88 + 1;
      v18 = v17 & (v88 + v18);
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( v50 == *v19 )
        goto LABEL_13;
      v88 = v117;
    }
    if ( v51 )
    {
      v84 = 128;
      goto LABEL_70;
    }
    v52 = *(unsigned int *)(v49 + 24);
LABEL_69:
    v84 = 16 * v52;
LABEL_70:
    v19 = (__int64 *)(v16 + v84);
LABEL_13:
    v21 = 128;
    if ( !v51 )
      v21 = 16LL * *(unsigned int *)(v49 + 24);
    if ( v19 == (__int64 *)(v16 + v21) )
    {
      v56 = a7[1];
      v57 = *(unsigned int *)(v56 + 24);
      v58 = *(_QWORD *)(v56 + 8);
      if ( (_DWORD)v57 )
      {
        v59 = 1;
        for ( i = (v57 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (v134 | ((unsigned __int64)(((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4)) << 32))) >> 31)
                 ^ (484763065 * v134)); ; i = (v57 - 1) & v62 )
        {
          v61 = (_QWORD *)(v58 + 24LL * i);
          if ( v50 == *v61 && v61[1] == a8 )
            break;
          if ( *v61 == -4096 && v61[1] == -4096 )
            goto LABEL_137;
          v62 = v59 + i;
          ++v59;
        }
      }
      else
      {
LABEL_137:
        v61 = (_QWORD *)(v58 + 24 * v57);
      }
      v124 = *a7;
      v128 = *v11;
      v137[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v61[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v61[2] + 24LL),
                  a8,
                  a9,
                  a7);
      v136 = v128;
      sub_BBCF50((__int64)v138, v124, &v136, v137);
      v19 = v139;
    }
    if ( *((_BYTE *)v19 + 8) )
      goto LABEL_101;
    v22 = *a7;
    v23 = v11[1];
    v24 = *(_BYTE *)(*a7 + 8) & 1;
    if ( v24 )
    {
      v25 = v22 + 16;
      v26 = 7;
    }
    else
    {
      v53 = *(unsigned int *)(v22 + 24);
      v25 = *(_QWORD *)(v22 + 16);
      if ( !(_DWORD)v53 )
        goto LABEL_72;
      v26 = v53 - 1;
    }
    v27 = v26 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v28 = (__int64 *)(v25 + 16LL * v27);
    v29 = *v28;
    if ( v23 == *v28 )
      goto LABEL_20;
    v89 = 1;
    while ( v29 != -4096 )
    {
      v118 = v89 + 1;
      v27 = v26 & (v89 + v27);
      v28 = (__int64 *)(v25 + 16LL * v27);
      v29 = *v28;
      if ( v23 == *v28 )
        goto LABEL_20;
      v89 = v118;
    }
    if ( v24 )
    {
      v85 = 128;
      goto LABEL_73;
    }
    v53 = *(unsigned int *)(v22 + 24);
LABEL_72:
    v85 = 16 * v53;
LABEL_73:
    v28 = (__int64 *)(v25 + v85);
LABEL_20:
    v30 = 128;
    if ( !v24 )
      v30 = 16LL * *(unsigned int *)(v22 + 24);
    if ( v28 == (__int64 *)(v25 + v30) )
    {
      v63 = a7[1];
      v64 = *(unsigned int *)(v63 + 24);
      v65 = *(_QWORD *)(v63 + 8);
      if ( (_DWORD)v64 )
      {
        v66 = 1;
        for ( j = (v64 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (v134 | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32))) >> 31)
                 ^ (484763065 * v134)); ; j = (v64 - 1) & v69 )
        {
          v68 = (_QWORD *)(v65 + 24LL * j);
          if ( v23 == *v68 && v68[1] == a8 )
            break;
          if ( *v68 == -4096 && v68[1] == -4096 )
            goto LABEL_140;
          v69 = v66 + j;
          ++v66;
        }
      }
      else
      {
LABEL_140:
        v68 = (_QWORD *)(v65 + 24 * v64);
      }
      v125 = *a7;
      v129 = v11[1];
      v137[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v68[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v68[2] + 24LL),
                  a8,
                  a9,
                  a7);
      v136 = v129;
      sub_BBCF50((__int64)v138, v125, &v136, v137);
      if ( *((_BYTE *)v139 + 8) )
      {
LABEL_142:
        ++v11;
        src = dest;
        goto LABEL_102;
      }
    }
    else if ( *((_BYTE *)v28 + 8) )
    {
      goto LABEL_142;
    }
    v31 = *a7;
    v32 = v11[2];
    v33 = *(_BYTE *)(*a7 + 8) & 1;
    if ( v33 )
    {
      v34 = v31 + 16;
      v35 = 7;
    }
    else
    {
      v54 = *(unsigned int *)(v31 + 24);
      v34 = *(_QWORD *)(v31 + 16);
      if ( !(_DWORD)v54 )
        goto LABEL_75;
      v35 = v54 - 1;
    }
    v36 = v35 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v37 = (__int64 *)(v34 + 16LL * v36);
    v38 = *v37;
    if ( v32 == *v37 )
      goto LABEL_27;
    v90 = 1;
    while ( v38 != -4096 )
    {
      v119 = v90 + 1;
      v36 = v35 & (v90 + v36);
      v37 = (__int64 *)(v34 + 16LL * v36);
      v38 = *v37;
      if ( v32 == *v37 )
        goto LABEL_27;
      v90 = v119;
    }
    if ( v33 )
    {
      v86 = 128;
      goto LABEL_76;
    }
    v54 = *(unsigned int *)(v31 + 24);
LABEL_75:
    v86 = 16 * v54;
LABEL_76:
    v37 = (__int64 *)(v34 + v86);
LABEL_27:
    v39 = 128;
    if ( !v33 )
      v39 = 16LL * *(unsigned int *)(v31 + 24);
    if ( v37 == (__int64 *)(v34 + v39) )
    {
      v70 = a7[1];
      v71 = *(unsigned int *)(v70 + 24);
      v72 = *(_QWORD *)(v70 + 8);
      if ( (_DWORD)v71 )
      {
        v73 = 1;
        for ( k = (v71 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (v134 | ((unsigned __int64)(((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4)) << 32))) >> 31)
                 ^ (484763065 * v134)); ; k = (v71 - 1) & v76 )
        {
          v75 = (_QWORD *)(v72 + 24LL * k);
          if ( v32 == *v75 && v75[1] == a8 )
            break;
          if ( *v75 == -4096 && v75[1] == -4096 )
            goto LABEL_144;
          v76 = v73 + k;
          ++v73;
        }
      }
      else
      {
LABEL_144:
        v75 = (_QWORD *)(v72 + 24 * v71);
      }
      v126 = *a7;
      v130 = v11[2];
      v137[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v75[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v75[2] + 24LL),
                  a8,
                  a9,
                  a7);
      v136 = v130;
      sub_BBCF50((__int64)v138, v126, &v136, v137);
      if ( *((_BYTE *)v139 + 8) )
      {
LABEL_146:
        v11 += 2;
        src = dest;
        goto LABEL_102;
      }
    }
    else if ( *((_BYTE *)v37 + 8) )
    {
      goto LABEL_146;
    }
    v40 = *a7;
    v41 = v11[3];
    v42 = *(_BYTE *)(*a7 + 8) & 1;
    if ( v42 )
    {
      v43 = v40 + 16;
      v44 = 7;
    }
    else
    {
      v55 = *(unsigned int *)(v40 + 24);
      v43 = *(_QWORD *)(v40 + 16);
      if ( !(_DWORD)v55 )
        goto LABEL_78;
      v44 = v55 - 1;
    }
    v45 = v44 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
    v46 = (__int64 *)(v43 + 16LL * v45);
    v47 = *v46;
    if ( v41 == *v46 )
      goto LABEL_34;
    v91 = 1;
    while ( v47 != -4096 )
    {
      v120 = v91 + 1;
      v45 = v44 & (v91 + v45);
      v46 = (__int64 *)(v43 + 16LL * v45);
      v47 = *v46;
      if ( v41 == *v46 )
        goto LABEL_34;
      v91 = v120;
    }
    if ( v42 )
    {
      v87 = 128;
      goto LABEL_79;
    }
    v55 = *(unsigned int *)(v40 + 24);
LABEL_78:
    v87 = 16 * v55;
LABEL_79:
    v46 = (__int64 *)(v43 + v87);
LABEL_34:
    v48 = 128;
    if ( !v42 )
      v48 = 16LL * *(unsigned int *)(v40 + 24);
    if ( v46 == (__int64 *)(v43 + v48) )
    {
      v77 = a7[1];
      v78 = *(unsigned int *)(v77 + 24);
      v79 = *(_QWORD *)(v77 + 8);
      if ( (_DWORD)v78 )
      {
        v80 = 1;
        for ( m = (v78 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * (v134 | ((unsigned __int64)(((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4)) << 32))) >> 31)
                 ^ (484763065 * v134)); ; m = (v78 - 1) & v83 )
        {
          v82 = (_QWORD *)(v79 + 24LL * m);
          if ( v41 == *v82 && v82[1] == a8 )
            break;
          if ( *v82 == -4096 && v82[1] == -4096 )
            goto LABEL_148;
          v83 = v80 + m;
          ++v80;
        }
      }
      else
      {
LABEL_148:
        v82 = (_QWORD *)(v79 + 24 * v78);
      }
      v127 = *a7;
      v131 = v11[3];
      v137[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v82[2] + 24LL) + 16LL))(
                  *(_QWORD *)(v82[2] + 24LL),
                  a8,
                  a9,
                  a7);
      v136 = v131;
      sub_BBCF50((__int64)v138, v127, &v136, v137);
      if ( *((_BYTE *)v139 + 8) )
      {
LABEL_150:
        v11 += 3;
        src = dest;
        goto LABEL_102;
      }
    }
    else if ( *((_BYTE *)v46 + 8) )
    {
      goto LABEL_150;
    }
    v11 += 4;
    --v15;
  }
  while ( v15 );
  v14 = dest - v11;
LABEL_166:
  if ( v14 != 2 )
  {
    if ( v14 == 3 )
    {
      if ( !(unsigned __int8)sub_BBD330(a7, *v11, a8, a9) )
      {
        ++v11;
        goto LABEL_167;
      }
      goto LABEL_101;
    }
    if ( v14 == 1 )
    {
      src = dest;
LABEL_97:
      if ( (unsigned __int8)sub_BBD330(a7, *v11, a8, a9) )
        goto LABEL_102;
      v11 = dest;
      result = *a1;
LABEL_5:
      v12 = (result >> 2) & 1;
      if ( ((result >> 2) & 1) == 0 )
        goto LABEL_6;
      goto LABEL_123;
    }
    v11 = dest;
    result = *a1;
LABEL_4:
    src = v11;
    v11 = dest;
    goto LABEL_5;
  }
LABEL_167:
  if ( !(unsigned __int8)sub_BBD330(a7, *v11, a8, a9) )
  {
    ++v11;
    src = dest;
    goto LABEL_97;
  }
LABEL_101:
  src = dest;
LABEL_102:
  if ( v11 != dest )
  {
    v92 = v11 + 1;
    if ( v11 + 1 != dest )
    {
      v93 = a8;
      v94 = dest;
      while ( 1 )
      {
        v102 = *a7;
        v103 = *v92;
        v104 = *(_BYTE *)(*a7 + 8) & 1;
        if ( v104 )
        {
          v95 = v102 + 16;
          v96 = 7;
        }
        else
        {
          v105 = *(unsigned int *)(v102 + 24);
          v95 = *(_QWORD *)(v102 + 16);
          if ( !(_DWORD)v105 )
            goto LABEL_130;
          v96 = v105 - 1;
        }
        v97 = v96 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
        v98 = v95 + 16LL * v97;
        v99 = *(_QWORD *)v98;
        if ( v103 != *(_QWORD *)v98 )
          break;
LABEL_107:
        v100 = 128;
        if ( !v104 )
          v100 = 16LL * *(unsigned int *)(v102 + 24);
        if ( v98 == v95 + v100 )
        {
          v106 = a7[1];
          v107 = *(unsigned int *)(v106 + 24);
          v108 = *(_QWORD *)(v106 + 8);
          if ( (_DWORD)v107 )
          {
            v109 = 1;
            for ( n = (v107 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)a8 >> 9) ^ ((unsigned int)a8 >> 4)
                        | ((unsigned __int64)(((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)a8 >> 9) ^ ((unsigned int)a8 >> 4)))); ; n = (v107 - 1) & v112 )
            {
              v111 = (_QWORD *)(v108 + 24LL * n);
              if ( v103 == *v111 && v111[1] == v93 )
                break;
              if ( *v111 == -4096 && v111[1] == -4096 )
                goto LABEL_160;
              v112 = v109 + n;
              ++v109;
            }
          }
          else
          {
LABEL_160:
            v111 = (_QWORD *)(v108 + 24 * v107);
          }
          desta = v94;
          v135 = v93;
          v137[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v111[2] + 24LL) + 16LL))(
                      *(_QWORD *)(v111[2] + 24LL),
                      v93,
                      a9,
                      a7);
          v136 = v103;
          sub_BBCF50((__int64)v138, v102, &v136, v137);
          v94 = desta;
          v93 = v135;
          v101 = *((_BYTE *)v139 + 8);
        }
        else
        {
          v101 = *(_BYTE *)(v98 + 8);
        }
        if ( !v101 )
          *v11++ = *v92;
        if ( ++v92 == v94 )
          goto LABEL_122;
      }
      v116 = 1;
      while ( v99 != -4096 )
      {
        v121 = v116 + 1;
        v97 = v96 & (v116 + v97);
        v98 = v95 + 16LL * v97;
        v99 = *(_QWORD *)v98;
        if ( v103 == *(_QWORD *)v98 )
          goto LABEL_107;
        v116 = v121;
      }
      if ( v104 )
      {
        v115 = 128;
      }
      else
      {
        v105 = *(unsigned int *)(v102 + 24);
LABEL_130:
        v115 = 16 * v105;
      }
      v98 = v95 + v115;
      goto LABEL_107;
    }
  }
LABEL_122:
  result = *a1;
  v12 = (*a1 >> 2) & 1;
  if ( ((*a1 >> 2) & 1) == 0 )
  {
LABEL_6:
    if ( a1 == v11 && src != v11 )
    {
      *a1 = 0;
      return (__int64)a1;
    }
    return result;
  }
LABEL_123:
  if ( result )
  {
    if ( (_BYTE)v12 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      v113 = result;
      if ( result )
      {
        result = *(_QWORD *)result;
        v114 = result + 8LL * *(unsigned int *)(v113 + 8) - (_QWORD)src;
        if ( src != (__int64 *)(result + 8LL * *(unsigned int *)(v113 + 8)) )
        {
          memmove(v11, src, result + 8LL * *(unsigned int *)(v113 + 8) - (_QWORD)src);
          result = *(_QWORD *)v113;
        }
        *(_DWORD *)(v113 + 8) = ((__int64)v11 + v114 - result) >> 3;
      }
    }
  }
  return result;
}
