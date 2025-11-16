// Function: sub_18DA650
// Address: 0x18da650
//
__int64 __fastcall sub_18DA650(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rbx
  __int64 v4; // r12
  int v5; // r11d
  _QWORD *v6; // r10
  unsigned int v7; // eax
  _QWORD *v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rdi
  int v12; // ecx
  unsigned int v13; // eax
  int v14; // r8d
  _QWORD *v15; // rsi
  __int64 v16; // rbx
  __int64 v17; // r11
  __int64 i; // r13
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  const char *v24; // rax
  __int64 *v25; // r12
  size_t v26; // rdx
  size_t v27; // rbx
  int v28; // eax
  _QWORD *v29; // rax
  _BYTE *v30; // rdi
  __int64 v31; // r9
  unsigned __int64 v32; // rax
  void *v33; // rdi
  _QWORD *v34; // rax
  void *v35; // rdx
  __int64 v36; // r14
  const char *v37; // rax
  size_t v38; // rdx
  char *v39; // r13
  size_t v40; // r15
  _QWORD *v41; // rbx
  _QWORD *v42; // r12
  __int64 v43; // rax
  _QWORD *v45; // rax
  __m128i *v46; // rdx
  __m128i si128; // xmm0
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r15
  __int64 v52; // r12
  __int64 v53; // rdx
  int v54; // r14d
  _QWORD *v55; // r10
  unsigned int v56; // eax
  _QWORD *v57; // rcx
  __int64 v58; // rdi
  __int64 *v59; // r12
  int v60; // eax
  __int64 *v61; // r14
  int v62; // r11d
  _QWORD *v63; // r10
  unsigned int v64; // edx
  _QWORD *v65; // rdi
  __int64 v66; // rcx
  __int64 v67; // rax
  int v68; // edx
  unsigned int v69; // ecx
  __int64 *v70; // rsi
  int v71; // r11d
  _QWORD *v72; // r9
  unsigned int v73; // ecx
  __int64 v74; // rdi
  __int64 *v75; // rsi
  int v76; // eax
  __int64 *v77; // rsi
  int v78; // r8d
  unsigned int v79; // eax
  __int64 v80; // rcx
  int v81; // r8d
  _QWORD *v82; // rsi
  int v83; // edi
  _QWORD *v84; // rsi
  unsigned int v85; // ecx
  __int64 v86; // r8
  int v87; // r11d
  _QWORD *v88; // r9
  __int64 *v89; // [rsp+8h] [rbp-F8h]
  __int64 v90; // [rsp+10h] [rbp-F0h]
  __int64 v91; // [rsp+10h] [rbp-F0h]
  __int64 *v92; // [rsp+18h] [rbp-E8h]
  __int64 v93; // [rsp+20h] [rbp-E0h]
  __int64 v95; // [rsp+28h] [rbp-D8h]
  char *s1; // [rsp+30h] [rbp-D0h]
  __int64 *v97; // [rsp+38h] [rbp-C8h]
  __int64 v98; // [rsp+38h] [rbp-C8h]
  __int64 v99; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v100; // [rsp+48h] [rbp-B8h]
  __int64 v101; // [rsp+50h] [rbp-B0h]
  __int64 v102; // [rsp+58h] [rbp-A8h]
  __int64 *v103; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v104; // [rsp+68h] [rbp-98h]
  __int64 *v105; // [rsp+70h] [rbp-90h]
  _QWORD v106[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v107; // [rsp+90h] [rbp-70h]
  __int64 v108; // [rsp+98h] [rbp-68h]
  int v109; // [rsp+A0h] [rbp-60h]
  __int64 v110; // [rsp+A8h] [rbp-58h]
  _QWORD *v111; // [rsp+B0h] [rbp-50h]
  __int64 v112; // [rsp+B8h] [rbp-48h]
  unsigned int v113; // [rsp+C0h] [rbp-40h]

  v2 = a2;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, a2);
    v3 = *(_QWORD *)(a2 + 88);
    v4 = v3 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, a2);
      v3 = *(_QWORD *)(a2 + 88);
    }
  }
  else
  {
    v3 = *(_QWORD *)(a2 + 88);
    v4 = v3 + 40LL * *(_QWORD *)(a2 + 96);
  }
  if ( v4 != v3 )
  {
    while ( 1 )
    {
      v106[0] = v3;
      v10 = v3;
      if ( (*(_BYTE *)(v3 + 23) & 0x20) != 0 )
      {
        if ( !(_DWORD)v102 )
        {
          ++v99;
          goto LABEL_10;
        }
        v5 = 1;
        v6 = 0;
        v7 = (v102 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
        v8 = (_QWORD *)(v100 + 8LL * v7);
        v9 = *v8;
        if ( *v8 != v3 )
          break;
      }
LABEL_6:
      v3 += 40;
      if ( v4 == v3 )
        goto LABEL_17;
    }
    while ( v9 != -8 )
    {
      if ( v9 != -16 || v6 )
        v8 = v6;
      v7 = (v102 - 1) & (v5 + v7);
      v9 = *(_QWORD *)(v100 + 8LL * v7);
      if ( v9 == v3 )
        goto LABEL_6;
      ++v5;
      v6 = v8;
      v8 = (_QWORD *)(v100 + 8LL * v7);
    }
    if ( !v6 )
      v6 = v8;
    ++v99;
    v12 = v101 + 1;
    if ( 4 * ((int)v101 + 1) >= (unsigned int)(3 * v102) )
    {
LABEL_10:
      sub_1353F00((__int64)&v99, 2 * v102);
      if ( !(_DWORD)v102 )
        goto LABEL_207;
      v11 = v106[0];
      v12 = v101 + 1;
      v13 = (v102 - 1) & ((LODWORD(v106[0]) >> 9) ^ (LODWORD(v106[0]) >> 4));
      v6 = (_QWORD *)(v100 + 8LL * v13);
      v10 = *v6;
      if ( v106[0] == *v6 )
        goto LABEL_127;
      v14 = 1;
      v15 = 0;
      while ( v10 != -8 )
      {
        if ( !v15 && v10 == -16 )
          v15 = v6;
        v13 = (v102 - 1) & (v14 + v13);
        v6 = (_QWORD *)(v100 + 8LL * v13);
        v10 = *v6;
        if ( v106[0] == *v6 )
          goto LABEL_127;
        ++v14;
      }
    }
    else
    {
      if ( (int)v102 - HIDWORD(v101) - v12 > (unsigned int)v102 >> 3 )
        goto LABEL_127;
      sub_1353F00((__int64)&v99, v102);
      if ( !(_DWORD)v102 )
      {
LABEL_207:
        LODWORD(v101) = v101 + 1;
        BUG();
      }
      v11 = v106[0];
      v15 = 0;
      v78 = 1;
      v12 = v101 + 1;
      v79 = (v102 - 1) & ((LODWORD(v106[0]) >> 9) ^ (LODWORD(v106[0]) >> 4));
      v6 = (_QWORD *)(v100 + 8LL * v79);
      v10 = *v6;
      if ( v106[0] == *v6 )
        goto LABEL_127;
      while ( v10 != -8 )
      {
        if ( v10 == -16 && !v15 )
          v15 = v6;
        v79 = (v102 - 1) & (v78 + v79);
        v6 = (_QWORD *)(v100 + 8LL * v79);
        v10 = *v6;
        if ( v106[0] == *v6 )
          goto LABEL_127;
        ++v78;
      }
    }
    v10 = v11;
    if ( v15 )
      v6 = v15;
LABEL_127:
    LODWORD(v101) = v12;
    if ( *v6 != -8 )
      --HIDWORD(v101);
    *v6 = v10;
    v75 = v104;
    if ( v104 == v105 )
    {
      sub_1287830((__int64)&v103, v104, v106);
    }
    else
    {
      if ( v104 )
      {
        *v104 = v106[0];
        v75 = v104;
      }
      v104 = v75 + 1;
    }
    goto LABEL_6;
  }
LABEL_17:
  v16 = *(_QWORD *)(v2 + 80);
  v17 = v2 + 72;
  if ( v2 + 72 != v16 )
  {
    while ( 1 )
    {
      if ( !v16 )
LABEL_206:
        BUG();
      i = *(_QWORD *)(v16 + 24);
      if ( i != v16 + 16 )
        break;
      v16 = *(_QWORD *)(v16 + 8);
      if ( v17 == v16 )
        goto LABEL_21;
    }
    if ( v17 != v16 )
    {
      v98 = v2;
      v51 = v2 + 72;
      while ( 1 )
      {
        if ( !i )
        {
          v106[0] = 0;
          BUG();
        }
        v52 = i - 24;
        v106[0] = i - 24;
        v53 = i - 24;
        if ( (*(_BYTE *)(i - 1) & 0x20) != 0 )
          break;
LABEL_74:
        if ( (*(_BYTE *)(i - 1) & 0x40) != 0 )
        {
          v59 = *(__int64 **)(i - 32);
          v60 = *(_DWORD *)(i - 4);
        }
        else
        {
          v60 = *(_DWORD *)(i - 4);
          v59 = (__int64 *)(v52 - 24LL * (v60 & 0xFFFFFFF));
        }
        v61 = &v59[3 * (v60 & 0xFFFFFFF)];
        if ( v61 != v59 )
        {
          while ( 1 )
          {
            v67 = *v59;
            v106[0] = v67;
            if ( (*(_BYTE *)(v67 + 23) & 0x20) == 0 )
              goto LABEL_79;
            if ( !(_DWORD)v102 )
            {
              ++v99;
LABEL_83:
              sub_1353F00((__int64)&v99, 2 * v102);
              if ( !(_DWORD)v102 )
                goto LABEL_207;
              v68 = v101 + 1;
              v69 = (v102 - 1) & ((LODWORD(v106[0]) >> 9) ^ (LODWORD(v106[0]) >> 4));
              v63 = (_QWORD *)(v100 + 8LL * v69);
              v67 = *v63;
              if ( v106[0] != *v63 )
              {
                v87 = 1;
                v88 = 0;
                while ( v67 != -8 )
                {
                  if ( !v88 && v67 == -16 )
                    v88 = v63;
                  v69 = (v102 - 1) & (v87 + v69);
                  v63 = (_QWORD *)(v100 + 8LL * v69);
                  v67 = *v63;
                  if ( v106[0] == *v63 )
                    goto LABEL_85;
                  ++v87;
                }
                v67 = v106[0];
                if ( v88 )
                  v63 = v88;
              }
              goto LABEL_85;
            }
            v62 = 1;
            v63 = 0;
            v64 = (v102 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
            v65 = (_QWORD *)(v100 + 8LL * v64);
            v66 = *v65;
            if ( v67 != *v65 )
            {
              while ( v66 != -8 )
              {
                if ( v63 || v66 != -16 )
                  v65 = v63;
                v64 = (v102 - 1) & (v62 + v64);
                v66 = *(_QWORD *)(v100 + 8LL * v64);
                if ( v67 == v66 )
                  goto LABEL_79;
                ++v62;
                v63 = v65;
                v65 = (_QWORD *)(v100 + 8LL * v64);
              }
              if ( !v63 )
                v63 = v65;
              ++v99;
              v68 = v101 + 1;
              if ( 4 * ((int)v101 + 1) >= (unsigned int)(3 * v102) )
                goto LABEL_83;
              if ( (int)v102 - HIDWORD(v101) - v68 <= (unsigned int)v102 >> 3 )
              {
                sub_1353F00((__int64)&v99, v102);
                if ( !(_DWORD)v102 )
                  goto LABEL_207;
                v67 = v106[0];
                v71 = 1;
                v72 = 0;
                v73 = (v102 - 1) & ((LODWORD(v106[0]) >> 9) ^ (LODWORD(v106[0]) >> 4));
                v63 = (_QWORD *)(v100 + 8LL * v73);
                v74 = *v63;
                v68 = v101 + 1;
                if ( *v63 != v106[0] )
                {
                  while ( v74 != -8 )
                  {
                    if ( !v72 && v74 == -16 )
                      v72 = v63;
                    v73 = (v102 - 1) & (v71 + v73);
                    v63 = (_QWORD *)(v100 + 8LL * v73);
                    v74 = *v63;
                    if ( v106[0] == *v63 )
                      goto LABEL_85;
                    ++v71;
                  }
                  if ( v72 )
                    v63 = v72;
                }
              }
LABEL_85:
              LODWORD(v101) = v68;
              if ( *v63 != -8 )
                --HIDWORD(v101);
              *v63 = v67;
              v70 = v104;
              if ( v104 == v105 )
              {
                sub_1287830((__int64)&v103, v104, v106);
                goto LABEL_79;
              }
              if ( v104 )
              {
                *v104 = v106[0];
                v70 = v104;
              }
              v59 += 3;
              v104 = v70 + 1;
              if ( v61 == v59 )
                break;
            }
            else
            {
LABEL_79:
              v59 += 3;
              if ( v61 == v59 )
                break;
            }
          }
        }
        for ( i = *(_QWORD *)(i + 8); i == v16 - 24 + 40; i = *(_QWORD *)(v16 + 24) )
        {
          v16 = *(_QWORD *)(v16 + 8);
          if ( v51 == v16 )
            goto LABEL_112;
          if ( !v16 )
            goto LABEL_206;
        }
        if ( v51 == v16 )
        {
LABEL_112:
          v2 = v98;
          goto LABEL_21;
        }
      }
      if ( (_DWORD)v102 )
      {
        v54 = 1;
        v55 = 0;
        v56 = (v102 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
        v57 = (_QWORD *)(v100 + 8LL * v56);
        v58 = *v57;
        if ( v52 == *v57 )
          goto LABEL_74;
        while ( v58 != -8 )
        {
          if ( v55 || v58 != -16 )
            v57 = v55;
          v56 = (v102 - 1) & (v54 + v56);
          v58 = *(_QWORD *)(v100 + 8LL * v56);
          if ( v52 == v58 )
            goto LABEL_74;
          ++v54;
          v55 = v57;
          v57 = (_QWORD *)(v100 + 8LL * v56);
        }
        if ( !v55 )
          v55 = v57;
        ++v99;
        v76 = v101 + 1;
        if ( 4 * ((int)v101 + 1) < (unsigned int)(3 * v102) )
        {
          if ( (int)v102 - HIDWORD(v101) - v76 <= (unsigned int)v102 >> 3 )
          {
            sub_1353F00((__int64)&v99, v102);
            if ( !(_DWORD)v102 )
              goto LABEL_207;
            v53 = v106[0];
            v83 = 1;
            v84 = 0;
            v85 = (v102 - 1) & ((LODWORD(v106[0]) >> 9) ^ (LODWORD(v106[0]) >> 4));
            v55 = (_QWORD *)(v100 + 8LL * v85);
            v86 = *v55;
            v76 = v101 + 1;
            if ( *v55 != v106[0] )
            {
              while ( v86 != -8 )
              {
                if ( v86 == -16 && !v84 )
                  v84 = v55;
                v85 = (v102 - 1) & (v85 + v83);
                v55 = (_QWORD *)(v100 + 8LL * v85);
                v86 = *v55;
                if ( v106[0] == *v55 )
                  goto LABEL_143;
                ++v83;
              }
              if ( v84 )
                v55 = v84;
            }
          }
          goto LABEL_143;
        }
      }
      else
      {
        ++v99;
      }
      sub_1353F00((__int64)&v99, 2 * v102);
      if ( !(_DWORD)v102 )
        goto LABEL_207;
      LODWORD(v80) = (v102 - 1) & ((LODWORD(v106[0]) >> 9) ^ (LODWORD(v106[0]) >> 4));
      v55 = (_QWORD *)(v100 + 8LL * (unsigned int)v80);
      v53 = *v55;
      v76 = v101 + 1;
      if ( v106[0] != *v55 )
      {
        v81 = 1;
        v82 = 0;
        while ( v53 != -8 )
        {
          if ( !v82 && v53 == -16 )
            v82 = v55;
          v80 = ((_DWORD)v102 - 1) & (unsigned int)(v80 + v81);
          v55 = (_QWORD *)(v100 + 8 * v80);
          v53 = *v55;
          if ( v106[0] == *v55 )
            goto LABEL_143;
          ++v81;
        }
        v53 = v106[0];
        if ( v82 )
          v55 = v82;
      }
LABEL_143:
      LODWORD(v101) = v76;
      if ( *v55 != -8 )
        --HIDWORD(v101);
      *v55 = v53;
      v77 = v104;
      if ( v104 == v105 )
      {
        sub_1287830((__int64)&v103, v104, v106);
      }
      else
      {
        if ( v104 )
        {
          *v104 = v106[0];
          v77 = v104;
        }
        v104 = v77 + 1;
      }
      goto LABEL_74;
    }
  }
LABEL_21:
  v106[1] = 0;
  v107 = 0;
  v19 = *(__int64 **)(a1 + 8);
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v20 = *v19;
  v21 = v19[1];
  if ( v20 == v21 )
LABEL_208:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4F96DB4 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_208;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F96DB4);
  v23 = *(_QWORD *)(v2 + 40);
  v106[0] = *(_QWORD *)(v22 + 160);
  v93 = sub_1632FA0(v23);
  v92 = v103;
  v89 = v104;
  if ( v104 != v103 )
  {
    while ( 1 )
    {
      v95 = *v92;
      v24 = sub_18DA510(*v92);
      v25 = v103;
      s1 = (char *)v24;
      v27 = v26;
      v97 = v104;
      if ( v104 != v103 )
        break;
LABEL_49:
      if ( v89 == ++v92 )
        goto LABEL_50;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v36 = *v25;
        v37 = sub_18DA510(*v25);
        v39 = (char *)v37;
        v40 = v38;
        if ( v38 >= v27 )
          break;
        if ( !v38 )
          goto LABEL_43;
        v28 = memcmp(s1, v37, v38);
        if ( v28 )
          goto LABEL_47;
LABEL_31:
        if ( v40 > v27 )
          goto LABEL_32;
LABEL_43:
        if ( v97 == ++v25 )
          goto LABEL_49;
      }
      if ( !v27 || (v28 = memcmp(s1, v37, v27)) == 0 )
      {
        if ( v40 == v27 )
          goto LABEL_43;
        goto LABEL_31;
      }
LABEL_47:
      if ( v28 < 0 )
      {
LABEL_32:
        v29 = sub_16E8CB0();
        v30 = (_BYTE *)v29[3];
        v31 = (__int64)v29;
        v32 = v29[2] - (_QWORD)v30;
        if ( v32 < v27 )
        {
          v48 = sub_16E7EE0(v31, s1, v27);
          v30 = *(_BYTE **)(v48 + 24);
          v31 = v48;
          if ( *(_QWORD *)(v48 + 16) - (_QWORD)v30 > 4u )
            goto LABEL_36;
        }
        else
        {
          if ( v27 )
          {
            v91 = v31;
            memcpy(v30, s1, v27);
            v31 = v91;
            v50 = *(_QWORD *)(v91 + 16);
            v30 = (_BYTE *)(*(_QWORD *)(v91 + 24) + v27);
            *(_QWORD *)(v91 + 24) = v30;
            v32 = v50 - (_QWORD)v30;
          }
          if ( v32 > 4 )
          {
LABEL_36:
            *(_DWORD *)v30 = 1684955424;
            v30[4] = 32;
            v33 = (void *)(*(_QWORD *)(v31 + 24) + 5LL);
            *(_QWORD *)(v31 + 24) = v33;
LABEL_37:
            if ( v40 > *(_QWORD *)(v31 + 16) - (_QWORD)v33 )
            {
              sub_16E7EE0(v31, v39, v40);
            }
            else if ( v40 )
            {
              v90 = v31;
              memcpy(v33, v39, v40);
              *(_QWORD *)(v90 + 24) += v40;
            }
            if ( (unsigned __int8)sub_18DDD00(v106, v95, v36, v93) )
            {
              v34 = sub_16E8CB0();
              v35 = (void *)v34[3];
              if ( v34[2] - (_QWORD)v35 <= 0xDu )
              {
                sub_16E7EE0((__int64)v34, " are related.\n", 0xEu);
              }
              else
              {
                qmemcpy(v35, " are related.\n", 14);
                v34[3] += 14LL;
              }
            }
            else
            {
              v45 = sub_16E8CB0();
              v46 = (__m128i *)v45[3];
              if ( v45[2] - (_QWORD)v46 <= 0x11u )
              {
                sub_16E7EE0((__int64)v45, " are not related.\n", 0x12u);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_42BD370);
                v46[1].m128i_i16[0] = 2606;
                *v46 = si128;
                v45[3] += 18LL;
              }
            }
            goto LABEL_43;
          }
        }
        v49 = sub_16E7EE0(v31, " and ", 5u);
        v33 = *(void **)(v49 + 24);
        v31 = v49;
        goto LABEL_37;
      }
      if ( v97 == ++v25 )
        goto LABEL_49;
    }
  }
LABEL_50:
  if ( v113 )
  {
    v41 = v111;
    v42 = &v111[4 * v113];
    do
    {
      if ( *v41 != -8 && *v41 != -16 )
      {
        v43 = v41[3];
        if ( v43 != 0 && v43 != -8 && v43 != -16 )
          sub_1649B30(v41 + 1);
      }
      v41 += 4;
    }
    while ( v42 != v41 );
  }
  j___libc_free_0(v111);
  j___libc_free_0(v107);
  if ( v103 )
    j_j___libc_free_0(v103, (char *)v105 - (char *)v103);
  j___libc_free_0(v100);
  return 0;
}
