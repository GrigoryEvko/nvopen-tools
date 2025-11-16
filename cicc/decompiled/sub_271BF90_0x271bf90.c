// Function: sub_271BF90
// Address: 0x271bf90
//
__int64 __fastcall sub_271BF90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // r8
  int v7; // r10d
  __int64 *v8; // r9
  __int64 v9; // rcx
  __int64 *v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  int v13; // edx
  __int64 v14; // rsi
  int v15; // edi
  __int64 *v16; // rcx
  __int64 v17; // rbx
  __int64 v18; // r11
  __int64 i; // r14
  const char *v20; // rax
  __int64 *v21; // rbx
  size_t v22; // rdx
  size_t v23; // r12
  _QWORD *v24; // rax
  _BYTE *v25; // rdi
  __int64 v26; // r9
  unsigned __int64 v27; // rax
  void *v28; // rdi
  _QWORD *v29; // rax
  void *v30; // rdx
  __int64 v31; // r14
  const char *v32; // rax
  size_t v33; // rdx
  size_t v34; // r15
  bool v35; // cc
  size_t v36; // rdx
  unsigned __int8 *v37; // r13
  int v38; // eax
  __int64 v39; // rax
  _QWORD *v40; // rbx
  _QWORD *v41; // r12
  __int64 v42; // rax
  __int64 v43; // rax
  _QWORD *v45; // rax
  __m128i *v46; // rdx
  __m128i si128; // xmm0
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  char v51; // al
  __int64 v52; // r10
  __int64 v53; // r9
  __int64 v54; // r8
  unsigned int v55; // r12d
  unsigned int v56; // ecx
  __int64 *v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 *v60; // r12
  __int64 v61; // r8
  unsigned int v62; // ecx
  _QWORD *v63; // rdx
  __int64 v64; // rax
  __int64 v65; // r13
  unsigned int v66; // edx
  _QWORD *v67; // r9
  __int64 v68; // rsi
  int v69; // eax
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 v72; // rax
  _QWORD *v73; // rdi
  unsigned int v74; // r15d
  __int64 v75; // rcx
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // rsi
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  int v82; // r13d
  __int64 *v83; // r15
  int v84; // edx
  __int64 v85; // rax
  unsigned __int64 v86; // rdx
  int v87; // edi
  unsigned int v88; // r12d
  __int64 v89; // rsi
  __int64 *v90; // rax
  unsigned int v91; // eax
  __int64 v92; // rdi
  int v93; // esi
  __int64 *v94; // rcx
  __int64 *v95; // rsi
  unsigned int v96; // r12d
  __int64 v97; // rcx
  int v98; // eax
  int v99; // r15d
  __int64 *v101; // [rsp+8h] [rbp-E8h]
  __int64 v102; // [rsp+10h] [rbp-E0h]
  __int64 v103; // [rsp+10h] [rbp-E0h]
  __int64 v104; // [rsp+10h] [rbp-E0h]
  __int64 v105; // [rsp+10h] [rbp-E0h]
  __int64 v106; // [rsp+10h] [rbp-E0h]
  __int64 v107; // [rsp+10h] [rbp-E0h]
  __int64 v108; // [rsp+10h] [rbp-E0h]
  __int64 v109; // [rsp+10h] [rbp-E0h]
  __int64 *v110; // [rsp+20h] [rbp-D0h]
  __int64 v111; // [rsp+20h] [rbp-D0h]
  int v112; // [rsp+20h] [rbp-D0h]
  __int64 v113; // [rsp+20h] [rbp-D0h]
  __int64 v114; // [rsp+20h] [rbp-D0h]
  __int64 v115; // [rsp+20h] [rbp-D0h]
  __int64 v116; // [rsp+20h] [rbp-D0h]
  __int64 v117; // [rsp+20h] [rbp-D0h]
  __int64 v119; // [rsp+28h] [rbp-C8h]
  char *s1; // [rsp+30h] [rbp-C0h]
  __int64 *v122; // [rsp+38h] [rbp-B8h]
  __int64 v123; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v124; // [rsp+48h] [rbp-A8h]
  __int64 v125; // [rsp+50h] [rbp-A0h]
  __int64 v126; // [rsp+58h] [rbp-98h]
  __int64 *v127; // [rsp+60h] [rbp-90h] BYREF
  __int64 v128; // [rsp+68h] [rbp-88h]
  _QWORD v129[2]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v130; // [rsp+80h] [rbp-70h]
  __int64 v131; // [rsp+88h] [rbp-68h]
  unsigned int v132; // [rsp+90h] [rbp-60h]
  __int64 v133; // [rsp+98h] [rbp-58h]
  _QWORD *v134; // [rsp+A0h] [rbp-50h]
  __int64 v135; // [rsp+A8h] [rbp-48h]
  unsigned int v136; // [rsp+B0h] [rbp-40h]

  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = v129;
  v128 = 0;
  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a3, a2, a3, a4);
    v79 = a3;
    v4 = *(_QWORD *)(a3 + 96);
    v5 = v4 + 40LL * *(_QWORD *)(v79 + 104);
    if ( (*(_BYTE *)(v79 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v79, v79, v77, v78);
      v4 = *(_QWORD *)(v79 + 96);
    }
  }
  else
  {
    v4 = *(_QWORD *)(a3 + 96);
    v5 = v4 + 40LL * *(_QWORD *)(a3 + 104);
  }
  for ( ; v5 != v4; v4 += 40 )
  {
    if ( (*(_BYTE *)(v4 + 7) & 0x10) != 0 )
    {
      if ( !(_DWORD)v126 )
      {
        ++v123;
        goto LABEL_10;
      }
      v6 = (unsigned int)(v126 - 1);
      v7 = 1;
      v8 = 0;
      LODWORD(v9) = v6 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v10 = (__int64 *)(v124 + 8LL * (unsigned int)v9);
      v11 = *v10;
      if ( *v10 != v4 )
      {
        while ( v11 != -4096 )
        {
          if ( v11 != -8192 || v8 )
            v10 = v8;
          v9 = (unsigned int)v6 & ((_DWORD)v9 + v7);
          v11 = *(_QWORD *)(v124 + 8 * v9);
          if ( v11 == v4 )
            goto LABEL_6;
          ++v7;
          v8 = v10;
          v10 = (__int64 *)(v124 + 8 * v9);
        }
        if ( !v8 )
          v8 = v10;
        ++v123;
        v13 = v125 + 1;
        if ( 4 * ((int)v125 + 1) >= (unsigned int)(3 * v126) )
        {
LABEL_10:
          sub_CE2A30((__int64)&v123, 2 * v126);
          if ( !(_DWORD)v126 )
            goto LABEL_210;
          v6 = (unsigned int)(v126 - 1);
          v12 = v6 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v8 = (__int64 *)(v124 + 8LL * v12);
          v13 = v125 + 1;
          v14 = *v8;
          if ( *v8 != v4 )
          {
            v15 = 1;
            v16 = 0;
            while ( v14 != -4096 )
            {
              if ( v14 == -8192 && !v16 )
                v16 = v8;
              v12 = v6 & (v15 + v12);
              v8 = (__int64 *)(v124 + 8LL * v12);
              v14 = *v8;
              if ( *v8 == v4 )
                goto LABEL_123;
              ++v15;
            }
            if ( v16 )
              v8 = v16;
          }
        }
        else if ( (int)v126 - HIDWORD(v125) - v13 <= (unsigned int)v126 >> 3 )
        {
          sub_CE2A30((__int64)&v123, v126);
          if ( !(_DWORD)v126 )
          {
LABEL_210:
            LODWORD(v125) = v125 + 1;
            BUG();
          }
          v6 = v124;
          v87 = 1;
          v88 = (v126 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v8 = (__int64 *)(v124 + 8LL * v88);
          v89 = *v8;
          v13 = v125 + 1;
          v90 = 0;
          if ( *v8 != v4 )
          {
            while ( v89 != -4096 )
            {
              if ( v89 == -8192 && !v90 )
                v90 = v8;
              v88 = (v126 - 1) & (v87 + v88);
              v8 = (__int64 *)(v124 + 8LL * v88);
              v89 = *v8;
              if ( *v8 == v4 )
                goto LABEL_123;
              ++v87;
            }
            if ( v90 )
              v8 = v90;
          }
        }
LABEL_123:
        LODWORD(v125) = v13;
        if ( *v8 != -4096 )
          --HIDWORD(v125);
        *v8 = v4;
        v80 = (unsigned int)v128;
        v81 = (unsigned int)v128 + 1LL;
        if ( v81 > HIDWORD(v128) )
        {
          sub_C8D5F0((__int64)&v127, v129, v81, 8u, v6, (__int64)v8);
          v80 = (unsigned int)v128;
        }
        v127[v80] = v4;
        LODWORD(v128) = v128 + 1;
      }
    }
LABEL_6:
    ;
  }
  v17 = *(_QWORD *)(a3 + 80);
  v18 = a3 + 72;
  if ( a3 + 72 == v17 )
  {
    i = 0;
  }
  else
  {
    if ( !v17 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v17 + 32);
      if ( i != v17 + 24 )
        break;
      v17 = *(_QWORD *)(v17 + 8);
      if ( v18 == v17 )
        goto LABEL_23;
      if ( !v17 )
        BUG();
    }
  }
  if ( v18 != v17 )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      v51 = *(_BYTE *)(i - 17);
      v52 = i - 24;
      if ( (v51 & 0x10) != 0 )
        break;
LABEL_75:
      v59 = 4LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
      v60 = (__int64 *)(v52 - v59 * 8);
      if ( (v51 & 0x40) != 0 )
      {
        v60 = *(__int64 **)(i - 32);
        v52 = (__int64)&v60[v59];
      }
      if ( v60 != (__int64 *)v52 )
      {
        while ( 1 )
        {
          v65 = *v60;
          if ( (*(_BYTE *)(*v60 + 7) & 0x10) == 0 )
            goto LABEL_80;
          if ( !(_DWORD)v126 )
          {
            ++v123;
            goto LABEL_84;
          }
          v61 = (unsigned int)(v126 - 1);
          v62 = v61 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
          v63 = (_QWORD *)(v124 + 8LL * v62);
          v64 = *v63;
          if ( v65 != *v63 )
          {
            v112 = 1;
            v67 = 0;
            while ( v64 != -4096 )
            {
              if ( v64 != -8192 || v67 )
                v63 = v67;
              v62 = v61 & (v112 + v62);
              v64 = *(_QWORD *)(v124 + 8LL * v62);
              if ( v65 == v64 )
                goto LABEL_80;
              ++v112;
              v67 = v63;
              v63 = (_QWORD *)(v124 + 8LL * v62);
            }
            if ( !v67 )
              v67 = v63;
            ++v123;
            v69 = v125 + 1;
            if ( 4 * ((int)v125 + 1) < (unsigned int)(3 * v126) )
            {
              if ( (int)v126 - HIDWORD(v125) - v69 <= (unsigned int)v126 >> 3 )
              {
                v105 = v52;
                v113 = v18;
                sub_CE2A30((__int64)&v123, v126);
                if ( !(_DWORD)v126 )
                {
LABEL_207:
                  LODWORD(v125) = v125 + 1;
                  BUG();
                }
                v73 = 0;
                v18 = v113;
                v74 = (v126 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
                v52 = v105;
                v61 = 1;
                v67 = (_QWORD *)(v124 + 8LL * v74);
                v75 = *v67;
                v69 = v125 + 1;
                if ( v65 != *v67 )
                {
                  while ( v75 != -4096 )
                  {
                    if ( !v73 && v75 == -8192 )
                      v73 = v67;
                    v74 = (v126 - 1) & (v61 + v74);
                    v67 = (_QWORD *)(v124 + 8LL * v74);
                    v75 = *v67;
                    if ( v65 == *v67 )
                      goto LABEL_86;
                    v61 = (unsigned int)(v61 + 1);
                  }
                  if ( v73 )
                    v67 = v73;
                }
              }
              goto LABEL_86;
            }
LABEL_84:
            v104 = v52;
            v111 = v18;
            sub_CE2A30((__int64)&v123, 2 * v126);
            if ( !(_DWORD)v126 )
              goto LABEL_207;
            v18 = v111;
            v52 = v104;
            v66 = (v126 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
            v67 = (_QWORD *)(v124 + 8LL * v66);
            v68 = *v67;
            v69 = v125 + 1;
            if ( v65 != *v67 )
            {
              v99 = 1;
              v61 = 0;
              while ( v68 != -4096 )
              {
                if ( v68 == -8192 && !v61 )
                  v61 = (__int64)v67;
                v66 = (v126 - 1) & (v99 + v66);
                v67 = (_QWORD *)(v124 + 8LL * v66);
                v68 = *v67;
                if ( v65 == *v67 )
                  goto LABEL_86;
                ++v99;
              }
              if ( v61 )
                v67 = (_QWORD *)v61;
            }
LABEL_86:
            LODWORD(v125) = v69;
            if ( *v67 != -4096 )
              --HIDWORD(v125);
            *v67 = v65;
            v70 = (unsigned int)v128;
            v71 = (unsigned int)v128 + 1LL;
            if ( v71 > HIDWORD(v128) )
            {
              v106 = v52;
              v114 = v18;
              sub_C8D5F0((__int64)&v127, v129, v71, 8u, v61, (__int64)v67);
              v70 = (unsigned int)v128;
              v52 = v106;
              v18 = v114;
            }
            v60 += 4;
            v127[v70] = v65;
            LODWORD(v128) = v128 + 1;
            if ( (__int64 *)v52 == v60 )
              break;
          }
          else
          {
LABEL_80:
            v60 += 4;
            if ( (__int64 *)v52 == v60 )
              break;
          }
        }
      }
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v17 + 32) )
      {
        v72 = v17 - 24;
        if ( !v17 )
          v72 = 0;
        if ( i != v72 + 48 )
          break;
        v17 = *(_QWORD *)(v17 + 8);
        if ( v18 == v17 )
          goto LABEL_23;
        if ( !v17 )
          BUG();
      }
      if ( v18 == v17 )
        goto LABEL_23;
    }
    if ( (_DWORD)v126 )
    {
      v53 = (unsigned int)(v126 - 1);
      v54 = v124;
      v55 = ((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4);
      v56 = v53 & v55;
      v57 = (__int64 *)(v124 + 8LL * ((unsigned int)v53 & v55));
      v58 = *v57;
      if ( v52 == *v57 )
        goto LABEL_75;
      v82 = 1;
      v83 = 0;
      while ( v58 != -4096 )
      {
        if ( v58 != -8192 || v83 )
          v57 = v83;
        v56 = v53 & (v82 + v56);
        v58 = *(_QWORD *)(v124 + 8LL * v56);
        if ( v52 == v58 )
          goto LABEL_75;
        ++v82;
        v83 = v57;
        v57 = (__int64 *)(v124 + 8LL * v56);
      }
      if ( !v83 )
        v83 = v57;
      ++v123;
      v84 = v125 + 1;
      if ( 4 * ((int)v125 + 1) < (unsigned int)(3 * v126) )
      {
        if ( (int)v126 - HIDWORD(v125) - v84 <= (unsigned int)v126 >> 3 )
        {
          v109 = v18;
          v117 = i - 24;
          sub_CE2A30((__int64)&v123, v126);
          if ( !(_DWORD)v126 )
          {
LABEL_206:
            LODWORD(v125) = v125 + 1;
            BUG();
          }
          v54 = v124;
          v52 = i - 24;
          v95 = 0;
          v96 = (v126 - 1) & v55;
          v18 = v109;
          v83 = (__int64 *)(v124 + 8LL * v96);
          v97 = *v83;
          v84 = v125 + 1;
          v98 = 1;
          if ( v117 != *v83 )
          {
            while ( v97 != -4096 )
            {
              if ( !v95 && v97 == -8192 )
                v95 = v83;
              v53 = (unsigned int)(v98 + 1);
              v96 = (v126 - 1) & (v98 + v96);
              v83 = (__int64 *)(v124 + 8LL * v96);
              v97 = *v83;
              if ( v117 == *v83 )
                goto LABEL_134;
              ++v98;
            }
            if ( v95 )
              v83 = v95;
          }
        }
        goto LABEL_134;
      }
    }
    else
    {
      ++v123;
    }
    v108 = v18;
    v116 = i - 24;
    sub_CE2A30((__int64)&v123, 2 * v126);
    if ( !(_DWORD)v126 )
      goto LABEL_206;
    v52 = i - 24;
    v54 = (unsigned int)(v126 - 1);
    v53 = v124;
    v18 = v108;
    v91 = v54 & (((unsigned int)v116 >> 9) ^ ((unsigned int)v116 >> 4));
    v83 = (__int64 *)(v124 + 8LL * v91);
    v84 = v125 + 1;
    v92 = *v83;
    if ( v116 != *v83 )
    {
      v93 = 1;
      v94 = 0;
      while ( v92 != -4096 )
      {
        if ( !v94 && v92 == -8192 )
          v94 = v83;
        v91 = v54 & (v93 + v91);
        v83 = (__int64 *)(v124 + 8LL * v91);
        v92 = *v83;
        if ( v116 == *v83 )
          goto LABEL_134;
        ++v93;
      }
      if ( v94 )
        v83 = v94;
    }
LABEL_134:
    LODWORD(v125) = v84;
    if ( *v83 != -4096 )
      --HIDWORD(v125);
    *v83 = v52;
    v85 = (unsigned int)v128;
    v86 = (unsigned int)v128 + 1LL;
    if ( v86 > HIDWORD(v128) )
    {
      v107 = v18;
      v115 = v52;
      sub_C8D5F0((__int64)&v127, v129, v86, 8u, v54, v53);
      v85 = (unsigned int)v128;
      v18 = v107;
      v52 = v115;
    }
    v127[v85] = v52;
    LODWORD(v128) = v128 + 1;
    v51 = *(_BYTE *)(i - 17);
    goto LABEL_75;
  }
LABEL_23:
  v129[1] = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v129[0] = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v110 = v127;
  v101 = &v127[(unsigned int)v128];
  if ( v101 != v127 )
  {
    while ( 1 )
    {
      v119 = *v110;
      v20 = sub_271BF70(*v110);
      v21 = v127;
      s1 = (char *)v20;
      v23 = v22;
      v122 = &v127[(unsigned int)v128];
      if ( v122 != v127 )
        break;
LABEL_46:
      if ( v101 == ++v110 )
        goto LABEL_47;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = *v21;
        v32 = sub_271BF70(*v21);
        v34 = v33;
        v35 = v33 <= v23;
        v36 = v23;
        v37 = (unsigned __int8 *)v32;
        if ( v35 )
          v36 = v34;
        if ( !v36 )
          break;
        v38 = memcmp(s1, v32, v36);
        if ( !v38 )
          break;
        if ( v38 < 0 )
          goto LABEL_28;
        if ( v122 == ++v21 )
          goto LABEL_46;
      }
      if ( v34 != v23 && v34 > v23 )
        break;
LABEL_39:
      if ( v122 == ++v21 )
        goto LABEL_46;
    }
LABEL_28:
    v24 = sub_CB72A0();
    v25 = (_BYTE *)v24[4];
    v26 = (__int64)v24;
    v27 = v24[3] - (_QWORD)v25;
    if ( v27 < v23 )
    {
      v48 = sub_CB6200(v26, (unsigned __int8 *)s1, v23);
      v25 = *(_BYTE **)(v48 + 32);
      v26 = v48;
      if ( *(_QWORD *)(v48 + 24) - (_QWORD)v25 > 4u )
      {
LABEL_32:
        *(_DWORD *)v25 = 1684955424;
        v25[4] = 32;
        v28 = (void *)(*(_QWORD *)(v26 + 32) + 5LL);
        *(_QWORD *)(v26 + 32) = v28;
LABEL_33:
        if ( v34 > *(_QWORD *)(v26 + 24) - (_QWORD)v28 )
        {
          sub_CB6200(v26, v37, v34);
        }
        else if ( v34 )
        {
          v102 = v26;
          memcpy(v28, v37, v34);
          *(_QWORD *)(v102 + 32) += v34;
        }
        if ( (unsigned __int8)sub_31843D0(v129, v119, v31) )
        {
          v29 = sub_CB72A0();
          v30 = (void *)v29[4];
          if ( v29[3] - (_QWORD)v30 <= 0xDu )
          {
            sub_CB6200((__int64)v29, " are related.\n", 0xEu);
          }
          else
          {
            qmemcpy(v30, " are related.\n", 14);
            v29[4] += 14LL;
          }
        }
        else
        {
          v45 = sub_CB72A0();
          v46 = (__m128i *)v45[4];
          if ( v45[3] - (_QWORD)v46 <= 0x11u )
          {
            sub_CB6200((__int64)v45, " are not related.\n", 0x12u);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_42BD370);
            v46[1].m128i_i16[0] = 2606;
            *v46 = si128;
            v45[4] += 18LL;
          }
        }
        goto LABEL_39;
      }
    }
    else
    {
      if ( v23 )
      {
        v103 = v26;
        memcpy(v25, s1, v23);
        v26 = v103;
        v50 = *(_QWORD *)(v103 + 24);
        v25 = (_BYTE *)(*(_QWORD *)(v103 + 32) + v23);
        *(_QWORD *)(v103 + 32) = v25;
        v27 = v50 - (_QWORD)v25;
      }
      if ( v27 > 4 )
        goto LABEL_32;
    }
    v49 = sub_CB6200(v26, (unsigned __int8 *)" and ", 5u);
    v28 = *(void **)(v49 + 32);
    v26 = v49;
    goto LABEL_33;
  }
LABEL_47:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  v39 = v136;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  if ( (_DWORD)v39 )
  {
    v40 = v134;
    v41 = &v134[7 * v39];
    do
    {
      if ( *v40 != -8192 && *v40 != -4096 )
      {
        v42 = v40[6];
        if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
          sub_BD60C0(v40 + 4);
        v43 = v40[3];
        if ( v43 != 0 && v43 != -4096 && v43 != -8192 )
          sub_BD60C0(v40 + 1);
      }
      v40 += 7;
    }
    while ( v41 != v40 );
    v39 = v136;
  }
  sub_C7D6A0((__int64)v134, 56 * v39, 8);
  sub_C7D6A0(v130, 24LL * v132, 8);
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  sub_C7D6A0(v124, 8LL * (unsigned int)v126, 8);
  return a1;
}
