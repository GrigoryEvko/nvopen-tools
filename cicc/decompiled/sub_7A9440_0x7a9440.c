// Function: sub_7A9440
// Address: 0x7a9440
//
__int64 __fastcall sub_7A9440(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rbx
  int v3; // ecx
  _BOOL4 v4; // edx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // r15
  char v9; // cl
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // r12
  char mm; // al
  __int64 v14; // rbx
  unsigned __int64 v15; // r13
  __int64 j; // rax
  unsigned int v17; // ebx
  __int64 v18; // rax
  char i; // dl
  char v20; // al
  __int64 v21; // rdx
  int v22; // eax
  unsigned __int64 v23; // rax
  __int64 v24; // r13
  unsigned __int64 v25; // rbx
  int v26; // eax
  unsigned int v27; // r11d
  unsigned int v28; // ecx
  char v29; // al
  __int64 v30; // rdi
  char v31; // al
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  int v34; // r9d
  unsigned __int64 v35; // r13
  unsigned int v36; // r13d
  unsigned __int64 v37; // rdx
  int v38; // eax
  int v39; // eax
  bool v40; // r13
  unsigned int v41; // eax
  unsigned __int64 v42; // rdx
  int v43; // eax
  __int64 v44; // rax
  int v45; // eax
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // r10
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // rax
  int v50; // r12d
  unsigned int v51; // edx
  unsigned __int64 v52; // rsi
  unsigned int v53; // eax
  unsigned int v54; // r14d
  __int64 v55; // rax
  unsigned __int8 v56; // di
  unsigned int v57; // eax
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  unsigned __int64 v62; // rax
  __int64 result; // rax
  int v64; // eax
  int v65; // eax
  __int64 v66; // rdx
  __int64 *v67; // r13
  __int64 v68; // rax
  __int64 v69; // rdi
  int v70; // eax
  char v71; // al
  unsigned __int64 v72; // rax
  __int64 v73; // r13
  __int64 m; // r12
  __int64 **v75; // rax
  __int64 *v76; // r13
  __int64 *v77; // rax
  FILE *v78; // rsi
  __int64 v79; // rcx
  __int64 v80; // rdx
  int v81; // eax
  __int64 v82; // r13
  unsigned __int64 v83; // rax
  __int64 **v84; // rdx
  __int64 *k; // rax
  __int64 *v86; // r12
  __int64 **v87; // rax
  __int64 *v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rbx
  __int64 v91; // rax
  __int64 *n; // rax
  __int64 v93; // rax
  __int64 v94; // r13
  __int64 *v95; // rsi
  __int64 ii; // rbx
  unsigned __int64 v97; // rax
  __int64 *v98; // r12
  __int64 jj; // r13
  __int64 kk; // r14
  unsigned __int64 v101; // rax
  _DWORD *v102; // rsi
  _QWORD *v103; // rax
  int v104; // eax
  int v105; // eax
  unsigned __int64 i1; // r13
  __int64 v107; // r13
  __int64 v108; // rdi
  __int64 i2; // rax
  _QWORD *v110; // rax
  __int64 v111; // rbx
  __int64 v112; // rax
  unsigned __int64 v113; // rax
  unsigned __int64 v114; // r13
  unsigned __int64 v115; // rax
  int v116; // [rsp+Ch] [rbp-F4h]
  int v117; // [rsp+Ch] [rbp-F4h]
  int v118; // [rsp+Ch] [rbp-F4h]
  int v119; // [rsp+Ch] [rbp-F4h]
  int v120; // [rsp+10h] [rbp-F0h]
  unsigned int v121; // [rsp+10h] [rbp-F0h]
  unsigned int v122; // [rsp+10h] [rbp-F0h]
  unsigned int v123; // [rsp+10h] [rbp-F0h]
  int v124; // [rsp+10h] [rbp-F0h]
  unsigned int v125; // [rsp+10h] [rbp-F0h]
  unsigned int v126; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v127; // [rsp+18h] [rbp-E8h]
  __int64 v128; // [rsp+20h] [rbp-E0h]
  unsigned __int8 nn; // [rsp+20h] [rbp-E0h]
  unsigned int v130; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v131; // [rsp+28h] [rbp-D8h]
  char v132; // [rsp+30h] [rbp-D0h]
  unsigned int v133; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v134; // [rsp+30h] [rbp-D0h]
  unsigned int v135; // [rsp+30h] [rbp-D0h]
  unsigned int v136; // [rsp+30h] [rbp-D0h]
  __int64 v137; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v138; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v140; // [rsp+58h] [rbp-A8h]
  _QWORD *v141; // [rsp+58h] [rbp-A8h]
  unsigned int v142; // [rsp+58h] [rbp-A8h]
  unsigned int v143; // [rsp+58h] [rbp-A8h]
  int v144; // [rsp+58h] [rbp-A8h]
  int v145; // [rsp+58h] [rbp-A8h]
  unsigned int v146; // [rsp+68h] [rbp-98h]
  char v147; // [rsp+6Eh] [rbp-92h]
  unsigned __int8 v148; // [rsp+6Fh] [rbp-91h]
  unsigned __int64 v149; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 v150; // [rsp+78h] [rbp-88h] BYREF
  __int64 *v151; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v152; // [rsp+88h] [rbp-78h] BYREF
  unsigned __int64 v153; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v154; // [rsp+98h] [rbp-68h]
  char v155; // [rsp+9Ch] [rbp-64h]
  unsigned __int64 v156; // [rsp+A0h] [rbp-60h]
  unsigned __int64 v157; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v158; // [rsp+B0h] [rbp-50h]
  unsigned __int64 v159; // [rsp+B8h] [rbp-48h]
  unsigned __int64 v160; // [rsp+C0h] [rbp-40h]
  __int64 v161; // [rsp+C8h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 177) & 0xC0) != 0 )
  {
    v52 = *(_QWORD *)(a1 + 128);
    goto LABEL_249;
  }
  v146 = 0;
  if ( *(char *)(a1 + 142) < 0 )
  {
    v17 = *(_DWORD *)(a1 + 136);
    *(_DWORD *)(a1 + 136) = 1;
    v146 = v17;
  }
  v155 = 0;
  v152 = 0;
  v151 = (__int64 *)a1;
  v153 = 0;
  v2 = *(_QWORD *)(a1 + 160);
  v154 = unk_4F06994;
  v156 = 0;
  v3 = dword_4F077C4;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  if ( dword_4F077C4 == 2 )
  {
    if ( !v2 )
    {
LABEL_417:
      if ( (*(_BYTE *)(a1 + 176) & 0x50) != 0 )
      {
LABEL_428:
        a2 = 0;
        v4 = 0;
      }
      else
      {
        v103 = *(_QWORD **)(a1 + 168);
        while ( 1 )
        {
          v103 = (_QWORD *)*v103;
          if ( !v103 )
            break;
          if ( (*(_BYTE *)(v103[5] + 179LL) & 1) == 0 )
            goto LABEL_428;
        }
        a2 = 1;
        v4 = 1;
      }
LABEL_414:
      *(_BYTE *)(a1 + 179) = a2 | *(_BYTE *)(a1 + 179) & 0xFE;
      v5 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        goto LABEL_14;
      goto LABEL_8;
    }
    while ( 1 )
    {
      if ( (*(_BYTE *)(v2 + 146) & 4) != 0 )
      {
        if ( (unsigned int)sub_7A80B0(*(_QWORD *)(v2 + 120)) )
          goto LABEL_53;
        if ( !dword_4F077BC )
          goto LABEL_51;
      }
      else if ( !dword_4F077BC )
      {
        goto LABEL_51;
      }
      v18 = *(_QWORD *)(v2 + 120);
      for ( i = *(_BYTE *)(v18 + 140); i == 12; i = *(_BYTE *)(v18 + 140) )
        v18 = *(_QWORD *)(v18 + 160);
      if ( i != 8 || (*(_BYTE *)(v18 + 169) & 0x20) == 0 )
      {
LABEL_51:
        if ( (*(_BYTE *)(v2 + 144) & 4) == 0 || *(_BYTE *)(v2 + 137) )
        {
          a2 = 0;
          v4 = 0;
          v3 = dword_4F077C4;
          goto LABEL_414;
        }
      }
LABEL_53:
      v2 = *(_QWORD *)(v2 + 112);
      if ( !v2 )
      {
        v3 = dword_4F077C4;
        goto LABEL_417;
      }
    }
  }
  if ( dword_4F077C0 )
  {
    while ( v2 )
    {
      for ( j = *(_QWORD *)(v2 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( *(_QWORD *)(j + 128) && ((*(_BYTE *)(v2 + 144) & 4) == 0 || *(_BYTE *)(v2 + 137)) )
      {
        a2 = 0;
        v4 = 0;
        goto LABEL_7;
      }
      v2 = *(_QWORD *)(v2 + 112);
    }
    a2 = 1;
    v4 = 1;
  }
  else
  {
    LOBYTE(a2) = v2 == 0;
    v4 = v2 == 0;
  }
LABEL_7:
  *(_BYTE *)(a1 + 179) = a2 | *(_BYTE *)(a1 + 179) & 0xFE;
  v5 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    goto LABEL_15;
LABEL_8:
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    v6 = a1;
    do
      v6 = *(_QWORD *)(v6 + 160);
    while ( *(_BYTE *)(v6 + 140) == 12 );
    v5 = *(_QWORD *)v6;
  }
  v7 = *(_QWORD *)(v5 + 96);
  if ( v4 )
  {
    *(_BYTE *)(v7 + 180) |= 0x40u;
  }
  else
  {
    if ( v3 != 2 )
    {
      if ( (*(_BYTE *)(v7 + 180) & 0x40) == 0 )
      {
        v73 = *(_QWORD *)(a1 + 160);
        if ( v73 )
          goto LABEL_299;
      }
      goto LABEL_15;
    }
    v84 = **(__int64 ****)(a1 + 168);
    if ( v84 )
    {
      while ( 1 )
      {
        if ( ((_BYTE)v84[12] & 1) != 0 )
        {
          for ( k = v84[5]; *((_BYTE *)k + 140) == 12; k = (__int64 *)k[20] )
            ;
          if ( (*(_BYTE *)(*(_QWORD *)(*k + 96) + 180LL) & 0x40) != 0 )
            break;
        }
        v84 = (__int64 **)*v84;
        if ( !v84 )
          goto LABEL_359;
      }
      *(_BYTE *)(v7 + 180) |= 0x40u;
    }
LABEL_359:
    if ( (*(_BYTE *)(v7 + 180) & 0x40) != 0 )
      goto LABEL_360;
    v73 = *(_QWORD *)(a1 + 160);
    if ( !v73 )
      goto LABEL_360;
    while ( 1 )
    {
LABEL_299:
      for ( m = *(_QWORD *)(v73 + 120); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
      if ( (unsigned int)sub_8D3410(m) )
      {
        for ( m = sub_8D40F0(m); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
          ;
      }
      if ( (unsigned int)sub_8D3A70(m) )
      {
        while ( *(_BYTE *)(m + 140) == 12 )
          m = *(_QWORD *)(m + 160);
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 180LL) & 0x40) != 0 )
          break;
      }
      v73 = *(_QWORD *)(v73 + 112);
      if ( !v73 )
        goto LABEL_304;
    }
    *(_BYTE *)(v7 + 180) |= 0x40u;
LABEL_304:
    v3 = dword_4F077C4;
  }
LABEL_14:
  if ( v3 == 2 )
  {
LABEL_360:
    v86 = v151;
    v87 = (__int64 **)v151[21];
    v88 = v87[3];
    if ( v88 )
    {
      v89 = v88[5];
      *((_BYTE *)v88 + 96) |= 0x40u;
      v159 += *(_QWORD *)(*(_QWORD *)(v89 + 168) + 32LL);
      v87 = (__int64 **)v86[21];
    }
    v90 = (__int64)v87[2];
    if ( v90 )
    {
      do
      {
        v91 = sub_5EBA50(v90);
        if ( v91 && (*(_BYTE *)(v91 + 96) & 0x40) == 0 )
        {
          *(_QWORD *)(v90 + 24) = v91;
          *(_BYTE *)(v91 + 96) |= 0x40u;
        }
        v90 = *(_QWORD *)(v90 + 16);
      }
      while ( v90 );
      v87 = (__int64 **)v86[21];
    }
    for ( n = *v87; n; n = (__int64 *)*n )
      *((_BYTE *)n + 96) &= ~0x40u;
    v93 = *(_QWORD *)(a1 + 168);
    v94 = *(_QWORD *)(v93 + 24);
    if ( v94 )
    {
      v95 = *(__int64 **)(v93 + 24);
      if ( (*(_BYTE *)(v94 + 96) & 2) != 0 )
        sub_7A9250((__int64)&v151, v95);
      else
        sub_7A9360(&v151, v95);
      for ( ii = *(_QWORD *)(v94 + 40); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
        ;
      if ( (unsigned int)sub_7A80B0(ii) )
        v97 = *(_QWORD *)(v94 + 104) + *(_QWORD *)(ii + 128);
      else
        v97 = *(_QWORD *)(*(_QWORD *)(ii + 168) + 32LL) + *(_QWORD *)(v94 + 104);
      if ( v158 + 1 < v97 )
        v158 = v97 - 1;
      if ( v159 + 1 < v97 )
        v159 = v97 - 1;
      if ( (*(_BYTE *)(v94 + 96) & 1) != 0 && (*(_BYTE *)(*(_QWORD *)(v94 + 40) + 179LL) & 8) != 0 )
        sub_7A6CF0((__int64 *)&v151, v94);
      a2 = (unsigned __int64)v151;
    }
    else
    {
      a2 = (unsigned __int64)v151;
      if ( (v151[22] & 0x50) != 0 )
      {
        v111 = v151[21];
        v112 = *(_QWORD *)(v111 + 80);
        if ( v112 )
        {
          *(_QWORD *)(v111 + 72) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v112 + 40) + 168LL) + 72LL)
                                 + *(_QWORD *)(v112 + 104);
        }
        else
        {
          v114 = unk_4F069A8;
          LODWORD(v150) = unk_4F069A4;
          sub_7A65D0(&v150, (__int64)v151);
          v115 = sub_7A8C40((__int64)&v151, v114, v150, 0);
          a2 = (unsigned __int64)v151;
          *(_QWORD *)(v111 + 72) = v115;
        }
      }
    }
    v98 = *(__int64 **)(a2 + 168);
    for ( jj = *v98; jj; jj = *(_QWORD *)jj )
    {
      if ( (*(_BYTE *)(jj + 96) & 3) == 1 && v98[3] != jj )
      {
        sub_7A9360(&v151, (__int64 *)jj);
        for ( kk = *(_QWORD *)(jj + 40); *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
          ;
        if ( (unsigned int)sub_7A80B0(kk) )
          v101 = *(_QWORD *)(jj + 104) + *(_QWORD *)(kk + 128);
        else
          v101 = *(_QWORD *)(*(_QWORD *)(kk + 168) + 32LL) + *(_QWORD *)(jj + 104);
        if ( v158 + 1 < v101 )
          v158 = v101 - 1;
        a2 = v159;
        if ( v159 + 1 < v101 )
          v159 = v101 - 1;
        if ( (*(_BYTE *)(jj + 96) & 1) != 0 && (*(_BYTE *)(*(_QWORD *)(jj + 40) + 179LL) & 8) != 0 )
        {
          a2 = jj;
          sub_7A6CF0((__int64 *)&v151, jj);
        }
      }
    }
    if ( dword_4D0425C && (unsigned __int64)(unk_4D04250 - 30300LL) <= 0x63 )
    {
      a2 = 0;
      sub_7A7280((__int64)&v151, 0);
    }
  }
LABEL_15:
  v8 = v151;
  v127 = 0;
  v148 = 0;
  v138 = v152;
  v9 = *((_BYTE *)v151 + 140);
  v131 = v152;
  v10 = unk_4F06990;
  while ( 1 )
  {
    while ( 1 )
    {
      v11 = v8[20];
      if ( v11 )
      {
        while ( 1 )
        {
          if ( !(_DWORD)v10 )
          {
            a2 = v148;
            if ( (*(_BYTE *)(v11 + 88) & 3) != v148 )
              goto LABEL_19;
          }
          if ( v9 == 11 )
          {
            v153 = 0;
            v152 = v138;
          }
          v12 = *(_QWORD *)(v11 + 120);
          for ( mm = *(_BYTE *)(v12 + 140); mm == 12; mm = *(_BYTE *)(v12 + 140) )
            v12 = *(_QWORD *)(v12 + 160);
          v14 = *(_QWORD *)(*(_QWORD *)(v11 + 40) + 32LL);
          if ( (*(_BYTE *)(v11 + 146) & 4) != 0 && (unsigned __int8)(mm - 9) <= 1u )
          {
            v147 = 1;
            goto LABEL_63;
          }
          if ( mm )
            break;
          v15 = *(_QWORD *)(v11 + 128);
LABEL_29:
          *(_BYTE *)(v11 + 144) |= 2u;
          a2 = dword_4D0424C;
          v159 = *(_QWORD *)(v12 + 128) + v15;
          if ( dword_4D0424C && v11 == *(_QWORD *)(v14 + 160) )
          {
            a2 = 0;
            sub_7A8F40(v11, 0, &v151);
          }
          v9 = *((_BYTE *)v8 + 140);
          v10 = unk_4F06990;
          if ( v9 != 11 )
            goto LABEL_19;
          if ( v152 <= v131 )
          {
            if ( v152 == v131 )
            {
              a2 = v153;
              if ( v153 > v127 )
              {
                v127 = v153;
                goto LABEL_35;
              }
            }
LABEL_19:
            v11 = *(_QWORD *)(v11 + 112);
            if ( !v11 )
              goto LABEL_36;
          }
          else
          {
            v127 = v153;
LABEL_35:
            v11 = *(_QWORD *)(v11 + 112);
            v131 = v152;
            if ( !v11 )
              goto LABEL_36;
          }
        }
        v147 = 0;
LABEL_63:
        sub_8D6090(v12);
        v20 = *(_BYTE *)(v11 + 144);
        if ( (v20 & 4) == 0 )
        {
          if ( dword_4F06AAC && v156 )
            sub_7A71F0(&v151);
          a2 = (unsigned __int64)&v153;
          v133 = sub_7A7D30(v11, 0);
          v26 = sub_7A6EA0(&v152, &v153, v133);
          v27 = v133;
          v28 = v26 == 0;
          if ( v133 > v154 )
            v154 = v133;
LABEL_102:
          if ( v28 )
            goto LABEL_91;
          v15 = v152;
          v132 = v153;
          if ( (*(_BYTE *)(v11 + 144) & 4) != 0 )
          {
LABEL_71:
            v21 = *(unsigned __int8 *)(v11 + 137);
            if ( unk_4F06A84 )
              v21 = *(_QWORD *)(v11 + 176);
            v128 = *(unsigned __int8 *)(v11 + 137);
            v22 = sub_7A6DD0(&v152, &v153, v21);
            a2 = (unsigned __int64)&dword_4F06AAC;
            if ( dword_4F06AAC )
            {
              if ( v156 )
              {
                v157 -= v128;
                if ( *(_BYTE *)(v14 + 140) == 11 )
                {
                  v144 = v22;
                  sub_7A71F0(&v151);
                  v22 = v144;
                }
              }
            }
            if ( !v22 )
              goto LABEL_91;
LABEL_78:
            *(_QWORD *)(v11 + 128) = v15;
            *(_BYTE *)(v11 + 136) = v132;
            if ( v147 && v15 + *(_QWORD *)(v12 + 128) > v160 )
              v160 = v15 + *(_QWORD *)(v12 + 128);
            goto LABEL_29;
          }
          a2 = (unsigned __int64)&dword_4F077C4;
          v29 = *(_BYTE *)(v11 + 146);
          if ( dword_4F077C4 == 2 && *(_BYTE *)(v14 + 140) != 11 )
          {
            if ( (v29 & 4) == 0 )
            {
              if ( v158 < v152 )
                goto LABEL_105;
LABEL_84:
              v140 = v27;
              v23 = v15;
              v24 = v14;
              v25 = v23;
              while ( 1 )
              {
                a2 = v12;
                if ( !(unsigned int)sub_7A8210((__int64 *)&v151, v12, v25, 1, 1, 1u) )
                {
                  if ( !dword_4D0425C )
                    break;
                  a2 = v11;
                  if ( !(unsigned int)sub_7A7130((__int64)v151, v11, v25) )
                    break;
                }
                if ( v140 > unk_4F06AC0 || v152 > unk_4F06AC0 - v140 )
                {
                  v14 = v24;
                  goto LABEL_91;
                }
                v25 = v152 + v140;
                v152 += v140;
              }
              v113 = v25;
              v14 = v24;
              v15 = v113;
LABEL_266:
              v29 = *(_BYTE *)(v11 + 146);
              goto LABEL_105;
            }
            v142 = v27;
            v64 = sub_7A80B0(*(_QWORD *)(v11 + 120));
            v27 = v142;
            if ( !v64
              || (a2 = v12,
                  *(_BYTE *)(v11 + 146) |= 8u,
                  v104 = sub_7A8210((__int64 *)&v151, v12, 0, 1, 1, 1u),
                  v27 = v142,
                  v104)
              && (!dword_4D0425C || (a2 = v11, v105 = sub_7A7130((__int64)v151, v11, 0), v27 = v142, !v105)) )
            {
              if ( v15 > v158 )
              {
                v29 = *(_BYTE *)(v11 + 146);
                v143 = v27;
                if ( (v29 & 4) == 0 )
                  goto LABEL_105;
                v65 = sub_7A80B0(*(_QWORD *)(v11 + 120));
                v27 = v143;
                if ( !v65 )
                  goto LABEL_266;
              }
              goto LABEL_84;
            }
            v29 = *(_BYTE *)(v11 + 146);
            v15 = 0;
          }
LABEL_105:
          if ( (v29 & 8) == 0 )
          {
            v30 = *(_QWORD *)(v11 + 120);
            v31 = *(_BYTE *)(v30 + 140);
            if ( v31 == 12 )
            {
              v32 = sub_8D4A00(v30);
            }
            else if ( dword_4F077C0 && (v31 == 1 || v31 == 7) )
            {
              v32 = 1;
            }
            else
            {
              v32 = *(_QWORD *)(v30 + 128);
            }
            if ( v147 )
            {
              v32 = sub_730E80(v12);
              v66 = *(_QWORD *)(v12 + 168);
              if ( *(_QWORD *)(v66 + 32) >= v32 )
                v32 = *(_QWORD *)(v66 + 32);
            }
            if ( v32 > unk_4F06AC0 || v152 > unk_4F06AC0 - v32 )
            {
LABEL_91:
              if ( !v155 )
              {
                a2 = (unsigned __int64)dword_4F07508;
                sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
                v155 = 1;
              }
              *(_BYTE *)(v11 + 144) |= 2u;
              v159 = *(_QWORD *)(v11 + 128) + *(_QWORD *)(v12 + 128);
              if ( dword_4D0424C && v11 == *(_QWORD *)(v14 + 160) )
              {
                a2 = 0;
                sub_7A8F40(v11, 0, &v151);
                v9 = *((_BYTE *)v8 + 140);
                v10 = unk_4F06990;
              }
              else
              {
                v9 = *((_BYTE *)v8 + 140);
                v10 = unk_4F06990;
              }
              goto LABEL_19;
            }
            v152 += v32;
          }
          goto LABEL_78;
        }
        a2 = (unsigned __int64)dword_4D03BE8;
        if ( !dword_4D03BE8[0]
          && ((v20 & 1) != 0
           || (*(_BYTE *)(v14 + 179) & 0x20) != 0
           && (!dword_4D0425C || (unsigned __int64)(unk_4D04250 - 40100LL) > 0x12B))
          && !*(_DWORD *)(v11 + 140) )
        {
          v15 = v152;
          v132 = v153;
          goto LABEL_71;
        }
        LODWORD(v149) = 1;
        v33 = *(_QWORD *)(v11 + 120);
        v34 = *(unsigned __int8 *)(v11 + 137);
        v35 = v33;
        for ( nn = *(_BYTE *)(v11 + 137); *(_BYTE *)(v35 + 140) == 12; v35 = *(_QWORD *)(v35 + 160) )
          ;
        if ( !*(_BYTE *)(v11 + 137) )
        {
          if ( dword_4F06A98 <= 0 )
          {
            if ( dword_4F06A98 )
            {
              a2 = 0;
              v45 = sub_7A7D30(v11, 0);
              v34 = 0;
              LODWORD(v149) = v45;
            }
          }
          else
          {
            LODWORD(v149) = dword_4F06A98;
          }
          v134 = 1;
          v28 = dword_4F06AAC;
          if ( dword_4F06AAC )
          {
            if ( v156 )
              sub_7A71F0(&v151);
            else
              LODWORD(v149) = 1;
            v28 = 0;
          }
          goto LABEL_124;
        }
        v42 = *(_QWORD *)(v11 + 176);
        if ( v42 != nn && v42 )
        {
          v56 = v42 >= (unsigned __int64)dword_4F06BA0 * unk_4F06B30 ? 3 : 0;
          if ( v42 >= (unsigned __int64)dword_4F06BA0 * unk_4F06B20 )
            v56 = 5;
          if ( v42 >= (unsigned __int64)dword_4F06BA0 * unk_4F06B10 )
            v56 = 7;
          if ( v42 >= unk_4F06B00 * (unsigned __int64)dword_4F06BA0 )
            v56 = 9;
          if ( unk_4D04290
            && unk_4F06B00 < unk_4F06AF0
            && unk_4F06AF0 <= *(_QWORD *)(v33 + 128)
            && v42 >= unk_4F06AF0 * (unsigned __int64)dword_4F06BA0 )
          {
            v56 = 11;
          }
          v124 = *(unsigned __int8 *)(v11 + 137);
          v141 = sub_72BA30(v56);
          v134 = v141[16];
          v57 = sub_88CF10(v141);
          a2 = (unsigned __int64)&v153;
          LODWORD(v149) = v57;
          *(_QWORD *)(v11 + 184) = v141;
          v58 = sub_7A6EA0(&v152, &v153, v57);
          v34 = v124;
          v28 = v58 == 0;
        }
        else
        {
          if ( dword_4F06AB0 > 0 )
          {
            v134 = dword_4F06AB0;
            if ( dword_4F06AB0 != 1 )
            {
              if ( dword_4F06AB0 == unk_4F06B30 )
              {
                LODWORD(v149) = unk_4F06B28;
              }
              else if ( dword_4F06AB0 == unk_4F06B20 )
              {
                LODWORD(v149) = unk_4F06B18;
              }
              else if ( dword_4F06AB0 == unk_4F06B10 )
              {
                LODWORD(v149) = unk_4F06B08;
              }
              else if ( dword_4F06AB0 == unk_4F06B00 )
              {
                LODWORD(v149) = unk_4F06AF8;
              }
            }
            a2 = (unsigned __int64)&v149;
            v120 = v34;
            sub_7A6510(v11, (unsigned int *)&v149);
            v34 = v120;
            v28 = 0;
            if ( HIDWORD(qword_4F077B4) )
            {
LABEL_125:
              if ( (*(_BYTE *)(v11 + 144) & 4) != 0 && (!*(_BYTE *)(v11 + 137) || *(_DWORD *)(v11 + 140))
                || !unk_4F06A88 )
              {
LABEL_129:
                if ( dword_4F06AAC )
                  goto LABEL_168;
                if ( *(_DWORD *)(v11 + 140) )
                  goto LABEL_175;
                if ( !dword_4D03BE8[0] || !*(_BYTE *)(v11 + 137) )
                {
                  v36 = v149;
                  if ( v34 )
                    goto LABEL_134;
                  goto LABEL_136;
                }
                if ( unk_4D04250 > 0x765Bu
                  || (v118 = v34, v125 = v28, v81 = sub_8D3B10(v151), v28 = v125, v34 = v118, !v81) )
                {
                  v119 = v34;
                  v126 = v28;
                  v41 = sub_88CF10(*(_QWORD *)(v11 + 120));
                  v28 = v126;
                  v34 = v119;
                  if ( v41 >= (unsigned int)v149 )
                  {
                    if ( dword_4D03BE8[0] <= v41 )
                      v41 = dword_4D03BE8[0];
                    if ( v154 >= v41 )
                      goto LABEL_147;
                    goto LABEL_146;
                  }
                }
LABEL_164:
                if ( !dword_4F06AAC )
                {
                  v36 = v149;
                  if ( v34 && !*(_DWORD *)(v11 + 140) )
                  {
LABEL_134:
                    v37 = v152 % v36;
                    a2 = v134 - v37;
                    if ( nn <= (v134 - v37) * dword_4F06BA0 - v153 && v134 > v37 )
                      goto LABEL_137;
                  }
LABEL_136:
                  a2 = (unsigned __int64)&v153;
                  v38 = sub_7A6EA0(&v152, &v153, v36);
                  v36 = v149;
                  v28 = v38 == 0;
                  goto LABEL_137;
                }
LABEL_168:
                if ( v34 )
                {
                  v121 = v28;
                  if ( v156 )
                  {
                    a2 = v35;
                    if ( (unsigned int)sub_8E38C0(v156) && v157 >= nn )
                    {
                      v36 = v149;
                      v28 = v121;
LABEL_137:
                      v135 = v28;
                      LODWORD(v150) = v36;
                      v39 = sub_8D3B10(v151);
                      v28 = v135;
                      v40 = v39 != 0;
                      if ( dword_4D0425C && v39 )
                        goto LABEL_183;
                      if ( *(_BYTE *)(v11 + 137) )
                      {
                        if ( *(_DWORD *)(v11 + 140) )
                        {
                          if ( unk_4F06A88 )
                            goto LABEL_145;
                          goto LABEL_178;
                        }
                      }
                      else
                      {
                        a2 = dword_4F06A94;
                        if ( !dword_4F06A94 )
                          goto LABEL_147;
                      }
                      if ( v39 && !unk_4F06A8C )
                        goto LABEL_147;
                      if ( !unk_4F06A90 )
                      {
                        v130 = v135;
                        v137 = *(_QWORD *)v11;
                        v44 = sub_87EA80();
                        v28 = v130;
                        if ( v137 == v44 )
                          goto LABEL_147;
                        if ( dword_4D0425C && v40 )
                        {
LABEL_183:
                          if ( !*(_BYTE *)(v11 + 137) && (unsigned int)v150 < unk_4F06B18 )
                            LODWORD(v150) = unk_4F06B18;
                        }
                      }
                      if ( unk_4F06A88 )
                        goto LABEL_145;
LABEL_178:
                      a2 = (unsigned __int64)v151;
                      v136 = v28;
                      sub_7A65D0(&v150, (__int64)v151);
                      v28 = v136;
LABEL_145:
                      v41 = v150;
                      if ( v154 < (unsigned int)v150 )
LABEL_146:
                        v154 = v41;
LABEL_147:
                      v27 = 0;
                      goto LABEL_102;
                    }
                    if ( v156 )
                      sub_7A71F0(&v151);
                  }
                  a2 = (unsigned __int64)&v153;
                  v43 = sub_7A6EA0(&v152, &v153, v149);
                  v156 = v35;
                  v36 = v149;
                  v28 = v43 == 0;
                  v157 = dword_4F06BA0 * v134;
                  goto LABEL_137;
                }
LABEL_175:
                v36 = v149;
                goto LABEL_136;
              }
LABEL_185:
              a2 = (unsigned __int64)v151;
              v116 = v34;
              v122 = v28;
              sub_7A65D0(&v149, (__int64)v151);
              v28 = v122;
              v34 = v116;
              if ( !HIDWORD(qword_4F077B4) )
                goto LABEL_164;
              goto LABEL_129;
            }
LABEL_163:
            if ( !unk_4F06A88 )
              goto LABEL_164;
            goto LABEL_185;
          }
          if ( !dword_4F06AB0 )
          {
            v134 = 1;
            if ( nn <= dword_4F06BA0 - v153 )
            {
LABEL_194:
              a2 = (unsigned __int64)&v149;
              v117 = v34;
              v123 = dword_4F06AB0;
              sub_7A6510(v11, (unsigned int *)&v149);
              v28 = v123;
              v34 = v117;
              goto LABEL_124;
            }
            v134 = unk_4F06B30;
            v46 = v152 % unk_4F06B28;
            if ( nn <= dword_4F06BA0 * (unk_4F06B30 - v46) - v153 && unk_4F06B30 > v46 )
            {
              LODWORD(v149) = unk_4F06B28;
              goto LABEL_194;
            }
            v47 = v152 % unk_4F06B18;
            if ( nn > dword_4F06BA0 * (unk_4F06B20 - v47) - v153 || unk_4F06B20 <= v47 )
            {
              v48 = v152 % unk_4F06B08;
              if ( nn <= dword_4F06BA0 * (unk_4F06B10 - v48) - v153 && unk_4F06B10 > v48 )
              {
LABEL_442:
                v134 = unk_4F06B10;
                LODWORD(v149) = unk_4F06B08;
                goto LABEL_194;
              }
              v49 = (nn + (unsigned __int64)(dword_4F06BA0 - 1)) / dword_4F06BA0;
              if ( v49 <= 1 )
              {
                v134 = 1;
                goto LABEL_194;
              }
              if ( unk_4F06B30 >= v49 )
              {
                LODWORD(v149) = unk_4F06B28;
                goto LABEL_194;
              }
              if ( unk_4F06B20 < v49 )
              {
                if ( unk_4F06B10 < v49 )
                {
                  v134 = unk_4F06B00;
                  if ( v49 > unk_4F06B00 )
                    v134 = (nn + (unsigned __int64)(dword_4F06BA0 - 1)) / dword_4F06BA0;
                  else
                    LODWORD(v149) = unk_4F06AF8;
                  goto LABEL_194;
                }
                goto LABEL_442;
              }
            }
            v134 = unk_4F06B20;
            LODWORD(v149) = unk_4F06B18;
            goto LABEL_194;
          }
          a2 = 0;
          v145 = *(unsigned __int8 *)(v11 + 137);
          v134 = *(_QWORD *)(v35 + 128);
          LODWORD(v72) = sub_7A7D30(v11, 0);
          v34 = v145;
          LODWORD(v149) = v72;
          v28 = dword_4D0425C;
          if ( dword_4D0425C )
          {
            a2 = v134;
            v72 = (unsigned int)v72;
            if ( v134 >= (unsigned int)v72 )
              v72 = v134;
            v28 = 0;
            v134 = v72;
          }
        }
LABEL_124:
        if ( HIDWORD(qword_4F077B4) )
          goto LABEL_125;
        goto LABEL_163;
      }
LABEL_36:
      if ( (_DWORD)v10 )
        goto LABEL_202;
      if ( v148 )
        break;
      v148 = 1;
    }
    if ( v148 != 1 )
      break;
    v148 = 2;
  }
LABEL_202:
  if ( v9 == 11 )
  {
    v152 = v131;
    v153 = v127;
  }
  else if ( dword_4F06AAC && v156 )
  {
    a2 = (unsigned __int64)&v153;
    sub_7A6DD0(&v152, &v153, v157);
    v156 = 0;
    v157 = 0;
  }
  if ( dword_4F077C4 != 2 )
  {
    if ( *(char *)(a1 + 142) >= 0 )
      goto LABEL_208;
    goto LABEL_274;
  }
  if ( dword_4D0425C )
    sub_7A80F0(&v151);
  v149 = 0;
  if ( (v151[22] & 0x10) != 0 )
  {
    v82 = v151[21];
    if ( v153 )
      sub_7A6D50((__int64)&v151);
    a2 = dword_4D0425C;
    v10 = v154;
    if ( v158 < v152 )
    {
      *(_QWORD *)(v82 + 32) = v152;
    }
    else
    {
      if ( dword_4D0425C && unk_4F068A8 )
      {
        *(_QWORD *)(v82 + 32) = v152;
        v83 = v158;
        *(_DWORD *)(v82 + 40) = v10;
        v152 = v83 + 1;
        goto LABEL_347;
      }
      *(_QWORD *)(v82 + 32) = v158 + 1;
    }
    *(_DWORD *)(v82 + 40) = v10;
    if ( !(_DWORD)a2 )
    {
LABEL_445:
      if ( unk_4F068A8 )
      {
        if ( dword_4D0424C )
        {
          a2 = (unsigned __int64)&v149;
          v150 = *(_QWORD *)(v82 + 32);
          if ( (unsigned int)sub_7A6EA0(&v150, &v149, *(_DWORD *)(v82 + 40)) )
          {
            if ( *(_QWORD *)(v82 + 32) != v150 )
            {
              a2 = (unsigned __int64)(v151 + 8);
              sub_684B30(0x4C9u, (_DWORD *)v151 + 16);
            }
          }
        }
      }
      else
      {
        a2 = (unsigned __int64)&v149;
        if ( (unsigned int)sub_7A6EA0((unsigned __int64 *)(v82 + 32), &v149, *(_DWORD *)(v82 + 40)) )
        {
          v152 = *(_QWORD *)(v82 + 32);
        }
        else if ( !v155 )
        {
          a2 = (unsigned __int64)dword_4F07508;
          sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
          v155 = 1;
        }
      }
      if ( *(_QWORD *)v82 )
      {
        for ( i1 = *(_QWORD *)(v82 + 16); i1; i1 = *(_QWORD *)(i1 + 16) )
        {
          if ( (*(_BYTE *)(i1 + 96) & 0x4A) == 2 )
          {
            a2 = i1;
            sub_7A9250((__int64)&v151, (__int64 *)i1);
          }
        }
      }
      if ( dword_4D0425C && (unsigned __int64)(unk_4D04250 - 30300LL) <= 0x63 )
      {
        a2 = 1;
        sub_7A7280((__int64)&v151, 1);
      }
      goto LABEL_273;
    }
LABEL_347:
    if ( unk_4D04250 <= 0x76BFu )
    {
      a2 = (unsigned __int64)&v153;
      if ( !(unsigned int)sub_7A6EA0(&v152, &v153, v10) && !v155 )
      {
        a2 = (unsigned __int64)dword_4F07508;
        sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
        v155 = 1;
      }
    }
    goto LABEL_445;
  }
LABEL_273:
  if ( *(char *)(a1 + 142) < 0 )
  {
LABEL_274:
    if ( v146 >= v154 )
      goto LABEL_287;
    v67 = *(__int64 **)(a1 + 104);
    if ( v67 )
    {
      while ( 1 )
      {
        if ( *((_BYTE *)v67 + 8) == 3 )
        {
          v68 = v67[4];
          v69 = *(_QWORD *)(v68 + 40);
          if ( *(_BYTE *)(v68 + 10) == 3 )
          {
            a2 = (unsigned __int64)&v150;
            if ( sub_620FA0(v69, &v150) == v146 )
              goto LABEL_284;
          }
          else
          {
            if ( *(char *)(v69 + 142) >= 0 && *(_BYTE *)(v69 + 140) == 12 )
              v70 = sub_8D4AB0(v69, a2, v10);
            else
              v70 = *(_DWORD *)(v69 + 136);
            if ( v146 == v70 )
            {
LABEL_284:
              v71 = *((_BYTE *)v67 + 9);
              if ( v71 == 1 || v71 == 4 )
              {
                if ( !HIDWORD(qword_4F077B4) || (_DWORD)qword_4F077B4 )
                {
                  sub_6851C0(0x759u, (_DWORD *)v67 + 14);
                  v146 = v154;
                  goto LABEL_287;
                }
                if ( (*(_BYTE *)(a1 + 179) & 0x20) != 0 )
                  goto LABEL_287;
              }
              else
              {
LABEL_286:
                if ( !HIDWORD(qword_4F077B4) || (*(_BYTE *)(a1 + 179) & 0x20) != 0 )
                  goto LABEL_287;
                if ( !v67 )
                {
LABEL_474:
                  v102 = (_DWORD *)(a1 + 64);
LABEL_405:
                  sub_684B30(0x488u, v102);
                  v146 = v154;
                  goto LABEL_287;
                }
              }
              v102 = v67 + 7;
              goto LABEL_405;
            }
          }
        }
        v67 = (__int64 *)*v67;
        if ( !v67 )
          goto LABEL_286;
      }
    }
    if ( HIDWORD(qword_4F077B4) && (*(_BYTE *)(a1 + 179) & 0x20) == 0 )
      goto LABEL_474;
LABEL_287:
    v154 = v146;
  }
  if ( dword_4F077C4 == 2 )
    sub_7A80F0(&v151);
LABEL_208:
  v50 = unk_4F068A8;
  if ( unk_4F068A8 )
  {
    if ( v153 )
      sub_7A6D50((__int64)&v151);
    v51 = v154;
    v59 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 )
      goto LABEL_245;
    if ( *(_BYTE *)(a1 + 140) == 12 )
    {
      v60 = a1;
      do
        v60 = *(_QWORD *)(v60 + 160);
      while ( *(_BYTE *)(v60 + 140) == 12 );
      v59 = *(_QWORD *)v60;
    }
    v50 = 1;
    if ( *(char *)(*(_QWORD *)(v59 + 96) + 178LL) >= 0 )
    {
LABEL_245:
      v50 = 0;
      if ( dword_4F077C4 == 2 && (*(_BYTE *)(a1 + 176) & 0x10) == 0 )
      {
        v50 = 0;
        v61 = *(_QWORD *)(a1 + 168);
        *(_QWORD *)(v61 + 32) = v152;
        *(_DWORD *)(v61 + 40) = v51;
      }
    }
  }
  else
  {
    v51 = v154;
  }
  if ( v160 && v160 > v152 )
  {
    v152 = v160;
    v153 = 0;
  }
  if ( (unsigned int)sub_7A6EA0(&v152, &v153, v51) || v155 )
  {
    v52 = v152;
    if ( dword_4F077C4 != 2 )
      goto LABEL_216;
  }
  else
  {
    sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
    v155 = 1;
    v52 = v152;
    if ( dword_4F077C4 != 2 )
      goto LABEL_216;
  }
  if ( unk_4F06AB8 >= v52 || v155 )
  {
LABEL_320:
    if ( !dword_4D0425C )
      goto LABEL_216;
    v76 = *(__int64 **)v151[21];
    if ( !v76 )
      goto LABEL_216;
    goto LABEL_325;
  }
  v75 = (__int64 **)v151[21];
  v76 = *v75;
  if ( *v75 )
  {
    v77 = *v75;
    do
    {
      if ( unk_4F06AB8 < (unsigned __int64)v77[13] )
      {
        sub_686C60(0x2C6u, (FILE *)(v77 + 9), *(_QWORD *)v77[5], *v151);
        v52 = v152;
        goto LABEL_320;
      }
      v77 = (__int64 *)*v77;
    }
    while ( v77 );
    if ( dword_4D0425C )
    {
      do
      {
LABEL_325:
        if ( (v76[12] & 2) != 0 && v76[13] > v52 )
        {
          v78 = (FILE *)(v76 + 9);
          v79 = *v151;
          v80 = *(_QWORD *)v76[5];
          if ( unk_4D04258 )
          {
            sub_686C80(0x504u, v78, v80, v79);
            v52 = v152;
          }
          else
          {
            sub_686C80(0x503u, v78, v80, v79);
            v52 = v152;
            v76[13] = v152;
          }
        }
        v76 = (__int64 *)*v76;
      }
      while ( v76 );
    }
  }
LABEL_216:
  v53 = v154;
  *(_QWORD *)(a1 + 128) = v52;
  *(_DWORD *)(a1 + 136) = v53;
  if ( v52 )
  {
    v54 = 0;
    if ( dword_4F077C4 != 2 )
      goto LABEL_252;
LABEL_220:
    if ( (*(_BYTE *)(a1 + 176) & 0x10) != 0 )
      goto LABEL_252;
    goto LABEL_221;
  }
  v54 = dword_4F077C0;
  if ( !dword_4F077C0 )
  {
    if ( dword_4F077BC )
    {
      v107 = *(_QWORD *)(a1 + 160);
      if ( v107 )
      {
        while ( 1 )
        {
          v108 = *(_QWORD *)(v107 + 120);
          for ( i2 = v108; *(_BYTE *)(i2 + 140) == 12; i2 = *(_QWORD *)(i2 + 160) )
            ;
          if ( *(_QWORD *)(i2 + 128)
            || !(unsigned int)sub_8D3410(v108) && !(unsigned int)sub_8D3A70(*(_QWORD *)(v107 + 120)) )
          {
            break;
          }
          v107 = *(_QWORD *)(v107 + 112);
          if ( !v107 )
          {
            v110 = *(_QWORD **)(a1 + 168);
            if ( v110 && *v110 )
              break;
            v52 = *(_QWORD *)(a1 + 128);
            if ( dword_4F077C4 == 2 && (*(_BYTE *)(a1 + 176) & 0x10) == 0 )
              goto LABEL_221;
            goto LABEL_249;
          }
        }
      }
    }
    v52 = 1;
    v54 = 1;
    *(_QWORD *)(a1 + 128) = 1;
    if ( dword_4F077C4 != 2 )
      goto LABEL_252;
    goto LABEL_220;
  }
  if ( dword_4F077C4 == 2 )
  {
    v54 = 0;
    if ( (*(_BYTE *)(a1 + 176) & 0x10) == 0 )
    {
LABEL_221:
      if ( !unk_4F068A8 || v54 | v50 )
      {
        v55 = *(_QWORD *)(a1 + 168);
        *(_QWORD *)(v55 + 32) = v52;
        *(_DWORD *)(v55 + 40) = *(_DWORD *)(a1 + 136);
        v52 = *(_QWORD *)(a1 + 128);
      }
LABEL_249:
      if ( v52 )
        goto LABEL_252;
    }
  }
  if ( HIDWORD(qword_4F077B4) )
    goto LABEL_255;
  v52 = 0;
LABEL_252:
  v62 = *(unsigned int *)(a1 + 136);
  if ( v62 > v52 )
  {
    *(_QWORD *)(a1 + 128) = v62;
    if ( dword_4F077C4 == 2 )
      *(_QWORD *)(*(_QWORD *)(a1 + 168) + 32LL) = *(unsigned int *)(*(_QWORD *)(a1 + 168) + 40LL);
  }
LABEL_255:
  result = a1;
  *(_BYTE *)(a1 + 141) &= ~0x20u;
  if ( *(_BYTE *)(a1 + 140) == 11 && (*(_BYTE *)(a1 + 179) & 0x10) != 0 )
  {
    result = sub_5D0610(a1, (__int64)dword_4F07508);
    if ( !(_DWORD)result )
      *(_BYTE *)(a1 + 179) &= ~0x10u;
  }
  return result;
}
