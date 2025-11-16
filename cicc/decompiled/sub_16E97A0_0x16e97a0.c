// Function: sub_16E97A0
// Address: 0x16e97a0
//
unsigned __int64 __fastcall sub_16E97A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  const char *v7; // rax
  bool v8; // dl
  bool v9; // cf
  bool v10; // zf
  const char *v11; // rdi
  const char *v12; // rsi
  __int64 v13; // rax
  int v14; // r13d
  __int64 v15; // rdx
  int v16; // eax
  size_t v17; // r14
  int v18; // eax
  unsigned __int64 v19; // rbx
  char *v20; // rax
  int v21; // ecx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rdx
  char *v25; // rax
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // rcx
  unsigned __int64 v29; // rdi
  __int64 v30; // rdx
  _BYTE *v31; // r15
  const char *v32; // rdx
  unsigned __int64 v33; // rax
  unsigned __int64 result; // rax
  char v35; // cl
  const char *v36; // rbx
  char v37; // dl
  const char *v38; // rcx
  const char *v39; // rdx
  unsigned __int64 v40; // rcx
  int v41; // ebx
  char v42; // r13
  int v43; // eax
  signed __int64 v44; // rax
  signed __int64 v45; // rcx
  unsigned __int64 v46; // rdx
  signed __int64 v47; // rsi
  __int64 v48; // rdx
  signed __int64 v49; // rdx
  signed __int64 v50; // rcx
  signed __int64 v51; // rsi
  __int64 v52; // rax
  const char *v53; // rax
  unsigned int v54; // ebx
  __int64 v55; // r9
  __int64 v56; // rsi
  int v57; // eax
  __int64 v58; // rdx
  int v59; // ebx
  __int64 v60; // r14
  unsigned __int8 v61; // r13
  unsigned __int8 v62; // al
  int v63; // edx
  char *v64; // rax
  unsigned __int8 v65; // cl
  char v66; // si
  unsigned __int8 v67; // r10
  int v68; // ecx
  __int64 j; // rdi
  __int64 v70; // rax
  int v71; // r8d
  unsigned __int64 v72; // r14
  _BYTE *v73; // r11
  __int64 v74; // rsi
  signed __int64 v75; // rdx
  signed __int64 v76; // rcx
  signed __int64 v77; // rsi
  __int64 v78; // rcx
  char v79; // dl
  _BYTE *v80; // rbx
  unsigned __int8 v81; // al
  const char *v82; // rbx
  const char *v83; // rbx
  int v84; // edi
  unsigned __int8 *v85; // r13
  const char *i; // r14
  size_t v87; // r13
  const char *v88; // rcx
  char **v89; // r14
  char *v90; // rax
  unsigned __int8 v91; // dl
  _BYTE *v92; // rcx
  const char *v93; // rbx
  __int64 v94; // r13
  size_t v95; // rax
  unsigned __int64 v96; // rdi
  unsigned __int64 v97; // rsi
  int v98; // edx
  int v99; // ecx
  int v100; // r8d
  int v101; // r9d
  char *v102; // rax
  __int64 v103; // rsi
  char v104; // al
  int v105; // edx
  __int64 v106; // rdx
  unsigned __int64 v107; // rdi
  int v108; // [rsp+0h] [rbp-40h]
  unsigned int v109; // [rsp+4h] [rbp-3Ch]
  int c[2]; // [rsp+8h] [rbp-38h]
  int ca[2]; // [rsp+8h] [rbp-38h]

  v7 = *(const char **)a1;
  if ( *(_QWORD *)(a1 + 8) <= (unsigned __int64)(*(_QWORD *)a1 + 5LL) )
    goto LABEL_7;
  v8 = memcmp(v7, "[:<:]]", 6u) != 0;
  v9 = 0;
  v10 = !v8;
  if ( !v8 )
  {
    if ( !*(_DWORD *)(a1 + 16) )
    {
      v49 = *(_QWORD *)(a1 + 40);
      v50 = *(_QWORD *)(a1 + 32);
      if ( v49 >= v50 )
      {
        v51 = ((v50 + 1 + ((unsigned __int64)(v50 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v50 + 1) / 2;
        if ( v50 < v51 )
        {
          sub_16E90A0(a1, v51, v49, v50, 0, a6);
          v49 = *(_QWORD *)(a1 + 40);
        }
      }
      v52 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v49 + 1;
      *(_QWORD *)(v52 + 8 * v49) = 2550136832LL;
      v7 = *(const char **)a1;
    }
    goto LABEL_40;
  }
  a4 = 6;
  v11 = "[:>:]]";
  v12 = v7;
  do
  {
    if ( !a4 )
      break;
    v9 = *v12 < (unsigned int)*v11;
    v10 = *v12++ == *v11++;
    --a4;
  }
  while ( v10 );
  if ( (!v9 && !v10) == v9 )
  {
    if ( !*(_DWORD *)(a1 + 16) )
    {
      v44 = *(_QWORD *)(a1 + 40);
      v45 = *(_QWORD *)(a1 + 32);
      if ( v44 >= v45 )
      {
        v46 = (v45 + 1 + ((unsigned __int64)(v45 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL;
        v47 = v46 + (v45 + 1) / 2;
        if ( v45 < v47 )
        {
          sub_16E90A0(a1, v47, v46, v45, a5, a6);
          v44 = *(_QWORD *)(a1 + 40);
        }
      }
      v48 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v44 + 1;
      *(_QWORD *)(v48 + 8 * v44) = 2684354560LL;
      v7 = *(const char **)a1;
    }
LABEL_40:
    result = (unsigned __int64)(v7 + 6);
    *(_QWORD *)a1 = result;
    return result;
  }
LABEL_7:
  v13 = *(_QWORD *)(a1 + 56);
  v14 = *(_DWORD *)(v13 + 20);
  *(_DWORD *)(v13 + 20) = v14 + 1;
  v15 = *(_QWORD *)(a1 + 56);
  v16 = *(_DWORD *)(a1 + 48);
  v17 = *(int *)(v15 + 16);
  if ( v14 < v16 )
    goto LABEL_14;
  v18 = v16 + 8;
  *(_DWORD *)(a1 + 48) = v18;
  if ( (unsigned __int64)v18 > 0x7FFFFFFFFFFFFFFLL )
  {
    v29 = *(_QWORD *)(v15 + 24);
LABEL_100:
    _libc_free(v29);
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL) = 0;
    _libc_free(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL));
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL) = 0;
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 12;
    result = (unsigned __int64)&unk_4FA17D0;
    *(_QWORD *)a1 = &unk_4FA17D0;
    *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
    return result;
  }
  v19 = v17 * ((unsigned __int64)v18 >> 3);
  v20 = realloc(*(_QWORD *)(v15 + 24), 32LL * v18, v15, a4, a5, a6);
  if ( !v20
    || (v24 = *(_QWORD *)(a1 + 56),
        *(_QWORD *)(v24 + 24) = v20,
        (v25 = realloc(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL), v19, v24, v21, v22, v23)) == 0) )
  {
    v29 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL);
    goto LABEL_100;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL) = v25;
  v26 = 0;
  if ( v14 > 0 )
  {
    do
    {
      v27 = v26;
      v28 = v26++;
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL) + 32 * v28) = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL)
                                                                       + v17 * (v27 >> 3);
    }
    while ( v14 != v26 );
  }
  memset((void *)(v19 - v17 + *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL)), 0, v17);
  v15 = *(_QWORD *)(a1 + 56);
LABEL_14:
  v29 = *(_QWORD *)(v15 + 24);
  if ( !v29 )
    goto LABEL_100;
  v30 = *(_QWORD *)(v15 + 32);
  if ( !v30 )
    goto LABEL_100;
  v31 = (_BYTE *)(v29 + 32LL * v14);
  v31[9] = 0;
  *((_QWORD *)v31 + 2) = 0;
  *((_QWORD *)v31 + 3) = 0;
  *(_QWORD *)v31 = v30 + v14 / 8 * v17;
  v31[8] = 1 << (v14 % 8);
  v32 = *(const char **)a1;
  v33 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 < v33 )
  {
    v35 = *v32;
    v108 = 0;
    if ( *v32 == 94 )
    {
      *(_QWORD *)a1 = v32 + 1;
      if ( v33 <= (unsigned __int64)(v32 + 1) )
        goto LABEL_17;
      v35 = v32[1];
      v108 = 1;
      ++v32;
    }
    if ( v35 == 93 )
    {
      *(_QWORD *)a1 = v32 + 1;
      *(_BYTE *)(*(_QWORD *)v31 + 93LL) |= v31[8];
      v31[9] += 93;
      goto LABEL_47;
    }
    if ( v35 == 45 )
    {
      *(_QWORD *)a1 = v32 + 1;
      *(_BYTE *)(*(_QWORD *)v31 + 45LL) |= v31[8];
      v31[9] += 45;
      goto LABEL_47;
    }
    v36 = *(const char **)a1;
    v37 = **(_BYTE **)a1;
    if ( v37 != 93 )
    {
      do
      {
        v38 = v36 + 1;
        if ( v37 == 45 )
        {
          if ( (unsigned __int64)v38 < v33 && v36[1] == 93 )
          {
            if ( v33 > (unsigned __int64)v36 )
            {
              *(_QWORD *)a1 = v38;
              *(_BYTE *)(*(_QWORD *)v31 + 45LL) |= v31[8];
              v31[9] += 45;
            }
            break;
          }
          if ( *(_DWORD *)(a1 + 16) )
            goto LABEL_21;
          *(_DWORD *)(a1 + 16) = 11;
          *(_QWORD *)a1 = &unk_4FA17D0;
          *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
          return sub_16E8E60(a1, (__int64)v31);
        }
        if ( v37 == 91 && (unsigned __int64)v38 < v33 )
        {
          v79 = v36[1];
          if ( v79 == 58 )
          {
            v83 = v36 + 2;
            *(_QWORD *)a1 = v83;
            if ( (unsigned __int64)v83 >= v33 )
            {
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 7;
              *(_QWORD *)a1 = &unk_4FA17D0;
              v83 = (const char *)&unk_4FA17D0;
              *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
            }
            v84 = *(unsigned __int8 *)v83;
            if ( (_BYTE)v84 == 45 || (_BYTE)v84 == 93 )
            {
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 4;
              v87 = 0;
              *(_QWORD *)a1 = &unk_4FA17D0;
              v83 = (const char *)&unk_4FA17D0;
              *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
            }
            else
            {
              v85 = *(unsigned __int8 **)(a1 + 8);
              if ( v85 <= (unsigned __int8 *)v83 )
              {
                v87 = 0;
              }
              else
              {
                for ( i = v83; isalpha(v84); v84 = *(unsigned __int8 *)i )
                {
                  *(_QWORD *)a1 = ++i;
                  if ( i == (const char *)v85 )
                    break;
                }
                v87 = i - v83;
              }
            }
            v88 = off_4CD3DA0;
            if ( !off_4CD3DA0 )
            {
LABEL_137:
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 4;
              goto LABEL_21;
            }
            v89 = &off_4CD3DA0;
            while ( 1 )
            {
              *(_QWORD *)ca = v88;
              if ( !strncmp(v88, v83, v87) && !*(_BYTE *)(*(_QWORD *)ca + v87) )
                break;
              v88 = v89[3];
              v89 += 3;
              if ( !v88 )
                goto LABEL_137;
            }
            v90 = v89[1];
            v91 = *v90;
            v92 = v90 + 1;
            if ( *v90 )
            {
              do
              {
                ++v92;
                *(_BYTE *)(*(_QWORD *)v31 + v91) |= v31[8];
                v31[9] += v91;
                v91 = *(v92 - 1);
              }
              while ( v91 );
            }
            v93 = v89[2];
            if ( *v93 )
            {
              do
              {
                while ( 1 )
                {
                  v94 = *((_QWORD *)v31 + 2);
                  v95 = strlen(v93);
                  v96 = *((_QWORD *)v31 + 3);
                  v97 = v94 + v95 + 1;
                  *((_QWORD *)v31 + 2) = v97;
                  v102 = realloc(v96, v97, v98, v99, v100, v101);
                  if ( !v102 )
                    break;
                  v103 = *((_QWORD *)v31 + 2);
                  *((_QWORD *)v31 + 3) = v102;
                  sub_16F0650(&v102[v94 - 1], v93, v103 + 1 - v94);
                  v93 += strlen(v93) + 1;
                  if ( !*v93 )
                    goto LABEL_144;
                }
                v107 = *((_QWORD *)v31 + 3);
                if ( v107 )
                  _libc_free(v107);
                *((_QWORD *)v31 + 3) = 0;
                if ( !*(_DWORD *)(a1 + 16) )
                  *(_DWORD *)(a1 + 16) = 12;
                *(_QWORD *)a1 = &unk_4FA17D0;
                *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
                v93 += strlen(v93) + 1;
              }
              while ( *v93 );
LABEL_149:
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 7;
              goto LABEL_21;
            }
LABEL_144:
            v82 = *(const char **)a1;
            v33 = *(_QWORD *)(a1 + 8);
            if ( *(_QWORD *)a1 >= v33 )
              goto LABEL_149;
            if ( v33 <= (unsigned __int64)(v82 + 1) || *v82 != 58 || v82[1] != 93 )
            {
              if ( *(_DWORD *)(a1 + 16) )
                goto LABEL_21;
              *(_DWORD *)(a1 + 16) = 4;
              *(_QWORD *)a1 = &unk_4FA17D0;
              *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
              return sub_16E8E60(a1, (__int64)v31);
            }
            goto LABEL_174;
          }
          if ( v79 == 61 )
          {
            v80 = v36 + 2;
            *(_QWORD *)a1 = v80;
            if ( (unsigned __int64)v80 >= v33 )
            {
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 7;
              *(_QWORD *)a1 = &unk_4FA17D0;
              v80 = &unk_4FA17D0;
              *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
            }
            if ( *v80 == 45 || *v80 == 93 )
            {
              if ( !*(_DWORD *)(a1 + 16) )
                *(_DWORD *)(a1 + 16) = 3;
              *(_QWORD *)a1 = &unk_4FA17D0;
              *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
            }
            v81 = sub_16E8EC0((const char **)a1, 61);
            *(_BYTE *)(*(_QWORD *)v31 + v81) |= v31[8];
            v31[9] += v81;
            v82 = *(const char **)a1;
            v33 = *(_QWORD *)(a1 + 8);
            if ( *(_QWORD *)a1 >= v33 )
              goto LABEL_149;
            if ( v33 <= (unsigned __int64)(v82 + 1) || *v82 != 61 || v82[1] != 93 )
            {
              if ( *(_DWORD *)(a1 + 16) )
                goto LABEL_21;
              *(_DWORD *)(a1 + 16) = 3;
              *(_QWORD *)a1 = &unk_4FA17D0;
              *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
              return sub_16E8E60(a1, (__int64)v31);
            }
LABEL_174:
            v36 = v82 + 2;
            *(_QWORD *)a1 = v36;
            goto LABEL_48;
          }
        }
        LOBYTE(v43) = sub_16E8FF0(a1);
        v39 = *(const char **)a1;
        v40 = *(_QWORD *)(a1 + 8);
        v41 = (char)v43;
        v42 = v43;
        v43 = (char)v43;
        if ( *(_QWORD *)a1 < v40 && *v39 == 45 && v40 > (unsigned __int64)(v39 + 1) && v39[1] != 93 )
        {
          *(_QWORD *)a1 = v39 + 1;
          v104 = v39[1];
          if ( v104 == 45 )
          {
            *(_QWORD *)a1 = v39 + 2;
            v105 = 45;
          }
          else
          {
            v104 = sub_16E8FF0(a1);
            v105 = v104;
          }
          if ( v42 > v104 )
          {
            if ( !*(_DWORD *)(a1 + 16) )
              *(_DWORD *)(a1 + 16) = 11;
            *(_QWORD *)a1 = &unk_4FA17D0;
            *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
          }
          if ( v105 < v41 )
            goto LABEL_47;
          v43 = v41;
          v41 = v105;
        }
        do
        {
          *(_BYTE *)(*(_QWORD *)v31 + (unsigned __int8)v43) |= v31[8];
          v31[9] += v43++;
        }
        while ( v43 != v41 + 1 );
LABEL_47:
        v36 = *(const char **)a1;
        v33 = *(_QWORD *)(a1 + 8);
LABEL_48:
        if ( (unsigned __int64)v36 >= v33 )
          goto LABEL_17;
        v37 = *v36;
      }
      while ( *v36 != 93 );
    }
    v53 = *(const char **)a1;
    if ( *(_QWORD *)a1 >= *(_QWORD *)(a1 + 8) )
      goto LABEL_17;
    v54 = *(_DWORD *)(a1 + 16);
    *(_QWORD *)a1 = v53 + 1;
    v109 = v54;
    if ( *v53 != 93 )
      goto LABEL_18;
    LODWORD(v55) = v54;
    if ( v54 )
      return sub_16E8E60(a1, (__int64)v31);
    v56 = *(_QWORD *)(a1 + 56);
    v57 = *(_DWORD *)(v56 + 40);
    if ( (v57 & 2) != 0 )
    {
      v58 = *(int *)(v56 + 16);
      v59 = *(_DWORD *)(v56 + 16) - 1;
      if ( v59 < 0 )
      {
        if ( v108 )
        {
LABEL_71:
          if ( (v57 & 8) != 0 )
          {
            *(_BYTE *)(*(_QWORD *)v31 + 10LL) &= ~v31[8];
            v31[9] -= 10;
            v56 = *(_QWORD *)(a1 + 56);
          }
          v58 = *(int *)(v56 + 16);
        }
LABEL_74:
        if ( v58 )
        {
          v55 = *(_QWORD *)v31;
          v67 = v31[8];
          v68 = 0;
          for ( j = 0; ; ++j )
          {
            v68 -= ((v67 & *(_BYTE *)(v55 + (unsigned __int8)j)) == 0) - 1;
            if ( v58 == j + 1 )
              break;
          }
          v70 = 0;
          if ( v68 == 1 )
          {
            while ( 1 )
            {
              v106 = (unsigned __int8)v70;
              if ( (*(_BYTE *)(v55 + (unsigned __int8)v70) & v67) != 0 )
                break;
              v106 = v70 + 1;
              if ( v70 == j )
                goto LABEL_171;
              ++v70;
            }
            v109 = (char)v70;
LABEL_171:
            sub_16EA3B0(a1, v109, v106);
            return sub_16E8E60(a1, (__int64)v31);
          }
        }
        result = *(_QWORD *)(v56 + 24);
        v71 = (unsigned __int8)v31[9];
        *(_QWORD *)c = result;
        v72 = result + 32LL * *(int *)(v56 + 20);
        if ( result < v72 )
        {
          v73 = *(_BYTE **)(v56 + 24);
          while ( 1 )
          {
            if ( v73[9] == (_BYTE)v71 && v31 != v73 )
            {
              if ( !v58 )
                break;
              result = 0;
              v55 = *(_QWORD *)v31;
              while ( ((*(_BYTE *)(*(_QWORD *)v73 + (unsigned __int8)result) & v73[8]) != 0) == ((*(_BYTE *)(v55 + (unsigned __int8)result)
                                                                                                & v31[8]) != 0) )
              {
                if ( v58 == ++result )
                  goto LABEL_89;
              }
              if ( v58 == result )
                break;
            }
            v73 += 32;
            if ( v72 <= (unsigned __int64)v73 )
              goto LABEL_91;
          }
LABEL_89:
          if ( v72 > (unsigned __int64)v73 )
          {
            v74 = (__int64)v31;
            v31 = v73;
            sub_16E8E60(a1, v74);
            result = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL);
            *(_QWORD *)c = result;
          }
        }
LABEL_91:
        if ( !*(_DWORD *)(a1 + 16) )
        {
          v75 = *(_QWORD *)(a1 + 40);
          v76 = *(_QWORD *)(a1 + 32);
          if ( v75 >= v76 )
          {
            v77 = ((v76 + 1 + ((unsigned __int64)(v76 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v76 + 1) / 2;
            if ( v76 < v77 )
            {
              sub_16E90A0(a1, v77, v75, v76, v71, v55);
              v75 = *(_QWORD *)(a1 + 40);
            }
          }
          v78 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 40) = v75 + 1;
          result = (int)(((__int64)&v31[-*(_QWORD *)c] >> 5) | 0x30000000);
          *(_QWORD *)(v78 + 8 * v75) = result;
        }
        return result;
      }
      do
      {
        v60 = *(_QWORD *)v31;
        v61 = v31[8];
        if ( (*(_BYTE *)(*(_QWORD *)v31 + (unsigned __int8)v59) & v61) != 0 && isalpha(v59) )
        {
          if ( isupper((unsigned __int8)v59) )
          {
            v62 = tolower((unsigned __int8)v59);
          }
          else
          {
            LODWORD(v55) = islower((unsigned __int8)v59);
            v62 = v59;
            if ( (_DWORD)v55 )
              v62 = toupper((unsigned __int8)v59);
          }
          if ( (char)v62 != v59 )
          {
            *(_BYTE *)(v60 + v62) |= v61;
            v31[9] += v62;
          }
        }
        v9 = v59-- == 0;
      }
      while ( !v9 );
      v56 = *(_QWORD *)(a1 + 56);
    }
    v58 = *(int *)(v56 + 16);
    if ( v108 )
    {
      v63 = v58 - 1;
      if ( v63 >= 0 )
      {
        do
        {
          while ( 1 )
          {
            v64 = (char *)(*(_QWORD *)v31 + (unsigned __int8)v63);
            v65 = v31[8];
            v66 = *v64;
            if ( (v65 & (unsigned __int8)*v64) == 0 )
              break;
            *v64 = v66 & ~v65;
            v31[9] -= v63;
            v9 = v63-- == 0;
            if ( v9 )
              goto LABEL_70;
          }
          *v64 = v66 | v65;
          v31[9] += v63;
          v9 = v63-- == 0;
        }
        while ( !v9 );
LABEL_70:
        v56 = *(_QWORD *)(a1 + 56);
        v57 = *(_DWORD *)(v56 + 40);
      }
      else
      {
        v57 = *(_DWORD *)(v56 + 40);
      }
      goto LABEL_71;
    }
    goto LABEL_74;
  }
LABEL_17:
  v109 = *(_DWORD *)(a1 + 16);
LABEL_18:
  if ( v109 )
  {
LABEL_21:
    *(_QWORD *)a1 = &unk_4FA17D0;
    *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
  }
  else
  {
    *(_DWORD *)(a1 + 16) = 7;
    *(_QWORD *)a1 = &unk_4FA17D0;
    *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
  }
  return sub_16E8E60(a1, (__int64)v31);
}
