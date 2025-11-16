// Function: sub_CB7E40
// Address: 0xcb7e40
//
unsigned __int64 __fastcall sub_CB7E40(__int64 a1, __int64 a2)
{
  const char *v3; // rax
  bool v4; // dl
  bool v5; // cf
  bool v6; // zf
  __int64 v7; // rcx
  const char *v8; // rdi
  __int64 v9; // rax
  int v10; // r13d
  __int64 v11; // rdx
  int v12; // eax
  size_t v13; // r14
  int v14; // eax
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rdx
  _BYTE *v23; // r15
  __int64 v24; // rax
  const char *v25; // rdx
  char v26; // cl
  const char *v27; // rbx
  __int64 v28; // rdx
  char v29; // cl
  char v30; // dl
  unsigned __int8 *v31; // rbx
  unsigned __int8 v32; // al
  const char *v33; // rbx
  unsigned __int64 result; // rax
  signed __int64 v35; // rdx
  signed __int64 v36; // rcx
  signed __int64 v37; // rsi
  __int64 v38; // rax
  signed __int64 v39; // rdx
  signed __int64 v40; // rcx
  signed __int64 v41; // rsi
  __int64 v42; // rax
  char v43; // r13
  const char *v44; // rax
  int v45; // ebx
  int v46; // eax
  __int64 v47; // rcx
  char v48; // dl
  char v49; // al
  int v50; // ecx
  const char *v51; // rbx
  int v52; // edi
  __int64 v53; // r13
  const char *i; // r14
  size_t v55; // r13
  const char *v56; // rcx
  char **v57; // r14
  char *v58; // rax
  unsigned __int8 v59; // dl
  _BYTE *v60; // rcx
  const char *v61; // rbx
  __int64 v62; // r13
  size_t v63; // rax
  void *v64; // rdi
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rsi
  const char *v68; // rdx
  __int64 v69; // rsi
  int v70; // eax
  __int64 v71; // rdx
  int v72; // ebx
  __int64 v73; // r14
  unsigned __int8 v74; // r13
  int v75; // r9d
  unsigned __int8 v76; // al
  int v77; // edx
  char *v78; // rax
  unsigned __int8 v79; // cl
  char v80; // si
  __int64 v81; // r9
  unsigned __int8 v82; // r10
  int v83; // ecx
  __int64 j; // rdi
  __int64 v85; // rax
  unsigned __int64 v86; // r14
  _BYTE *v87; // r11
  __int64 v88; // rsi
  signed __int64 v89; // rdx
  signed __int64 v90; // rcx
  signed __int64 v91; // rsi
  __int64 v92; // rcx
  __int64 v93; // rdx
  __int64 v94; // rdi
  int v95; // [rsp+0h] [rbp-40h]
  unsigned int v96; // [rsp+8h] [rbp-38h]
  const char *v97; // [rsp+8h] [rbp-38h]
  unsigned __int64 v98; // [rsp+8h] [rbp-38h]

  v3 = *(const char **)a1;
  if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) <= 5 )
    goto LABEL_7;
  v4 = memcmp(v3, "[:<:]]", 6u) != 0;
  v5 = 0;
  v6 = !v4;
  if ( !v4 )
  {
    if ( !*(_DWORD *)(a1 + 16) )
    {
      v35 = *(_QWORD *)(a1 + 40);
      v36 = *(_QWORD *)(a1 + 32);
      if ( v35 >= v36 )
      {
        v37 = ((v36 + 1 + ((unsigned __int64)(v36 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v36 + 1) / 2;
        if ( v36 < v37 )
        {
          sub_CB7740(a1, v37);
          v35 = *(_QWORD *)(a1 + 40);
        }
      }
      v38 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v35 + 1;
      *(_QWORD *)(v38 + 8 * v35) = 2550136832LL;
      v3 = *(const char **)a1;
    }
    goto LABEL_42;
  }
  v7 = 6;
  v8 = "[:>:]]";
  a2 = (__int64)v3;
  do
  {
    if ( !v7 )
      break;
    v5 = *(_BYTE *)a2 < *v8;
    v6 = *(_BYTE *)a2++ == *v8++;
    --v7;
  }
  while ( v6 );
  if ( (!v5 && !v6) == v5 )
  {
    if ( !*(_DWORD *)(a1 + 16) )
    {
      v39 = *(_QWORD *)(a1 + 40);
      v40 = *(_QWORD *)(a1 + 32);
      if ( v39 >= v40 )
      {
        v41 = ((v40 + 1 + ((unsigned __int64)(v40 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v40 + 1) / 2;
        if ( v40 < v41 )
        {
          sub_CB7740(a1, v41);
          v39 = *(_QWORD *)(a1 + 40);
        }
      }
      v42 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v39 + 1;
      *(_QWORD *)(v42 + 8 * v39) = 2684354560LL;
      v3 = *(const char **)a1;
    }
LABEL_42:
    result = (unsigned __int64)(v3 + 6);
    *(_QWORD *)a1 = result;
    return result;
  }
LABEL_7:
  v9 = *(_QWORD *)(a1 + 56);
  v10 = *(_DWORD *)(v9 + 20);
  *(_DWORD *)(v9 + 20) = v10 + 1;
  v11 = *(_QWORD *)(a1 + 56);
  v12 = *(_DWORD *)(a1 + 48);
  v13 = *(int *)(v11 + 16);
  if ( v10 < v12 )
    goto LABEL_14;
  v14 = v12 + 8;
  *(_DWORD *)(a1 + 48) = v14;
  a2 = v14;
  if ( (unsigned __int64)v14 > 0x7FFFFFFFFFFFFFFLL )
  {
    v21 = *(_QWORD *)(v11 + 24);
LABEL_58:
    _libc_free(v21, a2);
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL) = 0;
    _libc_free(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL), a2);
    *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL) = 0;
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 12;
    *(_QWORD *)a1 = byte_4F85140;
    *(_QWORD *)(a1 + 8) = byte_4F85140;
    return (unsigned __int64)byte_4F85140;
  }
  a2 = 32LL * v14;
  v15 = v13 * ((unsigned __int64)v14 >> 3);
  v16 = realloc(*(void **)(v11 + 24));
  if ( !v16
    || (a2 = v15,
        *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL) = v16,
        (v17 = realloc(*(void **)(*(_QWORD *)(a1 + 56) + 32LL))) == 0) )
  {
    v21 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL);
    goto LABEL_58;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL) = v17;
  v18 = 0;
  if ( v10 > 0 )
  {
    do
    {
      v19 = v18;
      v20 = v18++;
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL) + 32 * v20) = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL)
                                                                       + v13 * (v19 >> 3);
    }
    while ( v10 != v18 );
  }
  a2 = 0;
  memset((void *)(v15 - v13 + *(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL)), 0, v13);
  v11 = *(_QWORD *)(a1 + 56);
LABEL_14:
  v21 = *(_QWORD *)(v11 + 24);
  if ( !v21 )
    goto LABEL_58;
  v22 = *(_QWORD *)(v11 + 32);
  if ( !v22 )
    goto LABEL_58;
  v23 = (_BYTE *)(v21 + 32LL * v10);
  v23[9] = 0;
  *((_QWORD *)v23 + 2) = 0;
  *((_QWORD *)v23 + 3) = 0;
  *(_QWORD *)v23 = v22 + v10 / 8 * v13;
  v23[8] = 1 << (v10 % 8);
  v24 = *(_QWORD *)(a1 + 8);
  v25 = *(const char **)a1;
  if ( v24 - *(_QWORD *)a1 <= 0 )
    goto LABEL_52;
  v26 = *v25;
  v95 = 0;
  if ( *v25 == 94 )
  {
    *(_QWORD *)a1 = v25 + 1;
    if ( v24 - (__int64)(v25 + 1) > 0 )
    {
      v26 = v25[1];
      v95 = 1;
      ++v25;
      if ( v26 != 93 )
        goto LABEL_19;
LABEL_49:
      *(_QWORD *)a1 = v25 + 1;
      *(_BYTE *)(*(_QWORD *)v23 + 93LL) |= v23[8];
      v23[9] += 93;
      goto LABEL_50;
    }
LABEL_52:
    v96 = *(_DWORD *)(a1 + 16);
LABEL_125:
    if ( v96 )
    {
LABEL_96:
      *(_QWORD *)a1 = byte_4F85140;
      *(_QWORD *)(a1 + 8) = byte_4F85140;
    }
    else
    {
      *(_DWORD *)(a1 + 16) = 7;
      *(_QWORD *)a1 = byte_4F85140;
      *(_QWORD *)(a1 + 8) = byte_4F85140;
    }
    return sub_CB74F0(a1, (__int64)v23);
  }
  if ( v26 == 93 )
    goto LABEL_49;
LABEL_19:
  if ( v26 == 45 )
  {
    *(_QWORD *)a1 = v25 + 1;
    *(_BYTE *)(*(_QWORD *)v23 + 45LL) |= v23[8];
    v23[9] += 45;
    goto LABEL_50;
  }
  v27 = *(const char **)a1;
  v28 = v24 - *(_QWORD *)a1;
  while ( 1 )
  {
    v29 = *v27;
    if ( *v27 == 93 )
      break;
    if ( v28 == 1 )
    {
      if ( v29 == 45 )
        goto LABEL_78;
    }
    else if ( v29 == 45 )
    {
      if ( v27[1] != 93 )
      {
LABEL_78:
        if ( *(_DWORD *)(a1 + 16) )
          goto LABEL_96;
        *(_DWORD *)(a1 + 16) = 11;
        *(_QWORD *)a1 = byte_4F85140;
        *(_QWORD *)(a1 + 8) = byte_4F85140;
        return sub_CB74F0(a1, (__int64)v23);
      }
      *(_QWORD *)a1 = v27 + 1;
      *(_BYTE *)(*(_QWORD *)v23 + 45LL) |= v23[8];
      v23[9] += 45;
      v24 = *(_QWORD *)(a1 + 8);
      break;
    }
    if ( v29 == 91 && v28 != 1 )
    {
      v30 = v27[1];
      if ( v30 == 58 )
      {
        v51 = v27 + 2;
        *(_QWORD *)a1 = v51;
        if ( v24 - (__int64)v51 <= 0 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 7;
          *(_QWORD *)a1 = byte_4F85140;
          v51 = (const char *)byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
        }
        v52 = *(unsigned __int8 *)v51;
        if ( (_BYTE)v52 == 45 || (_BYTE)v52 == 93 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 4;
          v55 = 0;
          *(_QWORD *)a1 = byte_4F85140;
          v51 = (const char *)byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
        }
        else
        {
          v53 = *(_QWORD *)(a1 + 8);
          if ( v53 - (__int64)v51 <= 0 )
          {
            v55 = 0;
          }
          else
          {
            for ( i = v51; isalpha(v52); v52 = *(unsigned __int8 *)i )
            {
              *(_QWORD *)a1 = ++i;
              if ( v53 - (__int64)i <= 0 )
                break;
            }
            v55 = i - v51;
          }
        }
        v56 = off_4C5CD80;
        if ( !off_4C5CD80 )
        {
LABEL_94:
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 4;
          goto LABEL_96;
        }
        v57 = &off_4C5CD80;
        while ( 1 )
        {
          v97 = v56;
          if ( !strncmp(v56, v51, v55) && !v97[v55] )
            break;
          v56 = v57[3];
          v57 += 3;
          if ( !v56 )
            goto LABEL_94;
        }
        v58 = v57[1];
        v59 = *v58;
        v60 = v58 + 1;
        if ( *v58 )
        {
          do
          {
            ++v60;
            *(_BYTE *)(*(_QWORD *)v23 + v59) |= v23[8];
            v23[9] += v59;
            v59 = *(v60 - 1);
          }
          while ( v59 );
        }
        v61 = v57[2];
        if ( *v61 )
        {
          do
          {
            while ( 1 )
            {
              v62 = *((_QWORD *)v23 + 2);
              v63 = strlen(v61);
              v64 = (void *)*((_QWORD *)v23 + 3);
              v65 = v62 + v63 + 1;
              *((_QWORD *)v23 + 2) = v65;
              v66 = realloc(v64);
              if ( !v66 )
                break;
              v67 = *((_QWORD *)v23 + 2);
              *((_QWORD *)v23 + 3) = v66;
              sub_CBF040(v66 + v62 - 1, v61, v67 + 1 - v62);
              v61 += strlen(v61) + 1;
              if ( !*v61 )
                goto LABEL_102;
            }
            v94 = *((_QWORD *)v23 + 3);
            if ( v94 )
              _libc_free(v94, v65);
            *((_QWORD *)v23 + 3) = 0;
            if ( !*(_DWORD *)(a1 + 16) )
              *(_DWORD *)(a1 + 16) = 12;
            *(_QWORD *)a1 = byte_4F85140;
            *(_QWORD *)(a1 + 8) = byte_4F85140;
            v61 += strlen(v61) + 1;
          }
          while ( *v61 );
LABEL_108:
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 7;
          goto LABEL_96;
        }
LABEL_102:
        v24 = *(_QWORD *)(a1 + 8);
        v33 = *(const char **)a1;
        if ( v24 - *(_QWORD *)a1 <= 0 )
          goto LABEL_108;
        if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 == 1 || *v33 != 58 || v33[1] != 93 )
        {
          if ( *(_DWORD *)(a1 + 16) )
            goto LABEL_96;
          *(_DWORD *)(a1 + 16) = 4;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
          return sub_CB74F0(a1, (__int64)v23);
        }
        goto LABEL_128;
      }
      if ( v30 == 61 )
      {
        v31 = (unsigned __int8 *)(v27 + 2);
        *(_QWORD *)a1 = v31;
        if ( v24 - (__int64)v31 <= 0 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 7;
          *(_QWORD *)a1 = byte_4F85140;
          v31 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
        }
        if ( *v31 == 45 || *v31 == 93 )
        {
          if ( !*(_DWORD *)(a1 + 16) )
            *(_DWORD *)(a1 + 16) = 3;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
        }
        v32 = sub_CB7550((const char **)a1, 61);
        *(_BYTE *)(*(_QWORD *)v23 + v32) |= v23[8];
        v23[9] += v32;
        v24 = *(_QWORD *)(a1 + 8);
        v33 = *(const char **)a1;
        if ( v24 - *(_QWORD *)a1 <= 0 )
          goto LABEL_108;
        if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 == 1 || *v33 != 61 || v33[1] != 93 )
        {
          if ( *(_DWORD *)(a1 + 16) )
            goto LABEL_96;
          *(_DWORD *)(a1 + 16) = 3;
          *(_QWORD *)a1 = byte_4F85140;
          *(_QWORD *)(a1 + 8) = byte_4F85140;
          return sub_CB74F0(a1, (__int64)v23);
        }
LABEL_128:
        v27 = v33 + 2;
        *(_QWORD *)a1 = v27;
        goto LABEL_51;
      }
    }
    v43 = sub_CB7680(a1);
    v44 = *(const char **)a1;
    v45 = v43;
    if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) <= 0
      || *v44 != 45
      || *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 == 1
      || v44[1] == 93 )
    {
      v46 = v43;
    }
    else
    {
      v47 = *(_QWORD *)(a1 + 8) - (_QWORD)(v44 + 1);
      *(_QWORD *)a1 = v44 + 1;
      if ( v47 > 0 && (v48 = v44[1], v48 == 45) )
      {
        v50 = 45;
        *(_QWORD *)a1 = v44 + 2;
      }
      else
      {
        v49 = sub_CB7680(a1);
        v50 = v49;
        v48 = v49;
      }
      if ( v43 > v48 )
      {
        if ( !*(_DWORD *)(a1 + 16) )
          *(_DWORD *)(a1 + 16) = 11;
        *(_QWORD *)a1 = byte_4F85140;
        *(_QWORD *)(a1 + 8) = byte_4F85140;
      }
      if ( v50 < v43 )
        goto LABEL_50;
      v46 = v43;
      v45 = v50;
    }
    do
    {
      *(_BYTE *)(*(_QWORD *)v23 + (unsigned __int8)v46) |= v23[8];
      v23[9] += v46++;
    }
    while ( v46 != v45 + 1 );
LABEL_50:
    v24 = *(_QWORD *)(a1 + 8);
    v27 = *(const char **)a1;
LABEL_51:
    v28 = v24 - (_QWORD)v27;
    if ( v24 - (__int64)v27 <= 0 )
      goto LABEL_52;
  }
  v68 = *(const char **)a1;
  if ( v24 - *(_QWORD *)a1 <= 0 )
    goto LABEL_52;
  *(_QWORD *)a1 = v68 + 1;
  v96 = *(_DWORD *)(a1 + 16);
  if ( *v68 != 93 )
    goto LABEL_125;
  if ( v96 )
    return sub_CB74F0(a1, (__int64)v23);
  v69 = *(_QWORD *)(a1 + 56);
  v70 = *(_DWORD *)(v69 + 40);
  if ( (v70 & 2) != 0 )
  {
    v71 = *(int *)(v69 + 16);
    v72 = *(_DWORD *)(v69 + 16) - 1;
    if ( v72 >= 0 )
    {
      do
      {
        v73 = *(_QWORD *)v23;
        v74 = v23[8];
        if ( (*(_BYTE *)(*(_QWORD *)v23 + (unsigned __int8)v72) & v74) != 0 && isalpha(v72) )
        {
          if ( isupper((unsigned __int8)v72) )
          {
            v76 = tolower((unsigned __int8)v72);
          }
          else
          {
            v75 = islower((unsigned __int8)v72);
            v76 = v72;
            if ( v75 )
              v76 = toupper((unsigned __int8)v72);
          }
          if ( (char)v76 != v72 )
          {
            *(_BYTE *)(v73 + v76) |= v74;
            v23[9] += v76;
          }
        }
        v5 = v72-- == 0;
      }
      while ( !v5 );
      v69 = *(_QWORD *)(a1 + 56);
      goto LABEL_145;
    }
    if ( v95 )
    {
LABEL_152:
      if ( (v70 & 8) != 0 )
      {
        *(_BYTE *)(*(_QWORD *)v23 + 10LL) &= ~v23[8];
        v23[9] -= 10;
        v69 = *(_QWORD *)(a1 + 56);
      }
      v71 = *(int *)(v69 + 16);
    }
  }
  else
  {
LABEL_145:
    v71 = *(int *)(v69 + 16);
    if ( v95 )
    {
      v77 = v71 - 1;
      if ( v77 >= 0 )
      {
        do
        {
          while ( 1 )
          {
            v78 = (char *)(*(_QWORD *)v23 + (unsigned __int8)v77);
            v79 = v23[8];
            v80 = *v78;
            if ( (v79 & (unsigned __int8)*v78) == 0 )
              break;
            *v78 = v80 & ~v79;
            v23[9] -= v77;
            v5 = v77-- == 0;
            if ( v5 )
              goto LABEL_151;
          }
          *v78 = v80 | v79;
          v23[9] += v77;
          v5 = v77-- == 0;
        }
        while ( !v5 );
LABEL_151:
        v69 = *(_QWORD *)(a1 + 56);
        v70 = *(_DWORD *)(v69 + 40);
      }
      else
      {
        v70 = *(_DWORD *)(v69 + 40);
      }
      goto LABEL_152;
    }
  }
  if ( !v71 )
    goto LABEL_160;
  v81 = *(_QWORD *)v23;
  v82 = v23[8];
  v83 = 0;
  for ( j = 0; ; ++j )
  {
    v83 -= ((v82 & *(_BYTE *)(v81 + (unsigned __int8)j)) == 0) - 1;
    if ( v71 == j + 1 )
      break;
  }
  v85 = 0;
  if ( v83 != 1 )
  {
LABEL_160:
    result = *(_QWORD *)(v69 + 24);
    v98 = result;
    v86 = result + 32LL * *(int *)(v69 + 20);
    if ( result < v86 )
    {
      v87 = *(_BYTE **)(v69 + 24);
      while ( 1 )
      {
        if ( v87[9] == v23[9] && v23 != v87 )
        {
          if ( !v71 )
            break;
          result = 0;
          while ( ((*(_BYTE *)(*(_QWORD *)v87 + (unsigned __int8)result) & v87[8]) != 0) == ((*(_BYTE *)(*(_QWORD *)v23 + (unsigned __int8)result)
                                                                                            & v23[8]) != 0) )
          {
            if ( v71 == ++result )
              goto LABEL_170;
          }
          if ( v71 == result )
            break;
        }
        v87 += 32;
        if ( v86 <= (unsigned __int64)v87 )
          goto LABEL_172;
      }
LABEL_170:
      if ( (unsigned __int64)v87 < v86 )
      {
        v88 = (__int64)v23;
        v23 = v87;
        sub_CB74F0(a1, v88);
        result = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 24LL);
        v98 = result;
      }
    }
LABEL_172:
    if ( !*(_DWORD *)(a1 + 16) )
    {
      v89 = *(_QWORD *)(a1 + 40);
      v90 = *(_QWORD *)(a1 + 32);
      if ( v89 >= v90 )
      {
        v91 = ((v90 + 1 + ((unsigned __int64)(v90 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v90 + 1) / 2;
        if ( v90 < v91 )
        {
          sub_CB7740(a1, v91);
          v89 = *(_QWORD *)(a1 + 40);
        }
      }
      v92 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 40) = v89 + 1;
      result = (int)(((__int64)&v23[-v98] >> 5) | 0x30000000);
      *(_QWORD *)(v92 + 8 * v89) = result;
    }
    return result;
  }
  while ( 1 )
  {
    v93 = (unsigned __int8)v85;
    if ( (*(_BYTE *)(v81 + (unsigned __int8)v85) & v82) != 0 )
      break;
    v93 = v85 + 1;
    if ( v85 == j )
      goto LABEL_181;
    ++v85;
  }
  v96 = (char)v85;
LABEL_181:
  sub_CB8AB0(a1, v96, v93);
  return sub_CB74F0(a1, (__int64)v23);
}
