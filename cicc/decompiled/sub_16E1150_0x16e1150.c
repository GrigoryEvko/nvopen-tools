// Function: sub_16E1150
// Address: 0x16e1150
//
_QWORD *__fastcall sub_16E1150(_QWORD *a1, char *a2, char *a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  _BYTE *v9; // r9
  bool v10; // dl
  bool v11; // r15
  bool v12; // r12
  bool v13; // r14
  __int64 v14; // r12
  unsigned int v15; // ebx
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // r15
  __int64 *v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v21; // rsi
  bool v22; // al
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  _QWORD *v25; // r15
  __int64 *v26; // rcx
  __int64 v27; // r12
  __int64 v28; // r14
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 *v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // r9
  unsigned __int64 v36; // rcx
  unsigned int v37; // eax
  int v38; // eax
  bool v39; // al
  _BOOL4 v40; // edx
  int v42; // eax
  int v43; // ebx
  int v44; // eax
  __int64 v45; // rdx
  unsigned __int64 v46; // rsi
  __int64 v47; // rdi
  unsigned int v48; // eax
  int v49; // eax
  int v50; // eax
  __int64 *v51; // rdx
  __int64 v52; // rsi
  __int64 v53; // rdi
  _BYTE *v54; // rax
  unsigned int v55; // edx
  __int64 *v56; // rax
  unsigned __int64 v57; // rcx
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  const char *v61; // rdi
  const char *v62; // rsi
  const char **v63; // rax
  const char *v64; // r15
  const char *v65; // rbx
  __int64 v66; // rax
  _BYTE *v67; // rdx
  unsigned int v68; // edi
  const char **v69; // rax
  _BYTE *v70; // rax
  __int64 *v71; // rax
  __int64 *i; // rdx
  __int64 *v73; // rcx
  char *v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdx
  char *v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 *v80; // rax
  __int64 *v81; // rdx
  __int64 *v82; // rax
  __int64 *v83; // rdx
  __int64 *v84; // rax
  __int64 *v85; // rdx
  _BYTE *v86; // rdi
  size_t v87; // rdx
  __int64 v88; // rcx
  unsigned __int64 v89; // rcx
  _BYTE *v90; // [rsp+8h] [rbp-158h]
  int v91; // [rsp+18h] [rbp-148h]
  int v92; // [rsp+24h] [rbp-13Ch]
  unsigned int v93; // [rsp+30h] [rbp-130h]
  __int64 v94; // [rsp+30h] [rbp-130h]
  int v95; // [rsp+38h] [rbp-128h]
  unsigned int v96; // [rsp+3Ch] [rbp-124h]
  char v97; // [rsp+40h] [rbp-120h]
  unsigned __int64 v98; // [rsp+40h] [rbp-120h]
  __int64 v99; // [rsp+40h] [rbp-120h]
  bool v100; // [rsp+48h] [rbp-118h]
  __int64 v101; // [rsp+48h] [rbp-118h]
  unsigned __int64 v102; // [rsp+48h] [rbp-118h]
  __int64 v103; // [rsp+48h] [rbp-118h]
  char *v104[3]; // [rsp+50h] [rbp-110h] BYREF
  _BYTE v105[4]; // [rsp+6Ch] [rbp-F4h] BYREF
  _QWORD v106[2]; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v107[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v108; // [rsp+90h] [rbp-D0h]
  void *dest; // [rsp+A0h] [rbp-C0h]
  size_t v110; // [rsp+A8h] [rbp-B8h]
  _QWORD v111[2]; // [rsp+B0h] [rbp-B0h] BYREF
  void *src; // [rsp+C0h] [rbp-A0h] BYREF
  size_t n; // [rsp+C8h] [rbp-98h]
  _QWORD v114[2]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v115; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v116; // [rsp+E8h] [rbp-78h]
  _BYTE v117[112]; // [rsp+F0h] [rbp-70h] BYREF

  v104[0] = a2;
  v104[1] = a3;
  v115 = (__int64 *)v117;
  v116 = 0x400000000LL;
  sub_16D2880(v104, (__int64)&v115, 45, -1, 1, a6);
  v93 = v116;
  if ( (_DWORD)v116 )
  {
    v42 = sub_16DF2D0(*v115, v115[1]);
    v43 = v116;
    LODWORD(v8) = v42;
    if ( (unsigned int)v116 <= 1 )
    {
      v95 = 0;
      v13 = v42 != 0;
      v10 = 0;
      v11 = 0;
      v12 = 0;
      v96 = 0;
      v92 = 0;
      v100 = 0;
      v97 = 0;
      v93 = 0;
      goto LABEL_3;
    }
    v44 = sub_16DE1A0(v115[2], v115[3]);
    v13 = (_DWORD)v8 != 0;
    v92 = v44;
    v12 = v44 != 0;
    if ( v43 == 2 )
    {
      v95 = 0;
      v10 = 0;
      v11 = 0;
      v96 = 0;
      v100 = 0;
      v97 = 0;
      v93 = 0;
      goto LABEL_3;
    }
    v46 = *(_QWORD *)(v7 + 40);
    v47 = *(_QWORD *)(v7 + 32);
    v103 = v7;
    v48 = sub_16DE880(v47, v46, v45, v7, v8);
    v7 = v103;
    v96 = v48;
    if ( v46 <= 5 )
    {
      v100 = 0;
      if ( v46 != 5 )
      {
        v97 = 0;
LABEL_122:
        LODWORD(v8) = v96;
        v11 = v96 != 0;
        if ( v43 == 3 )
        {
          v95 = 0;
          v10 = 0;
          v93 = 0;
        }
        else
        {
          v94 = v7;
          v95 = sub_16DE390(*(_QWORD *)(v7 + 48), *(_QWORD *)(v7 + 56));
          v10 = v95 != 0;
          if ( v43 == 4 )
            v93 = 0;
          else
            v93 = sub_16DDFE0(*(_QWORD *)(v94 + 64), *(_QWORD *)(v94 + 72));
        }
        goto LABEL_3;
      }
    }
    else
    {
      if ( *(_DWORD *)v47 != 2003269987 || (v49 = 0, *(_WORD *)(v47 + 4) != 28265) )
        v49 = 1;
      v100 = v49 == 0;
    }
    if ( *(_DWORD *)v47 != 1735289197 || (v50 = 0, *(_BYTE *)(v47 + 4) != 119) )
      v50 = 1;
    v97 = v50 == 0;
    goto LABEL_122;
  }
  v100 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v95 = 0;
  v13 = 0;
  v96 = 0;
  v92 = 0;
  v97 = 0;
LABEL_3:
  v105[1] = v12;
  v14 = 0;
  v105[0] = v13;
  v15 = 0;
  v105[2] = v11;
  v105[3] = v10;
  if ( !v13 )
    goto LABEL_6;
LABEL_4:
  if ( ++v14 == 4 )
    goto LABEL_18;
  do
  {
    v15 = v14;
    if ( v105[v14] )
      goto LABEL_4;
LABEL_6:
    if ( !(_DWORD)v116 )
      goto LABEL_4;
    LODWORD(v16) = 0;
    v17 = 0;
    while ( 1 )
    {
      if ( (unsigned int)v16 <= 3 && v105[(unsigned int)v16] )
        goto LABEL_41;
      v18 = 2 * v17;
      v19 = &v115[v18];
      v20 = v115[v18];
      v21 = v115[v18 + 1];
      if ( v14 != 2 )
      {
        if ( v15 == 3 )
        {
          v95 = sub_16DE390(v20, v21);
          if ( v95 )
            goto LABEL_15;
          v93 = sub_16DDFE0(v20, v21);
          v22 = v93 != 0;
        }
        else if ( v15 == 1 )
        {
          v92 = sub_16DE1A0(v20, v21);
          v22 = v92 != 0;
        }
        else
        {
          v22 = (unsigned int)sub_16DF2D0(v20, v21) != 0;
        }
        if ( v22 )
          goto LABEL_15;
        goto LABEL_41;
      }
      v99 = *v19;
      v102 = v19[1];
      v37 = sub_16DE880(v20, v21, *v19, v7, v8);
      v96 = v37;
      if ( v102 > 5 )
      {
        if ( *(_DWORD *)v99 != 2003269987 || (v38 = 0, *(_WORD *)(v99 + 4) != 28265) )
          v38 = 1;
        v39 = v38 == 0;
      }
      else
      {
        if ( v102 != 5 )
        {
          v100 = 0;
          v97 = 0;
          if ( v37 )
            goto LABEL_15;
          goto LABEL_41;
        }
        v39 = 0;
      }
      v40 = *(_DWORD *)v99 != 1735289197 || *(_BYTE *)(v99 + 4) != 119;
      v7 = v96;
      LOBYTE(v7) = v96 != 0;
      v100 = v39 || v96 != 0;
      if ( !v40 )
        break;
      if ( v100 )
      {
        v100 = v39;
        v97 = 0;
        goto LABEL_15;
      }
      v97 = 0;
      v96 = 0;
LABEL_41:
      v17 = (unsigned int)(v16 + 1);
      LODWORD(v16) = v17;
      if ( (_DWORD)v17 == (_DWORD)v116 )
        goto LABEL_4;
    }
    if ( v39 || v96 != 0 )
    {
      v97 = v39 || v96 != 0;
      v100 = v39;
    }
    else
    {
      v96 = 0;
      v97 = 1;
    }
LABEL_15:
    if ( v15 < (unsigned int)v16 )
    {
      v9 = v105;
      v51 = &v115[v18];
      v52 = v115[v18 + 1];
      v53 = v115[v18];
      v51[1] = 0;
      *v51 = (__int64)byte_3F871B3;
      if ( v52 )
      {
        if ( v15 <= 3 )
        {
LABEL_71:
          v54 = &v105[v15];
          do
          {
            v55 = v15++;
            if ( !*v54 )
              goto LABEL_74;
            ++v54;
          }
          while ( v15 != 4 );
          v55 = 4;
          goto LABEL_74;
        }
        while ( 1 )
        {
          v55 = v15;
LABEL_74:
          v15 = v55 + 1;
          v56 = &v115[2 * v55];
          v7 = v56[1];
          v8 = *v56;
          v56[1] = v52;
          *v56 = v53;
          if ( !v7 )
            break;
          v53 = v8;
          v52 = v7;
          if ( v15 <= 3 )
            goto LABEL_71;
        }
      }
    }
    else
    {
      v9 = v105;
      if ( v15 > (unsigned int)v16 )
      {
        LODWORD(v8) = v15;
        do
        {
          v60 = (unsigned int)v16;
          v7 = (unsigned int)v16;
          v61 = byte_3F871B3;
          v62 = 0;
          if ( (unsigned int)v16 < (unsigned int)v116 )
          {
            while ( 1 )
            {
              v63 = (const char **)&v115[2 * v60];
              v64 = v63[1];
              v65 = *v63;
              v63[1] = v62;
              *v63 = v61;
              v62 = v64;
              if ( !v64 )
                break;
              v66 = (unsigned int)(v7 + 1);
              v67 = &v9[v66];
              while ( 1 )
              {
                v7 = (unsigned int)v66;
                if ( (unsigned int)v66 > 3 )
                  break;
                if ( !*v67++ )
                  break;
                LODWORD(v66) = v66 + 1;
              }
              v68 = v116;
              v60 = (unsigned int)v66;
              if ( (unsigned int)v66 >= (unsigned int)v116 )
              {
                if ( (unsigned int)v116 >= HIDWORD(v116) )
                {
                  v90 = v9;
                  v91 = v8;
                  sub_16CD150((__int64)&v115, v117, 0, 16, v8, (int)v9);
                  v68 = v116;
                  v9 = v90;
                  LODWORD(v8) = v91;
                }
                v69 = (const char **)&v115[2 * v68];
                *v69 = v65;
                v69[1] = v64;
                LODWORD(v116) = v116 + 1;
                break;
              }
              v61 = v65;
            }
          }
          v16 = (unsigned int)(v16 + 1);
          v70 = &v9[v16];
          while ( *v70 )
          {
            LODWORD(v16) = v16 + 1;
            ++v70;
            if ( (_DWORD)v16 == 4 )
              goto LABEL_17;
          }
        }
        while ( (unsigned int)v16 < (unsigned int)v8 );
      }
    }
LABEL_17:
    v9[v14++] = 1;
  }
  while ( v14 != 4 );
LABEL_18:
  v110 = 0;
  dest = v111;
  LOBYTE(v111[0]) = 0;
  if ( v95 != 10 )
  {
    if ( v92 == 15 && v95 == 4 )
    {
      v23 = (unsigned __int64)v115;
      v115[6] = (__int64)"gnueabihf";
      *(_QWORD *)(v23 + 56) = 9;
    }
    goto LABEL_22;
  }
  v57 = (unsigned __int64)v115;
  v58 = v115[7];
  if ( v58 <= 0xA )
    goto LABEL_22;
  v59 = v115[6];
  if ( *(_QWORD *)v59 != 0x6564696F72646E61LL || *(_WORD *)(v59 + 8) != 25185 || *(_BYTE *)(v59 + 10) != 105 )
    goto LABEL_22;
  v106[0] = v59 + 11;
  v106[1] = v58 - 11;
  if ( v58 == 11 )
  {
    v115[7] = 7;
    *(_QWORD *)(v57 + 48) = "android";
    goto LABEL_22;
  }
  v107[0] = "android";
  v108 = 1283;
  v107[1] = v106;
  sub_16E2FC0(&src, v107);
  v86 = dest;
  v87 = n;
  if ( src == v114 )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = v114[0];
      else
        memcpy(dest, src, n);
      v87 = n;
      v86 = dest;
    }
    v110 = v87;
    v86[v87] = 0;
    v86 = src;
    goto LABEL_181;
  }
  if ( dest == v111 )
  {
    dest = src;
    v110 = n;
    v111[0] = v114[0];
    goto LABEL_185;
  }
  v88 = v111[0];
  dest = src;
  v110 = n;
  v111[0] = v114[0];
  if ( !v86 )
  {
LABEL_185:
    src = v114;
    v86 = v114;
    goto LABEL_181;
  }
  src = v86;
  v114[0] = v88;
LABEL_181:
  n = 0;
  *v86 = 0;
  if ( src != v114 )
    j_j___libc_free_0(src, v114[0] + 1LL);
  v89 = (unsigned __int64)v115;
  v115[6] = (__int64)dest;
  *(_QWORD *)(v89 + 56) = v110;
LABEL_22:
  v24 = (unsigned int)v116;
  if ( v96 != 15 )
  {
    if ( v97 )
    {
      if ( (unsigned int)v116 > 4uLL )
      {
        LODWORD(v116) = 4;
        v26 = v115;
      }
      else if ( (unsigned int)v116 == 4 )
      {
        v26 = v115;
      }
      else
      {
        if ( HIDWORD(v116) <= 3 )
        {
          sub_16CD150((__int64)&v115, v117, 4u, 16, v8, (int)v9);
          v24 = (unsigned int)v116;
        }
        v26 = v115;
        v82 = &v115[2 * v24];
        v83 = v115 + 8;
        if ( v82 != v115 + 8 )
        {
          do
          {
            if ( v82 )
            {
              *v82 = 0;
              v82[1] = 0;
            }
            v82 += 2;
          }
          while ( v83 != v82 );
          v26 = v115;
        }
        LODWORD(v116) = 4;
      }
      v26[5] = 7;
      v26[4] = (__int64)"windows";
      v26[6] = (__int64)"gnu";
      v26[7] = 3;
      if ( v93 <= 1 )
        goto LABEL_128;
    }
    else
    {
      if ( !v100 )
        goto LABEL_25;
      if ( (unsigned int)v116 <= 4uLL )
      {
        if ( (unsigned int)v116 == 4 )
        {
          v26 = v115;
        }
        else
        {
          if ( HIDWORD(v116) <= 3 )
          {
            sub_16CD150((__int64)&v115, v117, 4u, 16, v8, (int)v9);
            v24 = (unsigned int)v116;
          }
          v26 = v115;
          v80 = &v115[2 * v24];
          v81 = v115 + 8;
          if ( v80 != v115 + 8 )
          {
            do
            {
              if ( v80 )
              {
                *v80 = 0;
                v80[1] = 0;
              }
              v80 += 2;
            }
            while ( v81 != v80 );
            v26 = v115;
          }
          LODWORD(v116) = 4;
        }
      }
      else
      {
        LODWORD(v116) = 4;
        v26 = v115;
      }
      v26[5] = 7;
      v26[4] = (__int64)"windows";
      v26[6] = (__int64)"cygnus";
      v26[7] = 6;
      if ( v93 <= 1 )
      {
LABEL_128:
        v25 = a1 + 2;
        a1[1] = 0;
        LODWORD(v8) = 4;
        *a1 = a1 + 2;
        *((_BYTE *)a1 + 16) = 0;
        goto LABEL_28;
      }
    }
LABEL_105:
    if ( HIDWORD(v116) <= 4 )
    {
      sub_16CD150((__int64)&v115, v117, 5u, 16, v8, (int)v9);
      v14 = (unsigned int)v116;
    }
    v71 = &v115[2 * v14];
    for ( i = v115 + 10; i != v71; v71 += 2 )
    {
      if ( v71 )
      {
        *v71 = 0;
        v71[1] = 0;
      }
    }
    LODWORD(v116) = 5;
    goto LABEL_140;
  }
  if ( (unsigned int)v116 > 4uLL )
  {
    LODWORD(v116) = 4;
    v73 = v115;
  }
  else if ( (unsigned int)v116 == 4 )
  {
    v73 = v115;
  }
  else
  {
    if ( HIDWORD(v116) <= 3 )
    {
      sub_16CD150((__int64)&v115, v117, 4u, 16, v8, (int)v9);
      v24 = (unsigned int)v116;
    }
    v73 = v115;
    v84 = &v115[2 * v24];
    v85 = v115 + 8;
    if ( v84 != v115 + 8 )
    {
      do
      {
        if ( v84 )
        {
          *v84 = 0;
          v84[1] = 0;
        }
        v84 += 2;
      }
      while ( v85 != v84 );
      v73 = v115;
    }
    LODWORD(v116) = 4;
  }
  v73[5] = 7;
  v73[4] = (__int64)"windows";
  if ( v95 )
  {
    LODWORD(v8) = v116;
    if ( v93 > 1 )
    {
LABEL_138:
      v14 = (unsigned int)v8;
      if ( (unsigned int)v8 <= 5uLL )
      {
        if ( (unsigned int)v8 != 5 )
          goto LABEL_105;
      }
      else
      {
        LODWORD(v116) = 5;
      }
LABEL_140:
      v77 = sub_16DDE20(v93);
      *(_QWORD *)(v78 + 64) = v77;
      *(_QWORD *)(v78 + 72) = v79;
      LODWORD(v8) = v116;
    }
  }
  else if ( v93 <= 1 )
  {
    v73[7] = 4;
    LODWORD(v8) = v116;
    v73[6] = (__int64)"msvc";
  }
  else
  {
    v74 = sub_16DDE20(v93);
    *(_QWORD *)(v75 + 48) = v74;
    *(_QWORD *)(v75 + 56) = v76;
    if ( v97 || (LODWORD(v24) = v116, v100) )
    {
      LODWORD(v8) = v116;
      goto LABEL_138;
    }
LABEL_25:
    LODWORD(v8) = v24;
  }
  v25 = a1 + 2;
  a1[1] = 0;
  *a1 = a1 + 2;
  *((_BYTE *)a1 + 16) = 0;
  if ( (_DWORD)v8 )
  {
    v26 = v115;
LABEL_28:
    v27 = (unsigned int)(v8 - 1);
    v28 = 0;
    v29 = 0;
    while ( 1 )
    {
      v31 = &v26[2 * v28];
      v32 = v31[1];
      if ( v32 > 0x3FFFFFFFFFFFFFFFLL - v29 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(a1, *v31, v32, v26);
      if ( v27 == v28 )
        break;
      if ( (_DWORD)v28 != -1 )
      {
        v33 = a1[1];
        v34 = *a1;
        v35 = v33 + 1;
        if ( (_QWORD *)*a1 == v25 )
          v36 = 15;
        else
          v36 = a1[2];
        if ( v35 > v36 )
        {
          v98 = v33 + 1;
          v101 = a1[1];
          sub_2240BB0(a1, v101, 0, 0, 1);
          v34 = *a1;
          v35 = v98;
          v33 = v101;
        }
        *(_BYTE *)(v34 + v33) = 45;
        v30 = *a1;
        a1[1] = v35;
        *(_BYTE *)(v30 + v33 + 1) = 0;
      }
      v29 = a1[1];
      v26 = v115;
      ++v28;
    }
  }
  if ( dest != v111 )
    j_j___libc_free_0(dest, v111[0] + 1LL);
  if ( v115 != (__int64 *)v117 )
    _libc_free((unsigned __int64)v115);
  return a1;
}
