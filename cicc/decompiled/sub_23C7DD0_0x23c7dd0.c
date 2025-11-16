// Function: sub_23C7DD0
// Address: 0x23c7dd0
//
__int64 *__fastcall sub_23C7DD0(__int64 *a1, __int64 a2, _BYTE *a3, size_t a4, __int32 a5, char a6)
{
  const char *v7; // rax
  __int64 (__fastcall **v8)(); // r13
  __int64 v13; // rax
  _QWORD *v14; // rdx
  int v15; // eax
  unsigned int v16; // r8d
  __int64 *v17; // rcx
  unsigned __int64 i; // rdx
  char *v19; // rax
  char *v20; // r15
  _BYTE *v21; // rdi
  __int64 v22; // rsi
  __int64 (__fastcall **v23)(); // r15
  __int64 v24; // r13
  __int64 *v25; // rax
  __int64 v26; // rbx
  __int64 *v27; // rax
  size_t v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  char v31; // dl
  __int64 v32; // rax
  _QWORD *v33; // r15
  __int64 v34; // rsi
  size_t v35; // rdx
  _QWORD *v36; // rdi
  unsigned __int64 *v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // rcx
  unsigned __int64 v40; // rdx
  char *v41; // rdx
  char *v42; // r13
  __int64 v43; // r15
  char *v44; // rax
  __int64 v45; // r14
  __int64 v46; // r8
  __int64 v47; // rdx
  unsigned __int64 v48; // r13
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // r15
  __int64 v51; // r14
  unsigned __int64 v52; // rdi
  char v53; // al
  unsigned __int64 v54; // r13
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // r14
  __int64 v57; // r15
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // r14
  __int64 v61; // r15
  unsigned __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // r13
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // r14
  __int64 v67; // r15
  unsigned __int64 v68; // rdi
  char *v69; // r13
  unsigned __int64 v70; // rdi
  unsigned __int64 v71; // r14
  __int64 v72; // r15
  unsigned __int64 v73; // rdi
  unsigned __int64 *v74; // r13
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // r15
  __int64 v77; // r14
  unsigned __int64 v78; // rdi
  unsigned __int64 *v79; // r15
  unsigned __int64 v80; // rax
  unsigned __int64 v81; // r13
  unsigned __int64 v82; // rdi
  unsigned __int64 *v83; // r14
  unsigned __int64 *v84; // rax
  unsigned __int64 v85; // rdi
  unsigned __int64 v86; // rdi
  unsigned __int64 *v87; // r13
  unsigned __int64 *v88; // r14
  unsigned __int64 *v89; // rdx
  unsigned __int64 v90; // r14
  __int64 v91; // r15
  unsigned __int64 v92; // rdi
  unsigned __int64 v93; // rdi
  unsigned __int64 v94; // rdi
  __int64 v95; // [rsp+0h] [rbp-130h]
  unsigned __int64 v96; // [rsp+10h] [rbp-120h]
  char *v97; // [rsp+10h] [rbp-120h]
  __int64 *v98; // [rsp+18h] [rbp-118h]
  __int64 v99; // [rsp+18h] [rbp-118h]
  unsigned __int64 v100; // [rsp+18h] [rbp-118h]
  unsigned __int64 v101; // [rsp+18h] [rbp-118h]
  char *v102; // [rsp+18h] [rbp-118h]
  unsigned int v103; // [rsp+20h] [rbp-110h]
  unsigned __int64 *v104; // [rsp+20h] [rbp-110h]
  char *v105; // [rsp+20h] [rbp-110h]
  char *v106; // [rsp+20h] [rbp-110h]
  char *v107; // [rsp+20h] [rbp-110h]
  char *v108; // [rsp+20h] [rbp-110h]
  char *v109; // [rsp+20h] [rbp-110h]
  int v110; // [rsp+28h] [rbp-108h]
  __int64 v112[2]; // [rsp+30h] [rbp-100h] BYREF
  void *dest; // [rsp+40h] [rbp-F0h] BYREF
  size_t v114; // [rsp+48h] [rbp-E8h]
  _QWORD v115[2]; // [rsp+50h] [rbp-E0h] BYREF
  unsigned __int64 v116; // [rsp+60h] [rbp-D0h] BYREF
  size_t n; // [rsp+68h] [rbp-C8h]
  _QWORD src[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v119[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD v120[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int16 v121; // [rsp+A0h] [rbp-90h]
  __m128i v122; // [rsp+B0h] [rbp-80h] BYREF
  char *v123; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v124; // [rsp+C8h] [rbp-68h]
  _WORD v125[20]; // [rsp+D0h] [rbp-60h] BYREF
  char v126; // [rsp+F8h] [rbp-38h]

  if ( !a4 )
  {
    v7 = "glob";
    if ( !a6 )
      v7 = "regex";
    v121 = 771;
    v125[0] = 770;
    v120[0] = v7;
    v122 = (__m128i)(unsigned __int64)v119;
    v119[0] = (__int64)"Supplied ";
    v123 = " was blank";
    v8 = sub_2241E50();
    sub_CA0F50((__int64 *)&v116, (void **)&v122);
    sub_C63F00(a1, (__int64)&v116, 0x16u, (__int64)v8);
    if ( (_QWORD *)v116 != src )
      j_j___libc_free_0(v116);
    return a1;
  }
  if ( a6 )
  {
    v15 = sub_C92610();
    v16 = sub_C92740(a2, a3, a4, v15);
    v17 = (__int64 *)(*(_QWORD *)a2 + 8LL * v16);
    if ( *v17 )
    {
      if ( *v17 != -8 )
      {
LABEL_15:
        *a1 = 1;
        return a1;
      }
      --*(_DWORD *)(a2 + 16);
    }
    v98 = v17;
    v103 = v16;
    v24 = sub_C7D670(a4 + 89, 8);
    memcpy((void *)(v24 + 88), a3, a4);
    *(_BYTE *)(v24 + a4 + 88) = 0;
    *(_QWORD *)(v24 + 24) = v24 + 40;
    *(_QWORD *)v24 = a4;
    *(_QWORD *)(v24 + 72) = 0;
    *(_QWORD *)(v24 + 32) = 0x100000000LL;
    *(_DWORD *)(v24 + 80) = 0;
    *(_OWORD *)(v24 + 8) = 0;
    *(_OWORD *)(v24 + 40) = 0;
    *(_OWORD *)(v24 + 56) = 0;
    *v98 = v24;
    ++*(_DWORD *)(a2 + 12);
    v25 = (__int64 *)(*(_QWORD *)a2 + 8LL * (unsigned int)sub_C929D0((__int64 *)a2, v103));
    v26 = *v25;
    if ( !*v25 || v26 == -8 )
    {
      v27 = v25 + 1;
      do
      {
        do
          v26 = *v27++;
        while ( !v26 );
      }
      while ( v26 == -8 );
    }
    v28 = *(_QWORD *)v26;
    v119[0] = 1024;
    v119[1] = 1;
    sub_109B500(&v122, v26 + 88, v28, 0x400u, 1);
    v31 = v126 & 1;
    v126 = (2 * (v126 & 1)) | v126 & 0xFD;
    if ( v31 )
      goto LABEL_41;
    *(__m128i *)(v26 + 8) = _mm_loadu_si128(&v122);
    if ( (char **)(v26 + 24) != &v123 )
    {
      v37 = *(unsigned __int64 **)(v26 + 24);
      v38 = *(unsigned int *)(v26 + 32);
      v99 = (__int64)v37;
      v104 = v37;
      if ( v123 == (char *)v125 )
      {
        v39 = (unsigned int)v124;
        v110 = v124;
        v96 = (unsigned int)v124;
        if ( v38 >= (unsigned int)v124 )
        {
          v63 = *(_QWORD *)(v26 + 24);
          if ( (_DWORD)v124 )
          {
            v87 = (unsigned __int64 *)v125;
            v102 = (char *)&v37[5 * (unsigned int)v124];
            do
            {
              v88 = v104 + 5;
              v89 = v104 + 5;
              if ( v104 != v87 )
              {
                v90 = *v104;
                v91 = *v104 + 80LL * *((unsigned int *)v104 + 2);
                if ( *((_DWORD *)v87 + 2) )
                {
                  if ( v90 != v91 )
                  {
                    do
                    {
                      v91 -= 80;
                      v93 = *(_QWORD *)(v91 + 8);
                      if ( v93 != v91 + 24 )
                        _libc_free(v93);
                    }
                    while ( v91 != v90 );
                    v90 = *v104;
                  }
                  if ( (unsigned __int64 *)v90 != v104 + 2 )
                    _libc_free(v90);
                  *v104 = *v87;
                  *((_DWORD *)v104 + 2) = *((_DWORD *)v87 + 2);
                  *((_DWORD *)v104 + 3) = *((_DWORD *)v87 + 3);
                  *v87 = (unsigned __int64)(v87 + 2);
                  *((_DWORD *)v87 + 3) = 0;
                  *((_DWORD *)v87 + 2) = 0;
                }
                else
                {
                  while ( v91 != v90 )
                  {
                    v91 -= 80;
                    v92 = *(_QWORD *)(v91 + 8);
                    if ( v92 != v91 + 24 )
                      _libc_free(v92);
                  }
                  *((_DWORD *)v104 + 2) = 0;
                }
                if ( v87[3] )
                {
                  v94 = v104[2];
                  v88 = v104 + 5;
                  if ( (unsigned __int64 *)v94 != v104 + 5 )
                    _libc_free(v94);
                  v104[2] = v87[2];
                  v104[3] = v87[3];
                  v104[4] = v87[4];
                  v89 = v87 + 5;
                  v87[2] = (unsigned __int64)(v87 + 5);
                  v87[4] = 0;
                  v87[3] = 0;
                }
                else
                {
                  v89 = v87 + 5;
                  v104[3] = 0;
                  v88 = v104 + 5;
                }
              }
              v104 = v88;
              v87 = v89;
            }
            while ( v88 != (unsigned __int64 *)v102 );
            v63 = *(_QWORD *)(v26 + 24);
            v38 = *(unsigned int *)(v26 + 32);
          }
          v64 = v63 + 40 * v38;
          if ( (unsigned __int64 *)v64 != v104 )
          {
            do
            {
              v64 -= 40;
              v65 = *(_QWORD *)(v64 + 16);
              if ( v65 != v64 + 40 )
                _libc_free(v65);
              v66 = *(_QWORD *)v64;
              v67 = *(_QWORD *)v64 + 80LL * *(unsigned int *)(v64 + 8);
              if ( *(_QWORD *)v64 != v67 )
              {
                do
                {
                  v67 -= 80;
                  v68 = *(_QWORD *)(v67 + 8);
                  if ( v68 != v67 + 24 )
                    _libc_free(v68);
                }
                while ( v66 != v67 );
                v66 = *(_QWORD *)v64;
              }
              if ( v66 != v64 + 16 )
                _libc_free(v66);
            }
            while ( v104 != (unsigned __int64 *)v64 );
          }
          *(_DWORD *)(v26 + 32) = v110;
          v69 = &v123[40 * (unsigned int)v124];
          v109 = v123;
          while ( v109 != v69 )
          {
            v69 -= 40;
            v70 = *((_QWORD *)v69 + 2);
            if ( (char *)v70 != v69 + 40 )
              _libc_free(v70);
            v71 = *(_QWORD *)v69;
            v72 = *(_QWORD *)v69 + 80LL * *((unsigned int *)v69 + 2);
            if ( *(_QWORD *)v69 != v72 )
            {
              do
              {
                v72 -= 80;
                v73 = *(_QWORD *)(v72 + 8);
                if ( v73 != v72 + 24 )
                  _libc_free(v73);
              }
              while ( v71 != v72 );
              v71 = *(_QWORD *)v69;
            }
            if ( (char *)v71 != v69 + 16 )
              _libc_free(v71);
          }
          goto LABEL_145;
        }
        v40 = *(unsigned int *)(v26 + 36);
        if ( (unsigned int)v124 > v40 )
        {
          v74 = &v37[5 * v38];
          while ( v37 != v74 )
          {
            v74 -= 5;
            v75 = v74[2];
            if ( (unsigned __int64 *)v75 != v74 + 5 )
              _libc_free(v75);
            v76 = *v74;
            v77 = *v74 + 80LL * *((unsigned int *)v74 + 2);
            if ( *v74 != v77 )
            {
              do
              {
                v77 -= 80;
                v78 = *(_QWORD *)(v77 + 8);
                if ( v78 != v77 + 24 )
                  _libc_free(v78);
              }
              while ( v76 != v77 );
              v76 = *v74;
            }
            if ( (unsigned __int64 *)v76 != v74 + 2 )
              _libc_free(v76);
          }
          *(_DWORD *)(v26 + 32) = 0;
          sub_F31630(v26 + 24, v96, v40, v39, v29, v30);
          v41 = v123;
          v96 = (unsigned int)v124;
          v42 = v123;
          v99 = *(_QWORD *)(v26 + 24);
        }
        else
        {
          v41 = (char *)v125;
          v42 = (char *)v125;
          if ( *(_DWORD *)(v26 + 32) )
          {
            v79 = (unsigned __int64 *)v125;
            v95 = 40 * v38;
            v97 = (char *)&v37[5 * v38];
            do
            {
              if ( v104 == v79 )
              {
                v83 = v104 + 5;
                v84 = v104 + 5;
              }
              else
              {
                v80 = *v104;
                v81 = *v104 + 80LL * *((unsigned int *)v104 + 2);
                if ( *((_DWORD *)v79 + 2) )
                {
                  if ( v80 != v81 )
                  {
                    do
                    {
                      v81 -= 80LL;
                      v85 = *(_QWORD *)(v81 + 8);
                      if ( v85 != v81 + 24 )
                      {
                        v101 = v80;
                        _libc_free(v85);
                        v80 = v101;
                      }
                    }
                    while ( v81 != v80 );
                    v81 = *v104;
                  }
                  if ( (unsigned __int64 *)v81 != v104 + 2 )
                    _libc_free(v81);
                  *v104 = *v79;
                  *((_DWORD *)v104 + 2) = *((_DWORD *)v79 + 2);
                  *((_DWORD *)v104 + 3) = *((_DWORD *)v79 + 3);
                  *v79 = (unsigned __int64)(v79 + 2);
                  *((_DWORD *)v79 + 3) = 0;
                  *((_DWORD *)v79 + 2) = 0;
                }
                else
                {
                  while ( v81 != v80 )
                  {
                    v81 -= 80LL;
                    v82 = *(_QWORD *)(v81 + 8);
                    if ( v82 != v81 + 24 )
                    {
                      v100 = v80;
                      _libc_free(v82);
                      v80 = v100;
                    }
                  }
                  *((_DWORD *)v104 + 2) = 0;
                }
                if ( v79[3] )
                {
                  v86 = v104[2];
                  v83 = v104 + 5;
                  if ( (unsigned __int64 *)v86 != v104 + 5 )
                    _libc_free(v86);
                  v104[2] = v79[2];
                  v104[3] = v79[3];
                  v104[4] = v79[4];
                  v84 = v79 + 5;
                  v79[2] = (unsigned __int64)(v79 + 5);
                  v79[4] = 0;
                  v79[3] = 0;
                }
                else
                {
                  v104[3] = 0;
                  v83 = v104 + 5;
                  v84 = v79 + 5;
                }
              }
              v104 = v83;
              v79 = v84;
            }
            while ( v97 != (char *)v83 );
            v39 = (__int64)v123;
            v108 = v123;
            v42 = &v123[v95];
            v43 = *(_QWORD *)(v26 + 24) + v95;
            v44 = &v123[40 * (unsigned int)v124];
            if ( v44 == &v123[v95] )
              goto LABEL_111;
            goto LABEL_71;
          }
        }
        v43 = v99;
        v44 = &v41[40 * v96];
        if ( v44 == v42 )
        {
          *(_DWORD *)(v26 + 32) = v110;
LABEL_145:
          LODWORD(v124) = 0;
          v53 = v126 & 1;
LABEL_94:
          v126 &= ~2u;
          if ( !v53 )
          {
            v116 = 1;
            v106 = v123;
            v54 = (unsigned __int64)&v123[40 * (unsigned int)v124];
            if ( v123 != (char *)v54 )
            {
              do
              {
                v54 -= 40LL;
                v55 = *(_QWORD *)(v54 + 16);
                if ( v55 != v54 + 40 )
                  _libc_free(v55);
                v56 = *(_QWORD *)v54;
                v57 = *(_QWORD *)v54 + 80LL * *(unsigned int *)(v54 + 8);
                if ( *(_QWORD *)v54 != v57 )
                {
                  do
                  {
                    v57 -= 80;
                    v58 = *(_QWORD *)(v57 + 8);
                    if ( v58 != v57 + 24 )
                      _libc_free(v58);
                  }
                  while ( v56 != v57 );
                  v56 = *(_QWORD *)v54;
                }
                if ( v56 != v54 + 16 )
                  _libc_free(v56);
              }
              while ( v106 != (char *)v54 );
              v54 = (unsigned __int64)v123;
            }
            if ( (_WORD *)v54 != v125 )
              _libc_free(v54);
            goto LABEL_42;
          }
LABEL_41:
          v116 = v122.m128i_i64[0] | 1;
LABEL_42:
          if ( (v116 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            *a1 = v116 & 0xFFFFFFFFFFFFFFFELL | 1;
            return a1;
          }
          *(_DWORD *)(v26 + 80) = a5;
          goto LABEL_15;
        }
        do
        {
LABEL_71:
          v45 = 40;
          if ( v43 )
          {
            v46 = v43 + 16;
            *(_DWORD *)(v43 + 8) = 0;
            *(_QWORD *)v43 = v43 + 16;
            *(_DWORD *)(v43 + 12) = 0;
            v47 = *((unsigned int *)v42 + 2);
            if ( (_DWORD)v47 )
            {
              v105 = v44;
              sub_23C7A40(v43, (__int64)v42, v47, v39, v46, v30);
              v46 = v43 + 16;
              v44 = v105;
            }
            v45 = v43 + 40;
            *(_QWORD *)(v43 + 24) = 0;
            *(_QWORD *)(v43 + 16) = v43 + 40;
            *(_QWORD *)(v43 + 32) = 0;
            if ( *((_QWORD *)v42 + 3) )
            {
              v107 = v44;
              sub_23C6BC0(v46, (char **)v42 + 2, v47, v39, v46, v30);
              v44 = v107;
            }
          }
          v42 += 40;
          v43 = v45;
        }
        while ( v44 != v42 );
        v108 = v123;
        v42 = &v123[40 * (unsigned int)v124];
LABEL_111:
        *(_DWORD *)(v26 + 32) = v110;
        while ( v42 != v108 )
        {
          v42 -= 40;
          v59 = *((_QWORD *)v42 + 2);
          if ( (char *)v59 != v42 + 40 )
            _libc_free(v59);
          v60 = *(_QWORD *)v42;
          v61 = *(_QWORD *)v42 + 80LL * *((unsigned int *)v42 + 2);
          if ( *(_QWORD *)v42 != v61 )
          {
            do
            {
              v61 -= 80;
              v62 = *(_QWORD *)(v61 + 8);
              if ( v62 != v61 + 24 )
                _libc_free(v62);
            }
            while ( v60 != v61 );
            v60 = *(_QWORD *)v42;
          }
          if ( (char *)v60 != v42 + 16 )
            _libc_free(v60);
        }
        goto LABEL_145;
      }
      v48 = (unsigned __int64)&v37[5 * v38];
      if ( v37 != (unsigned __int64 *)v48 )
      {
        do
        {
          v48 -= 40LL;
          v49 = *(_QWORD *)(v48 + 16);
          if ( v49 != v48 + 40 )
            _libc_free(v49);
          v50 = *(_QWORD *)v48;
          v51 = *(_QWORD *)v48 + 80LL * *(unsigned int *)(v48 + 8);
          if ( *(_QWORD *)v48 != v51 )
          {
            do
            {
              v51 -= 80;
              v52 = *(_QWORD *)(v51 + 8);
              if ( v52 != v51 + 24 )
                _libc_free(v52);
            }
            while ( v50 != v51 );
            v50 = *(_QWORD *)v48;
          }
          if ( v50 != v48 + 16 )
            _libc_free(v50);
        }
        while ( v37 != (unsigned __int64 *)v48 );
        v48 = *(_QWORD *)(v26 + 24);
      }
      if ( v48 != v26 + 40 )
        _libc_free(v48);
      *(_QWORD *)(v26 + 24) = v123;
      *(_QWORD *)(v26 + 32) = v124;
      v124 = 0;
      v123 = (char *)v125;
    }
    v53 = v126 & 1;
    goto LABEL_94;
  }
  if ( !a3 )
  {
    LOBYTE(v115[0]) = 0;
    dest = v115;
    v114 = 0;
    goto LABEL_17;
  }
  v13 = a4;
  v122.m128i_i64[0] = a4;
  dest = v115;
  if ( a4 > 0xF )
  {
    dest = (void *)sub_22409D0((__int64)&dest, (unsigned __int64 *)&v122, 0);
    v36 = dest;
    v115[0] = v122.m128i_i64[0];
    goto LABEL_59;
  }
  if ( a4 != 1 )
  {
    v36 = v115;
LABEL_59:
    memcpy(v36, a3, a4);
    v13 = v122.m128i_i64[0];
    v14 = dest;
    goto LABEL_12;
  }
  LOBYTE(v115[0]) = *a3;
  v14 = v115;
LABEL_12:
  v114 = v13;
  *((_BYTE *)v14 + v13) = 0;
LABEL_17:
  for ( i = 0; ; i = (unsigned __int64)(v20 + 2) )
  {
    v19 = sub_22417D0((__int64 *)&dest, 42, i);
    v20 = v19;
    if ( v19 == (char *)-1LL )
      break;
    if ( (unsigned __int64)v19 > v114 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::replace", (size_t)v19, v114);
    sub_2241130((unsigned __int64 *)&dest, (size_t)v19, v114 != (_QWORD)v19, ".*", 2u);
  }
  v119[0] = (__int64)"^(";
  v121 = 1283;
  v120[0] = dest;
  v125[0] = 770;
  v120[1] = v114;
  v122.m128i_i64[0] = (__int64)v119;
  v123 = ")$";
  sub_CA0F50((__int64 *)&v116, (void **)&v122);
  v21 = dest;
  if ( (_QWORD *)v116 == src )
  {
    v35 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v35 = n;
      v21 = dest;
    }
    v114 = v35;
    v21[v35] = 0;
    v21 = (_BYTE *)v116;
  }
  else
  {
    if ( dest == v115 )
    {
      dest = (void *)v116;
      v114 = n;
      v115[0] = src[0];
    }
    else
    {
      v22 = v115[0];
      dest = (void *)v116;
      v114 = n;
      v115[0] = src[0];
      if ( v21 )
      {
        v116 = (unsigned __int64)v21;
        src[0] = v22;
        goto LABEL_25;
      }
    }
    v116 = (unsigned __int64)src;
    v21 = src;
  }
LABEL_25:
  n = 0;
  *v21 = 0;
  if ( (_QWORD *)v116 != src )
    j_j___libc_free_0(v116);
  sub_C88F40((__int64)v112, (__int64)dest, v114, 0);
  v116 = (unsigned __int64)src;
  n = 0;
  LOBYTE(src[0]) = 0;
  if ( !(unsigned __int8)sub_C89030(v112, &v116) )
  {
    v122.m128i_i64[0] = (__int64)&v116;
    v125[0] = 260;
    v23 = sub_2241E50();
    sub_CA0F50(v119, (void **)&v122);
    sub_C63F00(a1, (__int64)v119, 0x16u, (__int64)v23);
    if ( (_QWORD *)v119[0] != v120 )
      j_j___libc_free_0(v119[0]);
    goto LABEL_30;
  }
  v32 = sub_22077B0(0x10u);
  v33 = (_QWORD *)v32;
  if ( v32 )
    sub_C88FD0(v32, v112);
  v34 = *(_QWORD *)(a2 + 32);
  v122.m128i_i64[0] = (__int64)v33;
  v122.m128i_i32[2] = a5;
  if ( v34 == *(_QWORD *)(a2 + 40) )
  {
    sub_23C7710((unsigned __int64 *)(a2 + 24), (char *)v34, v122.m128i_i64);
    v33 = (_QWORD *)v122.m128i_i64[0];
LABEL_75:
    if ( v33 )
    {
      sub_C88FF0(v33);
      j_j___libc_free_0((unsigned __int64)v33);
    }
    goto LABEL_49;
  }
  if ( !v34 )
  {
    *(_QWORD *)(a2 + 32) = 16;
    goto LABEL_75;
  }
  *(_QWORD *)v34 = v33;
  *(_DWORD *)(v34 + 8) = v122.m128i_i32[2];
  *(_QWORD *)(a2 + 32) += 16LL;
LABEL_49:
  *a1 = 1;
LABEL_30:
  if ( (_QWORD *)v116 != src )
    j_j___libc_free_0(v116);
  sub_C88FF0(v112);
  if ( dest != v115 )
    j_j___libc_free_0((unsigned __int64)dest);
  return a1;
}
