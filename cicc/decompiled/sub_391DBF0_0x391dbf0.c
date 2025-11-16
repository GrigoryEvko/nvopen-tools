// Function: sub_391DBF0
// Address: 0x391dbf0
//
void __fastcall sub_391DBF0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  int v4; // r8d
  int v5; // r9d
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  int v10; // r14d
  _QWORD *v11; // rbx
  unsigned int v12; // eax
  __int64 v13; // rdx
  _QWORD *v14; // r13
  unsigned __int64 v15; // rdi
  int v16; // r13d
  __int64 v17; // rbx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r15
  unsigned int v21; // eax
  char *v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rsi
  _DWORD *v28; // rax
  _DWORD *v29; // rsi
  __int64 v30; // rsi
  _DWORD *v31; // rax
  _DWORD *v32; // rsi
  unsigned __int64 v33; // rdi
  __int64 v34; // r13
  __int64 v35; // rbx
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  __int64 v38; // rbx
  __int64 v39; // r13
  __int64 v40; // rbx
  unsigned __int64 v41; // rdi
  int v42; // eax
  __int64 v43; // rdx
  _QWORD *v44; // rax
  _QWORD *m; // rdx
  unsigned int v46; // ecx
  _QWORD *v47; // rdi
  unsigned int v48; // eax
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rax
  int v52; // ebx
  unsigned __int64 v53; // r13
  _QWORD *v54; // rax
  __int64 v55; // rdx
  _QWORD *n; // rdx
  unsigned int v57; // ecx
  _QWORD *v58; // rdi
  unsigned int v59; // eax
  int v60; // eax
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  int v63; // ebx
  unsigned __int64 v64; // r13
  _QWORD *v65; // rax
  __int64 v66; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v68; // rdi
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // rdi
  __int64 v71; // rdx
  int v72; // ebx
  unsigned int v73; // r13d
  unsigned int v74; // eax
  __int64 v75; // r13
  unsigned __int64 v76; // rdx
  unsigned __int64 v77; // rax
  __int64 v78; // rbx
  int v79; // r8d
  int v80; // r9d
  __int64 v81; // rax
  __int64 v82; // r13
  __int64 v83; // rdx
  int v84; // edx
  __int64 v85; // rcx
  int v86; // edx
  int v87; // ebx
  unsigned int v88; // r14d
  unsigned int v89; // eax
  _QWORD *v90; // rdi
  unsigned __int64 v91; // rax
  unsigned __int64 v92; // rdi
  _QWORD *v93; // rax
  __int64 v94; // rdx
  _QWORD *k; // rdx
  _QWORD *v96; // rax
  _QWORD *v97; // rax
  _QWORD *v98; // rax
  __int64 v99; // rbx
  int v100; // eax
  char *v101; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v102; // [rsp+20h] [rbp-A0h]
  char v103; // [rsp+28h] [rbp-98h] BYREF
  char *v104; // [rsp+30h] [rbp-90h] BYREF
  __int64 v105; // [rsp+38h] [rbp-88h]
  _BYTE v106[16]; // [rsp+40h] [rbp-80h] BYREF
  int v107; // [rsp+50h] [rbp-70h]
  char *v108; // [rsp+58h] [rbp-68h] BYREF
  __int64 v109; // [rsp+60h] [rbp-60h]
  char v110; // [rsp+68h] [rbp-58h] BYREF
  _BYTE *v111; // [rsp+70h] [rbp-50h] BYREF
  __int64 v112; // [rsp+78h] [rbp-48h]
  _BYTE v113[64]; // [rsp+80h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 != *(_QWORD *)(a1 + 40) )
    *(_QWORD *)(a1 + 40) = v2;
  v3 = *(_QWORD *)(a1 + 64);
  if ( v3 != *(_QWORD *)(a1 + 72) )
    *(_QWORD *)(a1 + 72) = v3;
  sub_391DA40(a1 + 96);
  sub_391DA40(a1 + 160);
  sub_391DA40(a1 + 128);
  v6 = *(_DWORD *)(a1 + 208);
  ++*(_QWORD *)(a1 + 192);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 212) )
      goto LABEL_11;
    v7 = *(unsigned int *)(a1 + 216);
    if ( (unsigned int)v7 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 200));
      *(_QWORD *)(a1 + 200) = 0;
      *(_QWORD *)(a1 + 208) = 0;
      *(_DWORD *)(a1 + 216) = 0;
      goto LABEL_11;
    }
    goto LABEL_8;
  }
  v57 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 216);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v57 = 64;
  if ( (unsigned int)v7 <= v57 )
  {
LABEL_8:
    v8 = *(_QWORD **)(a1 + 200);
    for ( i = &v8[3 * v7]; i != v8; v8 += 3 )
      *v8 = -8;
    *(_QWORD *)(a1 + 208) = 0;
    goto LABEL_11;
  }
  v58 = *(_QWORD **)(a1 + 200);
  v59 = v6 - 1;
  if ( !v59 )
  {
    v64 = 3072;
    v63 = 128;
LABEL_86:
    j___libc_free_0((unsigned __int64)v58);
    *(_DWORD *)(a1 + 216) = v63;
    v65 = (_QWORD *)sub_22077B0(v64);
    v66 = *(unsigned int *)(a1 + 216);
    *(_QWORD *)(a1 + 208) = 0;
    *(_QWORD *)(a1 + 200) = v65;
    for ( j = &v65[3 * v66]; j != v65; v65 += 3 )
    {
      if ( v65 )
        *v65 = -8;
    }
    goto LABEL_11;
  }
  _BitScanReverse(&v59, v59);
  v60 = 1 << (33 - (v59 ^ 0x1F));
  if ( v60 < 64 )
    v60 = 64;
  if ( (_DWORD)v7 != v60 )
  {
    v61 = (4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1);
    v62 = ((((v61 >> 2) | v61 | (((v61 >> 2) | v61) >> 4)) >> 8)
         | (v61 >> 2)
         | v61
         | (((v61 >> 2) | v61) >> 4)
         | (((((v61 >> 2) | v61 | (((v61 >> 2) | v61) >> 4)) >> 8) | (v61 >> 2) | v61 | (((v61 >> 2) | v61) >> 4)) >> 16))
        + 1;
    v63 = v62;
    v64 = 24 * v62;
    goto LABEL_86;
  }
  *(_QWORD *)(a1 + 208) = 0;
  v97 = &v58[3 * v7];
  do
  {
    if ( v58 )
      *v58 = -8;
    v58 += 3;
  }
  while ( v97 != v58 );
LABEL_11:
  v10 = *(_DWORD *)(a1 + 264);
  ++*(_QWORD *)(a1 + 248);
  if ( !v10 )
  {
    v5 = *(_DWORD *)(a1 + 268);
    if ( !v5 )
      goto LABEL_24;
  }
  v11 = *(_QWORD **)(a1 + 256);
  v12 = 4 * v10;
  v13 = *(unsigned int *)(a1 + 272);
  v14 = &v11[4 * v13];
  if ( (unsigned int)(4 * v10) < 0x40 )
    v12 = 64;
  if ( (unsigned int)v13 <= v12 )
  {
    for ( ; v11 != v14; v11 += 4 )
    {
      if ( *v11 != -8 )
      {
        if ( *v11 != -16 )
        {
          v15 = v11[1];
          if ( v15 )
            j_j___libc_free_0(v15);
        }
        *v11 = -8;
      }
    }
LABEL_23:
    *(_QWORD *)(a1 + 264) = 0;
    goto LABEL_24;
  }
  do
  {
    while ( *v11 == -16 )
    {
LABEL_93:
      v11 += 4;
      if ( v11 == v14 )
        goto LABEL_116;
    }
    if ( *v11 != -8 )
    {
      v68 = v11[1];
      if ( v68 )
        j_j___libc_free_0(v68);
      goto LABEL_93;
    }
    v11 += 4;
  }
  while ( v11 != v14 );
LABEL_116:
  v86 = *(_DWORD *)(a1 + 272);
  if ( !v10 )
  {
    if ( v86 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 256));
      *(_QWORD *)(a1 + 256) = 0;
      *(_QWORD *)(a1 + 264) = 0;
      *(_DWORD *)(a1 + 272) = 0;
      goto LABEL_24;
    }
    goto LABEL_23;
  }
  v87 = 64;
  v88 = v10 - 1;
  if ( v88 )
  {
    _BitScanReverse(&v89, v88);
    v87 = 1 << (33 - (v89 ^ 0x1F));
    if ( v87 < 64 )
      v87 = 64;
  }
  v90 = *(_QWORD **)(a1 + 256);
  if ( v87 == v86 )
  {
    *(_QWORD *)(a1 + 264) = 0;
    v98 = &v90[4 * (unsigned int)v87];
    do
    {
      if ( v90 )
        *v90 = -8;
      v90 += 4;
    }
    while ( v98 != v90 );
  }
  else
  {
    j___libc_free_0((unsigned __int64)v90);
    v91 = ((((((((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
             | (4 * v87 / 3u + 1)
             | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 4)
           | (((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
           | (4 * v87 / 3u + 1)
           | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
           | (4 * v87 / 3u + 1)
           | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 4)
         | (((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
         | (4 * v87 / 3u + 1)
         | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 16;
    v92 = (v91
         | (((((((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
             | (4 * v87 / 3u + 1)
             | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 4)
           | (((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
           | (4 * v87 / 3u + 1)
           | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
           | (4 * v87 / 3u + 1)
           | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 4)
         | (((4 * v87 / 3u + 1) | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1)) >> 2)
         | (4 * v87 / 3u + 1)
         | ((unsigned __int64)(4 * v87 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 272) = v92;
    v93 = (_QWORD *)sub_22077B0(32 * v92);
    v94 = *(unsigned int *)(a1 + 272);
    *(_QWORD *)(a1 + 264) = 0;
    *(_QWORD *)(a1 + 256) = v93;
    for ( k = &v93[4 * v94]; k != v93; v93 += 4 )
    {
      if ( v93 )
        *v93 = -8;
    }
  }
LABEL_24:
  v16 = *(_DWORD *)(a1 + 328);
  ++*(_QWORD *)(a1 + 312);
  if ( !v16 )
  {
    v4 = *(_DWORD *)(a1 + 332);
    if ( !v4 )
      goto LABEL_48;
  }
  v17 = *(_QWORD *)(a1 + 320);
  v18 = 64;
  v19 = *(unsigned int *)(a1 + 336);
  v20 = v17 + 72 * v19;
  v21 = 4 * v16;
  if ( (unsigned int)(4 * v16) < 0x40 )
    v21 = 64;
  if ( (unsigned int)v19 <= v21 )
  {
    v102 = 0x100000000LL;
    v104 = v106;
    v109 = 0x100000000LL;
    v22 = v113;
    v101 = &v103;
    v105 = 0x400000000LL;
    v108 = &v110;
    v111 = v113;
    v112 = 0x400000000LL;
    v107 = 2;
    if ( v20 == v17 )
    {
      *(_QWORD *)(a1 + 328) = 0;
      goto LABEL_46;
    }
    while ( 1 )
    {
      if ( *(_DWORD *)v17 == 1 )
      {
        v27 = *(unsigned int *)(v17 + 16);
        if ( v27 == (unsigned int)v102 )
        {
          v28 = *(_DWORD **)(v17 + 8);
          v22 = v101;
          v29 = &v28[v27];
          if ( v28 == v29 )
          {
LABEL_38:
            v30 = *(unsigned int *)(v17 + 40);
            if ( v30 == (unsigned int)v105 )
            {
              v31 = *(_DWORD **)(v17 + 32);
              v22 = v104;
              v32 = &v31[v30];
              if ( v31 == v32 )
                goto LABEL_32;
              if ( *v31 == *(_DWORD *)v104 )
              {
                do
                {
                  ++v31;
                  v22 += 4;
                  if ( v32 == v31 )
                    goto LABEL_32;
                }
                while ( *v31 == *(_DWORD *)v22 );
              }
            }
          }
          else
          {
            while ( *v28 == *(_DWORD *)v22 )
            {
              ++v28;
              v22 += 4;
              if ( v29 == v28 )
                goto LABEL_38;
            }
          }
        }
      }
      *(_DWORD *)v17 = 1;
      sub_39199E0(v17 + 8, (__int64)&v101, (__int64)v22, 1, v4, v5);
      sub_39199E0(v17 + 32, (__int64)&v104, v23, v24, v25, v26);
LABEL_32:
      v17 += 72;
      if ( v17 == v20 )
      {
        *(_QWORD *)(a1 + 328) = 0;
        if ( v104 != v106 )
          _libc_free((unsigned __int64)v104);
LABEL_46:
        v33 = (unsigned __int64)v101;
        if ( v101 != &v103 )
LABEL_47:
          _libc_free(v33);
        goto LABEL_48;
      }
    }
  }
  do
  {
    v69 = *(_QWORD *)(v17 + 32);
    if ( v69 != v17 + 48 )
      _libc_free(v69);
    v70 = *(_QWORD *)(v17 + 8);
    if ( v70 != v17 + 24 )
      _libc_free(v70);
    v17 += 72;
  }
  while ( v17 != v20 );
  v71 = *(unsigned int *)(a1 + 336);
  if ( !v16 )
  {
    if ( (_DWORD)v71 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 320));
      *(_QWORD *)(a1 + 320) = 0;
      *(_QWORD *)(a1 + 328) = 0;
      *(_DWORD *)(a1 + 336) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 328) = 0;
    }
    goto LABEL_48;
  }
  v72 = 64;
  v73 = v16 - 1;
  if ( v73 )
  {
    _BitScanReverse(&v74, v73);
    v18 = 33 - (v74 ^ 0x1F);
    v72 = 1 << (33 - (v74 ^ 0x1F));
    if ( v72 < 64 )
      v72 = 64;
  }
  v75 = *(_QWORD *)(a1 + 320);
  if ( (_DWORD)v71 == v72 )
  {
    *(_QWORD *)(a1 + 328) = 0;
    v109 = 0x100000000LL;
    v112 = 0x400000000LL;
    v99 = v75 + 72 * v71;
    v108 = &v110;
    v111 = v113;
    v107 = 1;
    do
    {
      if ( v75 )
      {
        v100 = v107;
        *(_DWORD *)(v75 + 16) = 0;
        *(_DWORD *)(v75 + 20) = 1;
        *(_DWORD *)v75 = v100;
        *(_QWORD *)(v75 + 8) = v75 + 24;
        if ( (_DWORD)v109 )
          sub_39199E0(v75 + 8, (__int64)&v108, v71, v18, v4, v5);
        *(_DWORD *)(v75 + 40) = 0;
        *(_QWORD *)(v75 + 32) = v75 + 48;
        *(_DWORD *)(v75 + 44) = 4;
        if ( (_DWORD)v112 )
          sub_39199E0(v75 + 32, (__int64)&v111, v71, v18, v4, v5);
      }
      v75 += 72;
    }
    while ( v99 != v75 );
  }
  else
  {
    j___libc_free_0(*(_QWORD *)(a1 + 320));
    v76 = ((((((((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
             | (4 * v72 / 3u + 1)
             | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 4)
           | (((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
           | (4 * v72 / 3u + 1)
           | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
           | (4 * v72 / 3u + 1)
           | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 4)
         | (((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
         | (4 * v72 / 3u + 1)
         | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 16;
    v77 = (v76
         | (((((((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
             | (4 * v72 / 3u + 1)
             | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 4)
           | (((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
           | (4 * v72 / 3u + 1)
           | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
           | (4 * v72 / 3u + 1)
           | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 4)
         | (((4 * v72 / 3u + 1) | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1)) >> 2)
         | (4 * v72 / 3u + 1)
         | ((unsigned __int64)(4 * v72 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 336) = v77;
    v108 = &v110;
    v78 = sub_22077B0(72 * v77);
    *(_QWORD *)(a1 + 320) = v78;
    v109 = 0x100000000LL;
    v112 = 0x400000000LL;
    v81 = *(unsigned int *)(a1 + 336);
    *(_QWORD *)(a1 + 328) = 0;
    v111 = v113;
    v82 = v78 + 72 * v81;
    v107 = 1;
    if ( v78 == v82 )
      goto LABEL_48;
    do
    {
      if ( v78 )
      {
        v84 = v107;
        v85 = (unsigned int)v109;
        *(_DWORD *)(v78 + 16) = 0;
        *(_DWORD *)(v78 + 20) = 1;
        *(_DWORD *)v78 = v84;
        *(_QWORD *)(v78 + 8) = v78 + 24;
        if ( (_DWORD)v85 )
          sub_39199E0(v78 + 8, (__int64)&v108, v78 + 24, v85, v79, v80);
        *(_DWORD *)(v78 + 40) = 0;
        *(_QWORD *)(v78 + 32) = v78 + 48;
        v83 = (unsigned int)v112;
        *(_DWORD *)(v78 + 44) = 4;
        if ( (_DWORD)v83 )
          sub_39199E0(v78 + 32, (__int64)&v111, v83, v85, v79, v80);
      }
      v78 += 72;
    }
    while ( v82 != v78 );
  }
  if ( v111 != v113 )
    _libc_free((unsigned __int64)v111);
  v33 = (unsigned __int64)v108;
  if ( v108 != &v110 )
    goto LABEL_47;
LABEL_48:
  v34 = *(_QWORD *)(a1 + 344);
  v35 = v34 + ((unsigned __int64)*(unsigned int *)(a1 + 352) << 6);
  while ( v34 != v35 )
  {
    while ( 1 )
    {
      v35 -= 64;
      v36 = *(_QWORD *)(v35 + 32);
      if ( v36 != v35 + 48 )
        _libc_free(v36);
      v37 = *(_QWORD *)(v35 + 8);
      if ( v37 == v35 + 24 )
        break;
      _libc_free(v37);
      if ( v34 == v35 )
        goto LABEL_54;
    }
  }
LABEL_54:
  v38 = *(unsigned int *)(a1 + 704);
  v39 = *(_QWORD *)(a1 + 696);
  *(_DWORD *)(a1 + 352) = 0;
  *(_DWORD *)(a1 + 624) = 0;
  v40 = v39 + (v38 << 6);
  while ( v39 != v40 )
  {
    v40 -= 64;
    v41 = *(_QWORD *)(v40 + 40);
    if ( v41 != v40 + 56 )
      _libc_free(v41);
  }
  v42 = *(_DWORD *)(a1 + 296);
  ++*(_QWORD *)(a1 + 280);
  *(_DWORD *)(a1 + 704) = 0;
  if ( v42 )
  {
    v46 = 4 * v42;
    v43 = *(unsigned int *)(a1 + 304);
    if ( (unsigned int)(4 * v42) < 0x40 )
      v46 = 64;
    if ( v46 >= (unsigned int)v43 )
    {
LABEL_61:
      v44 = *(_QWORD **)(a1 + 288);
      for ( m = &v44[2 * v43]; m != v44; v44 += 2 )
        *v44 = -8;
      *(_QWORD *)(a1 + 296) = 0;
      goto LABEL_64;
    }
    v47 = *(_QWORD **)(a1 + 288);
    v48 = v42 - 1;
    if ( v48 )
    {
      _BitScanReverse(&v48, v48);
      v49 = (unsigned int)(1 << (33 - (v48 ^ 0x1F)));
      if ( (int)v49 < 64 )
        v49 = 64;
      if ( (_DWORD)v49 == (_DWORD)v43 )
      {
        *(_QWORD *)(a1 + 296) = 0;
        v96 = &v47[2 * v49];
        do
        {
          if ( v47 )
            *v47 = -8;
          v47 += 2;
        }
        while ( v96 != v47 );
        goto LABEL_64;
      }
      v50 = (((4 * (int)v49 / 3u + 1) | ((unsigned __int64)(4 * (int)v49 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v49 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v49 / 3u + 1) >> 1)
          | (((((4 * (int)v49 / 3u + 1) | ((unsigned __int64)(4 * (int)v49 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v49 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v49 / 3u + 1) >> 1)) >> 4);
      v51 = (v50 >> 8) | v50;
      v52 = (v51 | (v51 >> 16)) + 1;
      v53 = 16 * ((v51 | (v51 >> 16)) + 1);
    }
    else
    {
      v53 = 2048;
      v52 = 128;
    }
    j___libc_free_0((unsigned __int64)v47);
    *(_DWORD *)(a1 + 304) = v52;
    v54 = (_QWORD *)sub_22077B0(v53);
    v55 = *(unsigned int *)(a1 + 304);
    *(_QWORD *)(a1 + 296) = 0;
    *(_QWORD *)(a1 + 288) = v54;
    for ( n = &v54[2 * v55]; n != v54; v54 += 2 )
    {
      if ( v54 )
        *v54 = -8;
    }
    goto LABEL_64;
  }
  if ( *(_DWORD *)(a1 + 300) )
  {
    v43 = *(unsigned int *)(a1 + 304);
    if ( (unsigned int)v43 <= 0x40 )
      goto LABEL_61;
    j___libc_free_0(*(_QWORD *)(a1 + 288));
    *(_QWORD *)(a1 + 288) = 0;
    *(_QWORD *)(a1 + 296) = 0;
    *(_DWORD *)(a1 + 304) = 0;
  }
LABEL_64:
  *(_QWORD *)(a1 + 968) = 0;
}
