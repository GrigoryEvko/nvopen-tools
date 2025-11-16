// Function: sub_1E53B20
// Address: 0x1e53b20
//
__int64 __fastcall sub_1E53B20(__int64 a1, __int64 a2)
{
  __int64 **v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 **v5; // rax
  __int64 **v6; // rax
  __int64 *v7; // rsi
  __int64 *v8; // r14
  __int64 *v9; // r12
  __int64 v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rbx
  void *v13; // rdi
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  bool v16; // zf
  unsigned __int64 v17; // r12
  char *v18; // rcx
  signed __int64 v19; // r12
  unsigned int v20; // ecx
  __int64 v21; // rdx
  char *v22; // rax
  char *v23; // rdx
  __int64 *v24; // r14
  __int64 *v25; // r12
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // r12
  char *v31; // rcx
  signed __int64 v32; // r12
  unsigned int v33; // r15d
  __int64 v34; // rax
  __int64 v35; // rcx
  unsigned int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // r14
  __int64 v42; // rax
  void *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  char *v46; // rcx
  size_t v47; // rdx
  char *v48; // rax
  __int64 *v49; // r12
  __int64 v50; // r10
  unsigned int v51; // eax
  void *v52; // rax
  __int64 *v53; // r12
  __int64 v54; // r10
  unsigned int v55; // eax
  void *v56; // rax
  unsigned int v57; // ecx
  char *v58; // rax
  char *v59; // rdx
  int v61; // r10d
  void *v62; // rax
  __int64 v63; // rdx
  void *v64; // rax
  __int64 v65; // rdx
  unsigned int v66; // eax
  int v67; // eax
  __int64 v68; // r10
  unsigned int v69; // eax
  unsigned int v70; // eax
  int v71; // eax
  unsigned __int64 v72; // r12
  unsigned int v73; // eax
  __int64 **v74; // [rsp+10h] [rbp-220h]
  __int64 **v75; // [rsp+10h] [rbp-220h]
  __int64 v77; // [rsp+20h] [rbp-210h]
  unsigned __int64 v79; // [rsp+38h] [rbp-1F8h]
  signed __int64 v80; // [rsp+38h] [rbp-1F8h]
  __int64 *v81; // [rsp+38h] [rbp-1F8h]
  __int64 *v82; // [rsp+38h] [rbp-1F8h]
  unsigned __int64 v83; // [rsp+38h] [rbp-1F8h]
  __int64 v84; // [rsp+40h] [rbp-1F0h]
  __int64 v85; // [rsp+40h] [rbp-1F0h]
  __int64 v86; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v87; // [rsp+58h] [rbp-1D8h]
  __int64 v88; // [rsp+60h] [rbp-1D0h]
  __int64 v89; // [rsp+68h] [rbp-1C8h]
  __int64 v90; // [rsp+70h] [rbp-1C0h]
  __int64 v91; // [rsp+78h] [rbp-1B8h]
  __int64 v92; // [rsp+80h] [rbp-1B0h]
  __int64 v93; // [rsp+90h] [rbp-1A0h] BYREF
  void *v94; // [rsp+98h] [rbp-198h]
  __int64 v95; // [rsp+A0h] [rbp-190h]
  __int64 v96; // [rsp+A8h] [rbp-188h]
  void *src; // [rsp+B0h] [rbp-180h]
  __int64 *v98; // [rsp+B8h] [rbp-178h]
  __int64 v99; // [rsp+C0h] [rbp-170h]
  char v100; // [rsp+C8h] [rbp-168h]
  __int64 v101; // [rsp+CCh] [rbp-164h]
  __int64 v102; // [rsp+D4h] [rbp-15Ch]
  __int64 v103; // [rsp+E0h] [rbp-150h]
  int v104; // [rsp+E8h] [rbp-148h]
  __int64 v105; // [rsp+F0h] [rbp-140h] BYREF
  _BYTE *v106; // [rsp+F8h] [rbp-138h]
  void *s; // [rsp+100h] [rbp-130h]
  _BYTE v108[12]; // [rsp+108h] [rbp-128h]
  _BYTE v109[72]; // [rsp+118h] [rbp-118h] BYREF
  __int64 v110; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v111; // [rsp+168h] [rbp-C8h]
  _QWORD v112[8]; // [rsp+170h] [rbp-C0h] BYREF
  __int64 *v113; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v114; // [rsp+1B8h] [rbp-78h]
  _BYTE v115[112]; // [rsp+1C0h] [rbp-70h] BYREF

  v2 = &v113;
  v106 = v109;
  v3 = *(_QWORD *)a2;
  s = v109;
  v4 = *(unsigned int *)(a2 + 8);
  v86 = 0;
  v87 = 0;
  v77 = v3 + 96 * v4;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v105 = 0;
  *(_QWORD *)v108 = 8;
  *(_DWORD *)&v108[8] = 0;
  if ( v3 == v77 )
    goto LABEL_11;
  do
  {
    v110 = 0;
    v5 = (__int64 **)v112;
    v111 = 1;
    do
      *v5++ = (__int64 *)-8LL;
    while ( v5 != v2 );
    v113 = (__int64 *)v115;
    v114 = 0x800000000LL;
    if ( !sub_1E537F0(v3, (__int64)&v110, 0) )
      goto LABEL_5;
    v93 = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    src = 0;
    v98 = 0;
    v99 = 0;
    v81 = &v113[(unsigned int)v114];
    if ( v113 == v81 )
      goto LABEL_94;
    v74 = v2;
    v49 = v113;
    do
    {
      v50 = *v49;
      ++v105;
      if ( s == v106 )
        goto LABEL_88;
      v84 = v50;
      v51 = 4 * (*(_DWORD *)&v108[4] - *(_DWORD *)&v108[8]);
      if ( v51 < 0x20 )
        v51 = 32;
      if ( v51 >= *(_DWORD *)v108 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v108);
        v50 = v84;
LABEL_88:
        *(_QWORD *)&v108[4] = 0;
        goto LABEL_89;
      }
      sub_16CC920((__int64)&v105);
      v50 = v84;
LABEL_89:
      ++v49;
      sub_1E515D0(v50, (__int64)&v93, (__int64)&v86, v3, (__int64)&v105);
    }
    while ( v81 != v49 );
    v52 = src;
    v2 = v74;
    if ( v98 != src )
    {
      sub_1E51930(v3, (__int64 *)src, v98);
      v52 = src;
    }
    if ( v52 )
      j_j___libc_free_0(v52, v99 - (_QWORD)v52);
LABEL_94:
    j___libc_free_0(v94);
LABEL_5:
    sub_1E48140((__int64)&v110);
    LODWORD(v114) = 0;
    if ( !sub_1E537F0((__int64)&v86, (__int64)&v110, 0) )
      goto LABEL_6;
    v93 = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    src = 0;
    v98 = 0;
    v99 = 0;
    v82 = &v113[(unsigned int)v114];
    if ( v82 == v113 )
      goto LABEL_108;
    v75 = v2;
    v53 = v113;
    while ( 2 )
    {
      v54 = *v53;
      ++v105;
      if ( s == v106 )
      {
LABEL_102:
        *(_QWORD *)&v108[4] = 0;
      }
      else
      {
        v85 = v54;
        v55 = 4 * (*(_DWORD *)&v108[4] - *(_DWORD *)&v108[8]);
        if ( v55 < 0x20 )
          v55 = 32;
        if ( *(_DWORD *)v108 <= v55 )
        {
          memset(s, -1, 8LL * *(unsigned int *)v108);
          v54 = v85;
          goto LABEL_102;
        }
        sub_16CC920((__int64)&v105);
        v54 = v85;
      }
      ++v53;
      sub_1E515D0(v54, (__int64)&v93, v3, (__int64)&v86, (__int64)&v105);
      if ( v82 != v53 )
        continue;
      break;
    }
    v56 = src;
    v2 = v75;
    if ( v98 != src )
    {
      sub_1E51930(v3, (__int64 *)src, v98);
      v56 = src;
    }
    if ( v56 )
      j_j___libc_free_0(v56, v99 - (_QWORD)v56);
LABEL_108:
    j___libc_free_0(v94);
LABEL_6:
    sub_1E51930((__int64)&v86, *(__int64 **)(v3 + 32), *(__int64 **)(v3 + 40));
    if ( v113 != (__int64 *)v115 )
      _libc_free((unsigned __int64)v113);
    if ( (v111 & 1) == 0 )
      j___libc_free_0(v112[0]);
    v3 += 96;
  }
  while ( v77 != v3 );
LABEL_11:
  v93 = 0;
  v6 = (__int64 **)v112;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  src = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v110 = 0;
  v111 = 1;
  do
    *v6++ = (__int64 *)-8LL;
  while ( v6 != v2 );
  v7 = &v110;
  v113 = (__int64 *)v115;
  v114 = 0x800000000LL;
  if ( sub_1E537F0((__int64)&v86, (__int64)&v110, 0) )
  {
    v8 = v113;
    v9 = &v113[(unsigned int)v114];
    if ( v9 != v113 )
    {
      do
      {
        v10 = *v8;
        v7 = &v93;
        ++v8;
        sub_1E51780(v10, (__int64)&v93, (__int64)&v86);
      }
      while ( v9 != v8 );
    }
  }
  if ( v98 != src )
  {
    v11 = *(_DWORD *)(a2 + 8);
    if ( v11 >= *(_DWORD *)(a2 + 12) )
    {
      v7 = 0;
      sub_1E44B20(a2, 0);
      v11 = *(_DWORD *)(a2 + 8);
    }
    v12 = *(_QWORD *)a2 + 96LL * v11;
    if ( v12 )
    {
      *(_QWORD *)v12 = 0;
      v13 = 0;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = 0;
      *(_DWORD *)(v12 + 24) = 0;
      j___libc_free_0(0);
      v15 = (unsigned int)v96;
      *(_DWORD *)(v12 + 24) = v96;
      if ( (_DWORD)v15 )
      {
        v62 = (void *)sub_22077B0(8 * v15);
        v63 = *(unsigned int *)(v12 + 24);
        *(_QWORD *)(v12 + 8) = v62;
        v13 = v62;
        *(_QWORD *)(v12 + 16) = v95;
        v7 = (__int64 *)v94;
        memcpy(v62, v94, 8 * v63);
      }
      else
      {
        *(_QWORD *)(v12 + 8) = 0;
        *(_QWORD *)(v12 + 16) = 0;
      }
      v17 = (char *)v98 - (_BYTE *)src;
      v16 = v98 == src;
      *(_QWORD *)(v12 + 32) = 0;
      *(_QWORD *)(v12 + 40) = 0;
      *(_QWORD *)(v12 + 48) = 0;
      if ( !v16 )
      {
        if ( v17 <= 0x7FFFFFFFFFFFFFF8LL )
        {
          v18 = (char *)sub_22077B0(v17);
          goto LABEL_25;
        }
LABEL_160:
        sub_4261EA(v13, v7, v14);
      }
      v18 = 0;
LABEL_25:
      *(_QWORD *)(v12 + 32) = v18;
      *(_QWORD *)(v12 + 48) = &v18[v17];
      *(_QWORD *)(v12 + 40) = v18;
      v19 = (char *)v98 - (_BYTE *)src;
      if ( v98 != src )
        v18 = (char *)memmove(v18, src, (char *)v98 - (_BYTE *)src);
      *(_QWORD *)(v12 + 40) = &v18[v19];
      *(_BYTE *)(v12 + 56) = v100;
      *(_QWORD *)(v12 + 60) = v101;
      *(_QWORD *)(v12 + 68) = v102;
      *(_QWORD *)(v12 + 80) = v103;
      *(_DWORD *)(v12 + 88) = v104;
      v11 = *(_DWORD *)(a2 + 8);
    }
    *(_DWORD *)(a2 + 8) = v11 + 1;
  }
  ++v93;
  if ( !(_DWORD)v95 )
  {
    if ( !HIDWORD(v95) )
      goto LABEL_36;
    v21 = (unsigned int)v96;
    if ( (unsigned int)v96 <= 0x40 )
      goto LABEL_33;
    j___libc_free_0(v94);
    LODWORD(v96) = 0;
    goto LABEL_114;
  }
  v20 = 4 * v95;
  v21 = (unsigned int)v96;
  if ( (unsigned int)(4 * v95) < 0x40 )
    v20 = 64;
  if ( (unsigned int)v96 > v20 )
  {
    if ( (_DWORD)v95 == 1 )
    {
      v72 = 86;
    }
    else
    {
      _BitScanReverse(&v70, v95 - 1);
      v71 = 1 << (33 - (v70 ^ 0x1F));
      if ( v71 < 64 )
        v71 = 64;
      if ( (_DWORD)v96 == v71 )
        goto LABEL_157;
      v72 = 4 * v71 / 3u + 1;
    }
    j___libc_free_0(v94);
    v73 = sub_1454B60(v72);
    LODWORD(v96) = v73;
    if ( v73 )
    {
      v94 = (void *)sub_22077B0(8LL * v73);
LABEL_157:
      sub_1E48530((__int64)&v93);
      goto LABEL_36;
    }
LABEL_114:
    v94 = 0;
    goto LABEL_35;
  }
LABEL_33:
  v22 = (char *)v94;
  v23 = (char *)v94 + 8 * v21;
  if ( v94 != v23 )
  {
    do
    {
      *(_QWORD *)v22 = -8;
      v22 += 8;
    }
    while ( v23 != v22 );
  }
LABEL_35:
  v95 = 0;
LABEL_36:
  if ( src != v98 )
    v98 = (__int64 *)src;
  v7 = &v110;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  if ( sub_1E534B0((__int64)&v86, (__int64)&v110, 0) )
  {
    v24 = v113;
    v25 = &v113[(unsigned int)v114];
    if ( v25 != v113 )
    {
      do
      {
        v26 = *v24;
        v7 = &v93;
        ++v24;
        sub_1E51780(v26, (__int64)&v93, (__int64)&v86);
      }
      while ( v25 != v24 );
    }
  }
  if ( v98 != src )
  {
    v27 = *(_DWORD *)(a2 + 8);
    if ( v27 >= *(_DWORD *)(a2 + 12) )
    {
      v7 = 0;
      sub_1E44B20(a2, 0);
      v27 = *(_DWORD *)(a2 + 8);
    }
    v28 = *(_QWORD *)a2 + 96LL * v27;
    if ( v28 )
    {
      *(_QWORD *)v28 = 0;
      v13 = 0;
      *(_QWORD *)(v28 + 8) = 0;
      *(_QWORD *)(v28 + 16) = 0;
      *(_DWORD *)(v28 + 24) = 0;
      j___libc_free_0(0);
      v29 = (unsigned int)v96;
      *(_DWORD *)(v28 + 24) = v96;
      if ( (_DWORD)v29 )
      {
        v64 = (void *)sub_22077B0(8 * v29);
        v65 = *(unsigned int *)(v28 + 24);
        *(_QWORD *)(v28 + 8) = v64;
        v13 = v64;
        *(_QWORD *)(v28 + 16) = v95;
        v7 = (__int64 *)v94;
        memcpy(v64, v94, 8 * v65);
      }
      else
      {
        *(_QWORD *)(v28 + 8) = 0;
        *(_QWORD *)(v28 + 16) = 0;
      }
      v30 = (char *)v98 - (_BYTE *)src;
      v16 = v98 == src;
      *(_QWORD *)(v28 + 32) = 0;
      *(_QWORD *)(v28 + 40) = 0;
      *(_QWORD *)(v28 + 48) = 0;
      if ( v16 )
      {
        v31 = 0;
      }
      else
      {
        if ( v30 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_160;
        v31 = (char *)sub_22077B0(v30);
      }
      *(_QWORD *)(v28 + 32) = v31;
      *(_QWORD *)(v28 + 48) = &v31[v30];
      *(_QWORD *)(v28 + 40) = v31;
      v32 = (char *)v98 - (_BYTE *)src;
      if ( v98 != src )
        v31 = (char *)memmove(v31, src, (char *)v98 - (_BYTE *)src);
      *(_QWORD *)(v28 + 40) = &v31[v32];
      *(_BYTE *)(v28 + 56) = v100;
      *(_QWORD *)(v28 + 60) = v101;
      *(_QWORD *)(v28 + 68) = v102;
      *(_QWORD *)(v28 + 80) = v103;
      *(_DWORD *)(v28 + 88) = v104;
      v27 = *(_DWORD *)(a2 + 8);
    }
    *(_DWORD *)(a2 + 8) = v27 + 1;
  }
  v33 = 0;
  v34 = 0;
  v35 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) != v35 )
  {
    while ( 2 )
    {
      v38 = v35 + 272 * v34;
      if ( !(_DWORD)v89 )
        goto LABEL_59;
      v36 = (v89 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v37 = *(_QWORD *)(v87 + 8LL * v36);
      if ( v38 == v37 )
      {
LABEL_57:
        v34 = ++v33;
        if ( v33 >= 0xF0F0F0F0F0F0F0F1LL * ((*(_QWORD *)(a1 + 56) - v35) >> 4) )
          goto LABEL_121;
        continue;
      }
      break;
    }
    v61 = 1;
    while ( v37 != -8 )
    {
      v36 = (v89 - 1) & (v61 + v36);
      v37 = *(_QWORD *)(v87 + 8LL * v36);
      if ( v38 == v37 )
        goto LABEL_57;
      ++v61;
    }
LABEL_59:
    ++v93;
    if ( !(_DWORD)v95 )
    {
      v7 = &v93;
      if ( HIDWORD(v95) )
      {
        v39 = (unsigned int)v96;
        if ( (unsigned int)v96 > 0x40 )
        {
          j___libc_free_0(v94);
          LODWORD(v96) = 0;
          goto LABEL_63;
        }
LABEL_118:
        v58 = (char *)v94;
        v59 = (char *)v94 + 8 * v39;
        if ( v94 != v59 )
        {
          do
          {
            *(_QWORD *)v58 = -8;
            v58 += 8;
          }
          while ( v59 != v58 );
        }
LABEL_64:
        v95 = 0;
        v7 = &v93;
      }
LABEL_65:
      if ( src != v98 )
        v98 = (__int64 *)src;
      v100 = 0;
      v101 = 0;
      v102 = 0;
      v103 = 0;
      sub_1E51780(v38, (__int64)&v93, (__int64)&v86);
      if ( v98 != src )
      {
        v40 = *(_DWORD *)(a2 + 8);
        if ( v40 >= *(_DWORD *)(a2 + 12) )
        {
          v7 = 0;
          sub_1E44B20(a2, 0);
          v40 = *(_DWORD *)(a2 + 8);
        }
        v41 = *(_QWORD *)a2 + 96LL * v40;
        if ( v41 )
        {
          *(_QWORD *)v41 = 0;
          v13 = 0;
          *(_QWORD *)(v41 + 8) = 0;
          *(_QWORD *)(v41 + 16) = 0;
          *(_DWORD *)(v41 + 24) = 0;
          j___libc_free_0(0);
          v42 = (unsigned int)v96;
          *(_DWORD *)(v41 + 24) = v96;
          if ( (_DWORD)v42 )
          {
            v43 = (void *)sub_22077B0(8 * v42);
            v44 = *(unsigned int *)(v41 + 24);
            *(_QWORD *)(v41 + 8) = v43;
            v13 = v43;
            *(_QWORD *)(v41 + 16) = v95;
            v7 = (__int64 *)v94;
            memcpy(v43, v94, 8 * v44);
          }
          else
          {
            *(_QWORD *)(v41 + 8) = 0;
            *(_QWORD *)(v41 + 16) = 0;
          }
          v14 = (char *)v98 - (_BYTE *)src;
          v16 = v98 == src;
          *(_QWORD *)(v41 + 32) = 0;
          *(_QWORD *)(v41 + 40) = 0;
          *(_QWORD *)(v41 + 48) = 0;
          if ( v16 )
          {
            v46 = 0;
          }
          else
          {
            if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_160;
            v79 = v14;
            v45 = sub_22077B0(v14);
            v14 = v79;
            v46 = (char *)v45;
          }
          *(_QWORD *)(v41 + 32) = v46;
          *(_QWORD *)(v41 + 48) = &v46[v14];
          *(_QWORD *)(v41 + 40) = v46;
          v47 = (char *)v98 - (_BYTE *)src;
          if ( v98 != src )
          {
            v80 = (char *)v98 - (_BYTE *)src;
            v48 = (char *)memmove(v46, src, v47);
            v47 = v80;
            v46 = v48;
          }
          *(_QWORD *)(v41 + 40) = &v46[v47];
          *(_BYTE *)(v41 + 56) = v100;
          *(_QWORD *)(v41 + 60) = v101;
          *(_QWORD *)(v41 + 68) = v102;
          *(_QWORD *)(v41 + 80) = v103;
          *(_DWORD *)(v41 + 88) = v104;
          v40 = *(_DWORD *)(a2 + 8);
        }
        *(_DWORD *)(a2 + 8) = v40 + 1;
      }
      v35 = *(_QWORD *)(a1 + 48);
      goto LABEL_57;
    }
    v57 = 4 * v95;
    v39 = (unsigned int)v96;
    if ( (unsigned int)(4 * v95) < 0x40 )
      v57 = 64;
    if ( v57 >= (unsigned int)v96 )
      goto LABEL_118;
    if ( (_DWORD)v95 == 1 )
    {
      v68 = 86;
LABEL_147:
      v83 = v68;
      j___libc_free_0(v94);
      v69 = sub_1454B60(v83);
      LODWORD(v96) = v69;
      if ( !v69 )
      {
LABEL_63:
        v94 = 0;
        goto LABEL_64;
      }
      v94 = (void *)sub_22077B0(8LL * v69);
    }
    else
    {
      _BitScanReverse(&v66, v95 - 1);
      v67 = 1 << (33 - (v66 ^ 0x1F));
      if ( v67 < 64 )
        v67 = 64;
      if ( (_DWORD)v96 != v67 )
      {
        v68 = 4 * v67 / 3u + 1;
        goto LABEL_147;
      }
    }
    sub_1E48530((__int64)&v93);
    v7 = &v93;
    goto LABEL_65;
  }
LABEL_121:
  if ( v113 != (__int64 *)v115 )
    _libc_free((unsigned __int64)v113);
  if ( (v111 & 1) == 0 )
    j___libc_free_0(v112[0]);
  if ( src )
    j_j___libc_free_0(src, v99 - (_QWORD)src);
  j___libc_free_0(v94);
  if ( s != v106 )
    _libc_free((unsigned __int64)s);
  if ( v90 )
    j_j___libc_free_0(v90, v92 - v90);
  return j___libc_free_0(v87);
}
