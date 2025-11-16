// Function: sub_24F5860
// Address: 0x24f5860
//
void __fastcall sub_24F5860(__int64 a1, __int64 **a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // r12
  char v10; // cl
  __int64 v11; // r8
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r10
  unsigned __int64 v16; // rsi
  unsigned int v17; // eax
  __int64 v18; // rdi
  int v19; // edx
  unsigned int v20; // ecx
  __int64 v21; // rcx
  int v22; // eax
  __int64 v23; // r15
  _QWORD *v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // r12
  unsigned __int64 v30; // rdi
  unsigned __int64 *v31; // r12
  __int64 v32; // rax
  unsigned __int64 *v33; // r8
  unsigned __int64 *v34; // r14
  __int64 v35; // r15
  __int64 v36; // r9
  int v37; // edx
  __int64 v38; // rax
  _BYTE *v39; // r10
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // rax
  _BYTE *v44; // rbx
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // rax
  _QWORD **v48; // rax
  _QWORD **v49; // rdx
  _QWORD *v50; // r9
  _QWORD *v51; // rax
  __int64 v52; // r8
  unsigned __int64 v53; // r9
  __int64 v54; // rax
  unsigned __int64 *v55; // rax
  __int64 v56; // rax
  int v57; // r11d
  __int64 v58; // rax
  int v59; // eax
  unsigned __int64 v60; // rdi
  _QWORD *v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rcx
  unsigned __int64 v64; // r12
  _QWORD *v65; // rcx
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // r8
  __int64 v69; // rbx
  __int64 v70; // r12
  unsigned __int64 v71; // r13
  unsigned __int64 v72; // rdi
  int v73; // eax
  int v74; // eax
  __int64 v75; // rcx
  int v76; // edx
  unsigned int v77; // eax
  __int64 v78; // rcx
  int v79; // edx
  unsigned int v80; // eax
  int v81; // r10d
  __int64 v82; // r8
  char *v83; // rbx
  char *v84; // r8
  __int64 v85; // r10
  unsigned __int64 v86; // rdi
  unsigned __int64 v87; // rcx
  __int64 v88; // rax
  unsigned __int64 *v89; // rax
  unsigned __int64 *v90; // rdx
  unsigned __int64 v91; // rax
  int v92; // edx
  int v93; // edx
  __int64 v94; // r10
  unsigned __int64 v95; // rcx
  char *v96; // rax
  size_t v97; // rdx
  _QWORD *v98; // rax
  unsigned __int64 v99; // rax
  unsigned __int64 v100; // rax
  int v101; // ecx
  __int64 v102; // rax
  unsigned __int64 v103; // rbx
  __int64 v104; // rax
  unsigned __int64 v105; // r9
  unsigned __int64 v106; // r8
  const void *v107; // rsi
  void *v108; // rcx
  __int64 v109; // rax
  void *v110; // rax
  int v111; // r10d
  unsigned __int64 v112; // [rsp+8h] [rbp-78h]
  _QWORD *v113; // [rsp+8h] [rbp-78h]
  unsigned __int64 v114; // [rsp+10h] [rbp-70h]
  char *v115; // [rsp+10h] [rbp-70h]
  unsigned __int64 v116; // [rsp+10h] [rbp-70h]
  __int64 v117; // [rsp+18h] [rbp-68h]
  unsigned __int64 v118; // [rsp+18h] [rbp-68h]
  __int64 v119; // [rsp+18h] [rbp-68h]
  unsigned __int64 v120; // [rsp+18h] [rbp-68h]
  unsigned __int64 v121; // [rsp+18h] [rbp-68h]
  unsigned __int64 v122; // [rsp+20h] [rbp-60h]
  __int64 *v123; // [rsp+20h] [rbp-60h]
  unsigned __int64 v124; // [rsp+20h] [rbp-60h]
  unsigned __int64 v125; // [rsp+20h] [rbp-60h]
  _QWORD *v126; // [rsp+20h] [rbp-60h]
  unsigned __int64 v127; // [rsp+20h] [rbp-60h]
  unsigned __int64 v128; // [rsp+20h] [rbp-60h]
  _QWORD *v129; // [rsp+20h] [rbp-60h]
  __int64 v130; // [rsp+28h] [rbp-58h]
  __int64 v131; // [rsp+28h] [rbp-58h]
  int v132; // [rsp+28h] [rbp-58h]
  unsigned __int64 v135[7]; // [rsp+48h] [rbp-38h] BYREF

  v7 = a1;
  v8 = *a2;
  v9 = **a2;
  v10 = *(_BYTE *)(a1 + 16) & 1;
  if ( v10 )
  {
    v11 = a1 + 24;
    v12 = 7;
  }
  else
  {
    v16 = *(unsigned int *)(a1 + 32);
    v11 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v16 )
    {
      v17 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)(a1 + 8);
      v18 = 0;
      v19 = (v17 >> 1) + 1;
LABEL_8:
      v20 = 3 * v16;
      goto LABEL_9;
    }
    v12 = v16 - 1;
  }
  v13 = v12 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v14 = (__int64 *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( v9 == *v14 )
    return;
  v57 = 1;
  v18 = 0;
  while ( v15 != -4096 )
  {
    if ( v15 != -8192 || v18 )
      v14 = (__int64 *)v18;
    v13 = v12 & (v57 + v13);
    a6 = v11 + 16LL * v13;
    v15 = *(_QWORD *)a6;
    if ( v9 == *(_QWORD *)a6 )
      return;
    ++v57;
    v18 = (__int64)v14;
    v14 = (__int64 *)(v11 + 16LL * v13);
  }
  if ( !v18 )
    v18 = (__int64)v14;
  v17 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)(v7 + 8);
  v19 = (v17 >> 1) + 1;
  if ( !v10 )
  {
    v16 = *(unsigned int *)(v7 + 32);
    goto LABEL_8;
  }
  v20 = 24;
  v16 = 8;
LABEL_9:
  if ( 4 * v19 < v20 )
  {
    if ( (int)v16 - *(_DWORD *)(v7 + 20) - v19 > (unsigned int)v16 >> 3 )
      goto LABEL_11;
    sub_24F5420(v7 + 8, v16);
    if ( (*(_BYTE *)(v7 + 16) & 1) != 0 )
    {
      v78 = v7 + 24;
      v79 = 7;
      goto LABEL_100;
    }
    v93 = *(_DWORD *)(v7 + 32);
    v78 = *(_QWORD *)(v7 + 24);
    if ( v93 )
    {
      v79 = v93 - 1;
LABEL_100:
      v80 = v79 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v18 = v78 + 16LL * v80;
      v16 = *(_QWORD *)v18;
      if ( v9 != *(_QWORD *)v18 )
      {
        v81 = 1;
        v82 = 0;
        while ( v16 != -4096 )
        {
          if ( v16 == -8192 && !v82 )
            v82 = v18;
          a6 = (unsigned int)(v81 + 1);
          v80 = v79 & (v81 + v80);
          v18 = v78 + 16LL * v80;
          v16 = *(_QWORD *)v18;
          if ( v9 == *(_QWORD *)v18 )
            goto LABEL_97;
          ++v81;
        }
LABEL_103:
        if ( v82 )
          v18 = v82;
        goto LABEL_97;
      }
      goto LABEL_97;
    }
LABEL_150:
    *(_DWORD *)(v7 + 16) = (2 * (*(_DWORD *)(v7 + 16) >> 1) + 2) | *(_DWORD *)(v7 + 16) & 1;
    BUG();
  }
  sub_24F5420(v7 + 8, 2 * v16);
  if ( (*(_BYTE *)(v7 + 16) & 1) != 0 )
  {
    v75 = v7 + 24;
    v76 = 7;
  }
  else
  {
    v92 = *(_DWORD *)(v7 + 32);
    v75 = *(_QWORD *)(v7 + 24);
    if ( !v92 )
      goto LABEL_150;
    v76 = v92 - 1;
  }
  v77 = v76 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v18 = v75 + 16LL * v77;
  v16 = *(_QWORD *)v18;
  if ( v9 != *(_QWORD *)v18 )
  {
    v111 = 1;
    v82 = 0;
    while ( v16 != -4096 )
    {
      if ( !v82 && v16 == -8192 )
        v82 = v18;
      a6 = (unsigned int)(v111 + 1);
      v77 = v76 & (v111 + v77);
      v18 = v75 + 16LL * v77;
      v16 = *(_QWORD *)v18;
      if ( v9 == *(_QWORD *)v18 )
        goto LABEL_97;
      ++v111;
    }
    goto LABEL_103;
  }
LABEL_97:
  v17 = *(_DWORD *)(v7 + 16);
LABEL_11:
  *(_DWORD *)(v7 + 16) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *(_QWORD *)v18 != -4096 )
    --*(_DWORD *)(v7 + 20);
  *(_DWORD *)(v18 + 8) = 0;
  *(_QWORD *)v18 = v9;
  *(_DWORD *)(v18 + 8) = *(_DWORD *)(v7 + 160);
  v21 = *(unsigned int *)(v7 + 160);
  v22 = v21;
  if ( *(_DWORD *)(v7 + 164) <= (unsigned int)v21 )
  {
    v18 = v7 + 152;
    v131 = v7 + 168;
    v23 = sub_C8D7D0(v7 + 152, v7 + 168, 0, 0x10u, v135, a6);
    v16 = 16LL * *(unsigned int *)(v7 + 160);
    v61 = (_QWORD *)(v16 + v23);
    if ( v16 + v23 )
    {
      v62 = *v8;
      v61[1] = 0;
      *v61 = v62;
      v16 = 16LL * *(unsigned int *)(v7 + 160);
    }
    v63 = *(_QWORD *)(v7 + 152);
    v64 = v63 + v16;
    if ( v63 != v63 + v16 )
    {
      v65 = (_QWORD *)(v63 + 8);
      v16 += v23;
      v66 = (_QWORD *)v23;
      do
      {
        if ( v66 )
        {
          *v66 = *(v65 - 1);
          v66[1] = *v65;
          *v65 = 0;
        }
        v66 += 2;
        v65 += 2;
      }
      while ( v66 != (_QWORD *)v16 );
      v67 = *(_QWORD *)(v7 + 152);
      v68 = 16LL * *(unsigned int *)(v7 + 160);
      v64 = v67 + v68;
      if ( v67 != v67 + v68 )
      {
        v123 = v8;
        v117 = v7;
        v69 = v67 + v68;
        v70 = v67;
        do
        {
          v71 = *(_QWORD *)(v69 - 8);
          v69 -= 16;
          if ( v71 )
          {
            v72 = *(_QWORD *)(v71 + 8);
            if ( v72 != v71 + 24 )
              _libc_free(v72);
            v16 = 72;
            v18 = v71;
            j_j___libc_free_0(v71);
          }
        }
        while ( v70 != v69 );
        v7 = v117;
        v8 = v123;
        v64 = *(_QWORD *)(v117 + 152);
      }
    }
    v73 = v135[0];
    if ( v131 != v64 )
    {
      v18 = v64;
      v132 = v135[0];
      _libc_free(v64);
      v73 = v132;
    }
    *(_DWORD *)(v7 + 164) = v73;
    v74 = *(_DWORD *)(v7 + 160);
    *(_QWORD *)(v7 + 152) = v23;
    v26 = (unsigned int)(v74 + 1);
    *(_DWORD *)(v7 + 160) = v26;
  }
  else
  {
    v23 = *(_QWORD *)(v7 + 152);
    v24 = (_QWORD *)(v23 + 16 * v21);
    if ( v24 )
    {
      v25 = *v8;
      v24[1] = 0;
      *v24 = v25;
      v22 = *(_DWORD *)(v7 + 160);
      v23 = *(_QWORD *)(v7 + 152);
    }
    v26 = (unsigned int)(v22 + 1);
    *(_DWORD *)(v7 + 160) = v26;
  }
  v27 = *a2;
  *a2 = 0;
  v28 = v23 + 16 * v26 - 16;
  v29 = *(_QWORD *)(v28 + 8);
  *(_QWORD *)(v28 + 8) = v27;
  if ( v29 )
  {
    v30 = *(_QWORD *)(v29 + 8);
    if ( v30 != v29 + 24 )
      _libc_free(v30);
    v16 = 72;
    v18 = v29;
    j_j___libc_free_0(v29);
  }
  v31 = (unsigned __int64 *)*v8;
  v32 = 32LL * (*(_DWORD *)(*v8 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(*v8 + 7) & 0x40) != 0 )
  {
    v33 = (unsigned __int64 *)*(v31 - 1);
    v31 = &v33[(unsigned __int64)v32 / 8];
  }
  else
  {
    v33 = &v31[v32 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v34 = v33;
  v130 = (__int64)(v8 + 1);
  if ( v33 != v31 )
  {
    v35 = v7;
    do
    {
      v44 = (_BYTE *)*v34;
      if ( *(_BYTE *)*v34 <= 0x1Cu )
        goto LABEL_34;
      v45 = *(_QWORD *)(v35 + 296);
      if ( !*(_QWORD *)(v45 + 16) )
        sub_4263D6(v18, v16, v27);
      v16 = *v34;
      v18 = *(_QWORD *)(v35 + 296);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(v45 + 24))(v18, *v34) )
        goto LABEL_34;
      v18 = *(_QWORD *)(v35 + 304);
      v16 = (unsigned __int64)v44;
      if ( !(unsigned __int8)sub_24F5180(v18, (__int64)v44, a4) )
        goto LABEL_34;
      v18 = *(_BYTE *)(v35 + 16) & 1;
      if ( (*(_BYTE *)(v35 + 16) & 1) != 0 )
      {
        v36 = v35 + 24;
        v37 = 7;
      }
      else
      {
        v47 = *(unsigned int *)(v35 + 32);
        v36 = *(_QWORD *)(v35 + 24);
        if ( !(_DWORD)v47 )
          goto LABEL_65;
        v37 = v47 - 1;
      }
      v16 = v37 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v38 = v36 + 16 * v16;
      v39 = *(_BYTE **)v38;
      if ( v44 == *(_BYTE **)v38 )
        goto LABEL_27;
      v59 = 1;
      while ( v39 != (_BYTE *)-4096LL )
      {
        v101 = v59 + 1;
        v16 = v37 & (unsigned int)(v59 + v16);
        v38 = v36 + 16LL * (unsigned int)v16;
        v39 = *(_BYTE **)v38;
        if ( v44 == *(_BYTE **)v38 )
          goto LABEL_27;
        v59 = v101;
      }
      if ( (_BYTE)v18 )
      {
        v58 = 128;
        goto LABEL_66;
      }
      v47 = *(unsigned int *)(v35 + 32);
LABEL_65:
      v58 = 16 * v47;
LABEL_66:
      v38 = v36 + v58;
LABEL_27:
      v40 = 128;
      if ( !(_BYTE)v18 )
        v40 = 16LL * *(unsigned int *)(v35 + 32);
      if ( v38 == v36 + v40
        || (v41 = *(_QWORD *)(v35 + 152) + 16LL * *(unsigned int *)(v38 + 8),
            v41 == *(_QWORD *)(v35 + 152) + 16LL * *(unsigned int *)(v35 + 160)) )
      {
        v48 = (_QWORD **)a3[2];
        v49 = (_QWORD **)a3[4];
        v18 = a3[5];
        v16 = a3[6];
LABEL_43:
        if ( (_QWORD **)v16 != v48 )
        {
          while ( 1 )
          {
            v50 = *v48;
            if ( (_BYTE *)**v48 == v44 )
              break;
            if ( v49 != ++v48 )
              goto LABEL_43;
            v48 = *(_QWORD ***)(v18 + 8);
            v18 += 8;
            v49 = v48 + 64;
            if ( (_QWORD **)v16 == v48 )
              goto LABEL_47;
          }
          v56 = *((unsigned int *)v8 + 4);
          if ( v56 + 1 > (unsigned __int64)*((unsigned int *)v8 + 5) )
          {
            v18 = (__int64)(v8 + 1);
            v16 = (unsigned __int64)(v8 + 3);
            v126 = v50;
            sub_C8D5F0(v130, v8 + 3, v56 + 1, 8u, v46, (__int64)v50);
            v56 = *((unsigned int *)v8 + 4);
            v50 = v126;
          }
          v27 = (__int64 *)v8[1];
          v27[v56] = (__int64)v50;
          ++*((_DWORD *)v8 + 4);
          goto LABEL_34;
        }
LABEL_47:
        v51 = (_QWORD *)sub_22077B0(0x48u);
        v53 = (unsigned __int64)v51;
        if ( v51 )
        {
          *v51 = v44;
          v51[1] = v51 + 3;
          v51[2] = 0x600000000LL;
        }
        v54 = *((unsigned int *)v8 + 4);
        if ( v54 + 1 > (unsigned __int64)*((unsigned int *)v8 + 5) )
        {
          v16 = (unsigned __int64)(v8 + 3);
          v125 = v53;
          sub_C8D5F0(v130, v8 + 3, v54 + 1, 8u, v52, v53);
          v54 = *((unsigned int *)v8 + 4);
          v53 = v125;
        }
        *(_QWORD *)(v8[1] + 8 * v54) = v53;
        ++*((_DWORD *)v8 + 4);
        v18 = a3[8];
        v55 = (unsigned __int64 *)a3[6];
        v27 = (__int64 *)(v18 - 8);
        if ( v55 == (unsigned __int64 *)(v18 - 8) )
        {
          v83 = (char *)a3[9];
          v16 = a3[5];
          v84 = &v83[-v16];
          v85 = (__int64)&v83[-v16] >> 3;
          if ( ((__int64)(a3[4] - a3[2]) >> 3) + ((v85 - 1) << 6) + ((__int64)((__int64)v55 - a3[7]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
          v86 = *a3;
          v87 = a3[1];
          if ( v87 - ((__int64)&v83[-*a3] >> 3) <= 1 )
          {
            v94 = v85 + 2;
            if ( v87 <= 2 * v94 )
            {
              v102 = 1;
              if ( v87 )
                v102 = a3[1];
              v103 = v87 + v102 + 2;
              if ( v103 > 0xFFFFFFFFFFFFFFFLL )
                sub_4261EA(v86, v16, v27);
              v119 = v94;
              v112 = a3[9] - v16;
              v114 = v53;
              v104 = sub_22077B0(8 * v103);
              v105 = v114;
              v106 = v112;
              v128 = v104;
              v107 = (const void *)a3[5];
              v108 = (void *)(v104 + 8 * ((v103 - v119) >> 1));
              v109 = a3[9] + 8;
              if ( (const void *)v109 != v107 )
              {
                v110 = memmove(v108, v107, v109 - (_QWORD)v107);
                v106 = v112;
                v105 = v114;
                v108 = v110;
              }
              v113 = v108;
              v115 = (char *)v106;
              v120 = v105;
              v16 = 8 * a3[1];
              j_j___libc_free_0(*a3);
              v84 = v115;
              v53 = v120;
              *a3 = v128;
              v95 = (unsigned __int64)v113;
              a3[1] = v103;
            }
            else
            {
              v95 = v86 + 8 * ((v87 - v94) >> 1);
              v96 = v83 + 8;
              v97 = (size_t)&v83[-v16 + 8];
              if ( v16 <= v95 )
              {
                if ( (char *)v16 != v96 )
                {
                  v116 = v53;
                  v121 = a3[9] - v16;
                  v129 = (_QWORD *)v95;
                  memmove((void *)v95, (const void *)v16, v97);
                  v95 = (unsigned __int64)v129;
                  v84 = (char *)v121;
                  v53 = v116;
                }
              }
              else if ( (char *)v16 != v96 )
              {
                v118 = a3[9] - v16;
                v127 = v53;
                v98 = memmove((void *)v95, (const void *)v16, v97);
                v53 = v127;
                v84 = (char *)v118;
                v95 = (unsigned __int64)v98;
              }
            }
            v83 = &v84[v95];
            a3[5] = v95;
            v99 = *(_QWORD *)v95;
            a3[9] = (unsigned __int64)&v84[v95];
            a3[3] = v99;
            a3[4] = v99 + 512;
            v100 = *(_QWORD *)&v84[v95];
            a3[7] = v100;
            a3[8] = v100 + 512;
          }
          v18 = 512;
          v124 = v53;
          v88 = sub_22077B0(0x200u);
          v53 = v124;
          *((_QWORD *)v83 + 1) = v88;
          v89 = (unsigned __int64 *)a3[6];
          if ( v89 )
          {
            *v89 = v124;
            v53 = 0;
          }
          v90 = (unsigned __int64 *)(a3[9] + 8);
          a3[9] = (unsigned __int64)v90;
          v91 = *v90;
          v27 = (__int64 *)(*v90 + 512);
          a3[7] = v91;
          a3[8] = (unsigned __int64)v27;
          a3[6] = v91;
          goto LABEL_72;
        }
        if ( !v55 )
        {
          a3[6] = 8;
LABEL_72:
          if ( v53 )
          {
            v60 = *(_QWORD *)(v53 + 8);
            if ( v60 != v53 + 24 )
            {
              v122 = v53;
              _libc_free(v60);
              v53 = v122;
            }
            v16 = 72;
            v18 = v53;
            j_j___libc_free_0(v53);
          }
          goto LABEL_34;
        }
        *v55 = v53;
        a3[6] += 8LL;
      }
      else
      {
        v42 = *(_QWORD *)(v41 + 8);
        v43 = *((unsigned int *)v8 + 4);
        if ( v43 + 1 > (unsigned __int64)*((unsigned int *)v8 + 5) )
        {
          v18 = (__int64)(v8 + 1);
          v16 = (unsigned __int64)(v8 + 3);
          sub_C8D5F0(v130, v8 + 3, v43 + 1, 8u, v46, v36);
          v43 = *((unsigned int *)v8 + 4);
        }
        v27 = (__int64 *)v8[1];
        v27[v43] = v42;
        ++*((_DWORD *)v8 + 4);
      }
LABEL_34:
      v34 += 4;
    }
    while ( v31 != v34 );
  }
}
