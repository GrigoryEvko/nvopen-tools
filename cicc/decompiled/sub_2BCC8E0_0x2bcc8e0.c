// Function: sub_2BCC8E0
// Address: 0x2bcc8e0
//
__int64 __fastcall sub_2BCC8E0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  unsigned int *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // rsi
  int v14; // eax
  unsigned __int64 v15; // rdx
  unsigned int *v16; // rbx
  __int64 v17; // rsi
  unsigned int *v18; // r12
  unsigned int *v19; // rbx
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rdi
  unsigned int *v22; // rsi
  unsigned int *v23; // rax
  unsigned int *v24; // rax
  __int64 v25; // rax
  __int64 v26; // r8
  unsigned int *v27; // r14
  __int64 v28; // rax
  unsigned int *v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // rdx
  unsigned int v32; // eax
  unsigned int v33; // eax
  __int64 v34; // r13
  __int64 *v35; // r12
  __int64 v36; // r9
  int v37; // ecx
  int v38; // esi
  __int64 v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // rax
  bool v42; // al
  bool v43; // r9
  unsigned int v44; // eax
  unsigned int v45; // r13d
  __int64 v46; // r8
  __int64 v47; // r9
  unsigned int v48; // r12d
  __int64 v49; // r14
  __int64 *v50; // r12
  __int64 *v51; // r15
  __int64 v52; // rdx
  unsigned int v53; // ecx
  __int64 v54; // rsi
  char *v55; // rax
  char *v56; // rdi
  unsigned int v57; // ecx
  unsigned int v58; // eax
  __int64 *v59; // rdx
  unsigned __int64 v60; // r12
  __int64 v61; // rax
  unsigned int *v62; // rbx
  __int64 v63; // rsi
  unsigned int *v64; // r13
  unsigned int v65; // r14d
  unsigned __int64 v66; // rbx
  unsigned __int64 v67; // rdi
  __int64 v69; // rax
  unsigned int v70; // r14d
  unsigned int v71; // r14d
  unsigned int v72; // eax
  __int64 v73; // rdx
  __int64 v74; // r13
  __int64 v75; // r12
  __int64 v76; // rax
  unsigned int *v77; // rax
  unsigned int *v78; // r12
  __int64 v79; // rdx
  unsigned int v80; // r8d
  __int64 v81; // rsi
  __int64 v82; // rcx
  unsigned int *v83; // rbx
  unsigned __int64 v84; // r14
  unsigned __int64 v85; // rdi
  int v86; // ebx
  unsigned int *v88; // [rsp+28h] [rbp-248h]
  unsigned int v89; // [rsp+34h] [rbp-23Ch]
  __int64 *v91; // [rsp+40h] [rbp-230h]
  __int64 *v92; // [rsp+48h] [rbp-228h]
  __int64 v93; // [rsp+58h] [rbp-218h]
  int v94; // [rsp+60h] [rbp-210h]
  char v95; // [rsp+60h] [rbp-210h]
  unsigned int v96; // [rsp+64h] [rbp-20Ch]
  unsigned int v97; // [rsp+68h] [rbp-208h]
  __int64 v98; // [rsp+68h] [rbp-208h]
  __int64 v99; // [rsp+70h] [rbp-200h]
  __int64 *v100; // [rsp+78h] [rbp-1F8h]
  _QWORD v101[2]; // [rsp+80h] [rbp-1F0h] BYREF
  unsigned __int8 v102; // [rsp+97h] [rbp-1D9h] BYREF
  unsigned int v103; // [rsp+98h] [rbp-1D8h] BYREF
  unsigned int v104; // [rsp+9Ch] [rbp-1D4h] BYREF
  unsigned __int64 v105; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 v106; // [rsp+A8h] [rbp-1C8h] BYREF
  _QWORD v107[6]; // [rsp+B0h] [rbp-1C0h] BYREF
  char v108[8]; // [rsp+E0h] [rbp-190h] BYREF
  __int64 v109; // [rsp+E8h] [rbp-188h] BYREF
  unsigned __int64 v110; // [rsp+F0h] [rbp-180h]
  __int64 *v111; // [rsp+F8h] [rbp-178h]
  __int64 *v112; // [rsp+100h] [rbp-170h]
  __int64 v113; // [rsp+108h] [rbp-168h]
  void *s; // [rsp+110h] [rbp-160h] BYREF
  __int64 v115; // [rsp+118h] [rbp-158h]
  _DWORD v116[12]; // [rsp+120h] [rbp-150h] BYREF
  unsigned int *v117; // [rsp+150h] [rbp-120h] BYREF
  __int64 v118; // [rsp+158h] [rbp-118h]
  _BYTE v119[64]; // [rsp+160h] [rbp-110h] BYREF
  __int64 v120; // [rsp+1A0h] [rbp-D0h] BYREF
  char *v121; // [rsp+1A8h] [rbp-C8h]
  __int64 v122; // [rsp+1B0h] [rbp-C0h]
  int v123; // [rsp+1B8h] [rbp-B8h]
  char v124; // [rsp+1BCh] [rbp-B4h]
  char v125; // [rsp+1C0h] [rbp-B0h] BYREF

  v7 = (unsigned int *)v119;
  v121 = &v125;
  v107[0] = v101;
  v107[5] = &v102;
  v118 = 0x100000000LL;
  v101[0] = a2;
  v101[1] = a3;
  v120 = 0;
  v122 = 16;
  v123 = 0;
  v124 = 1;
  v102 = 0;
  v107[1] = a5;
  v107[2] = a4;
  v107[3] = a1;
  v107[4] = &v120;
  v117 = (unsigned int *)v119;
  v91 = &a2[a3];
  if ( a2 == v91 )
  {
    v65 = 0;
    goto LABEL_107;
  }
  v8 = 0;
  v100 = a2;
  v96 = 0;
  v93 = 0;
  while ( 2 )
  {
    v12 = *v100;
    v13 = *(_QWORD *)(a4 + 1984);
    v14 = *(_DWORD *)(a4 + 2000);
    if ( v14 )
    {
      v9 = (unsigned int)(v14 - 1);
      v10 = v9 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v11 = *(_QWORD *)(v13 + 8LL * v10);
      if ( v12 == v11 )
        goto LABEL_4;
      a5 = 1;
      while ( v11 != -4096 )
      {
        a6 = (unsigned int)(a5 + 1);
        v10 = v9 & (a5 + v10);
        v11 = *(_QWORD *)(v13 + 8LL * v10);
        if ( v12 == v11 )
          goto LABEL_4;
        a5 = (unsigned int)a6;
      }
    }
    v15 = (unsigned int)v8;
    v16 = &v7[14 * (unsigned int)v8];
    if ( v93 )
    {
      if ( *(_QWORD *)(*(_QWORD *)(v12 - 64) + 8LL) != v93 )
      {
        if ( v7 != v16 )
        {
          do
          {
            v17 = (__int64)(v7 + 2);
            v7 += 14;
            sub_2BCAEC0((__int64)v107, v17, a7, v15, v8, a5, a6);
          }
          while ( v7 != v16 );
          v18 = v117;
          v19 = &v117[14 * (unsigned int)v118];
          if ( v117 != v19 )
          {
            do
            {
              v20 = *((_QWORD *)v19 - 4);
              v19 -= 14;
              while ( v20 )
              {
                sub_2B11860(*(_QWORD *)(v20 + 24));
                v21 = v20;
                v20 = *(_QWORD *)(v20 + 16);
                j_j___libc_free_0(v21);
              }
            }
            while ( v18 != v19 );
          }
          v12 = *v100;
        }
        LODWORD(v118) = 0;
        v15 = 0;
        LODWORD(v8) = 0;
        v93 = *(_QWORD *)(*(_QWORD *)(v12 - 64) + 8LL);
        v103 = v96;
        goto LABEL_16;
      }
    }
    else
    {
      v93 = *(_QWORD *)(*(_QWORD *)(v12 - 64) + 8LL);
    }
    v103 = v96;
    if ( v7 == v16 )
    {
LABEL_16:
      if ( v15 >= HIDWORD(v118) )
      {
        v74 = sub_C8D7D0((__int64)&v117, (__int64)v119, 0, 0x38u, (unsigned __int64 *)&s, a6);
        v75 = 14LL * (unsigned int)v118;
        v76 = v75 * 4 + v74;
        if ( v75 * 4 + v74 )
        {
          *(_DWORD *)v76 = 0;
          *(_DWORD *)(v76 + 16) = 0;
          *(_QWORD *)(v76 + 24) = 0;
          *(_QWORD *)(v76 + 32) = v76 + 16;
          *(_QWORD *)(v76 + 40) = v76 + 16;
          *(_QWORD *)(v76 + 48) = 0;
          v75 = 14LL * (unsigned int)v118;
        }
        v77 = v117;
        v78 = &v117[v75];
        if ( v117 != v78 )
        {
          v79 = v74;
          do
          {
            if ( v79 )
            {
              *(_DWORD *)v79 = *v77;
              v81 = *((_QWORD *)v77 + 3);
              v82 = v79 + 16;
              if ( v81 )
              {
                v80 = v77[4];
                *(_QWORD *)(v79 + 24) = v81;
                *(_DWORD *)(v79 + 16) = v80;
                *(_QWORD *)(v79 + 32) = *((_QWORD *)v77 + 4);
                *(_QWORD *)(v79 + 40) = *((_QWORD *)v77 + 5);
                *(_QWORD *)(v81 + 8) = v82;
                *(_QWORD *)(v79 + 48) = *((_QWORD *)v77 + 6);
                *((_QWORD *)v77 + 3) = 0;
                *((_QWORD *)v77 + 4) = v77 + 4;
                *((_QWORD *)v77 + 5) = v77 + 4;
                *((_QWORD *)v77 + 6) = 0;
              }
              else
              {
                *(_DWORD *)(v79 + 16) = 0;
                *(_QWORD *)(v79 + 24) = 0;
                *(_QWORD *)(v79 + 32) = v82;
                *(_QWORD *)(v79 + 40) = v82;
                *(_QWORD *)(v79 + 48) = 0;
              }
            }
            v77 += 14;
            v79 += 56;
          }
          while ( v78 != v77 );
          v78 = v117;
          v83 = &v117[14 * (unsigned int)v118];
          if ( v83 != v117 )
          {
            do
            {
              v84 = *((_QWORD *)v83 - 4);
              v83 -= 14;
              while ( v84 )
              {
                sub_2B11860(*(_QWORD *)(v84 + 24));
                v85 = v84;
                v84 = *(_QWORD *)(v84 + 16);
                j_j___libc_free_0(v85);
              }
            }
            while ( v83 != v78 );
            v78 = v117;
          }
        }
        v86 = (int)s;
        if ( v78 != (unsigned int *)v119 )
          _libc_free((unsigned __int64)v78);
        v117 = (unsigned int *)v74;
        HIDWORD(v118) = v86;
        LODWORD(v118) = v118 + 1;
        v24 = (unsigned int *)(v74 + 56LL * (unsigned int)v118 - 56);
      }
      else
      {
        v22 = v117;
        v23 = &v117[14 * v15];
        if ( v23 )
        {
          v23[4] = 0;
          *((_QWORD *)v23 + 3) = 0;
          *((_QWORD *)v23 + 4) = v23 + 4;
          *((_QWORD *)v23 + 5) = v23 + 4;
          *((_QWORD *)v23 + 6) = 0;
          *v23 = 0;
          LODWORD(v8) = v118;
          v22 = v117;
        }
        LODWORD(v118) = v8 + 1;
        v24 = &v22[14 * (unsigned int)(v8 + 1) - 14];
      }
      *v24 = v103;
      LODWORD(s) = 0;
      sub_2B08850((__int64)(v24 + 2), &v103, (int *)&s);
      goto LABEL_32;
    }
    while ( 1 )
    {
      v25 = *(_QWORD *)(v101[0] + 8LL * *v7);
      v106 = sub_D35010(
               *(_QWORD *)(*(_QWORD *)(v25 - 64) + 8LL),
               *(_QWORD *)(v25 - 32),
               *(_QWORD *)(*(_QWORD *)(v12 - 64) + 8LL),
               *(_QWORD *)(v12 - 32),
               a1[8],
               *a1,
               1,
               1);
      if ( BYTE4(v106) )
        break;
      v7 += 14;
      if ( v16 == v7 )
      {
        v15 = (unsigned int)v118;
        LODWORD(v8) = v118;
        goto LABEL_16;
      }
    }
    v27 = v7 + 4;
    v99 = (__int64)(v7 + 2);
    v28 = *((_QWORD *)v7 + 3);
    if ( !v28 )
      goto LABEL_31;
    v29 = v7 + 4;
    do
    {
      while ( 1 )
      {
        v30 = *(_QWORD *)(v28 + 16);
        v31 = *(_QWORD *)(v28 + 24);
        if ( (int)v106 <= *(_DWORD *)(v28 + 36) )
          break;
        v28 = *(_QWORD *)(v28 + 24);
        if ( !v31 )
          goto LABEL_29;
      }
      v29 = (unsigned int *)v28;
      v28 = *(_QWORD *)(v28 + 16);
    }
    while ( v30 );
LABEL_29:
    if ( v29 == v27 || (int)v106 < (int)v29[9] )
    {
LABEL_31:
      sub_2B08850(v99, &v103, (int *)&v106);
      goto LABEL_32;
    }
    sub_2BCAEC0((__int64)v107, v99, a7, v31, v30, v26, a6);
    v32 = v29[8];
    LODWORD(v109) = 0;
    v110 = 0;
    v97 = v32;
    v33 = v29[9];
    v112 = &v109;
    v113 = 0;
    v34 = *((_QWORD *)v7 + 4);
    v89 = v33;
    v111 = &v109;
    if ( v27 == (unsigned int *)v34 )
      goto LABEL_50;
    v88 = v7;
    v35 = &v109;
    do
    {
      while ( v97 >= *(_DWORD *)(v34 + 32) )
      {
        v34 = sub_220EF30(v34);
        if ( v27 == (unsigned int *)v34 )
          goto LABEL_49;
      }
      v36 = v34 + 32;
      if ( v35 == &v109 )
      {
        if ( v113 )
        {
          v40 = v112;
          if ( *((_DWORD *)v112 + 9) < *(_DWORD *)(v34 + 36) )
            goto LABEL_95;
        }
      }
      else
      {
        v37 = *(_DWORD *)(v34 + 36);
        v38 = *((_DWORD *)v35 + 9);
        v94 = v37;
        if ( v37 >= v38 )
        {
          if ( v37 <= v38 )
            goto LABEL_48;
          if ( v112 == v35 )
            goto LABEL_94;
          v61 = sub_220EEE0((__int64)v35);
          v36 = v34 + 32;
          v40 = (__int64 *)v61;
          if ( v94 < *(_DWORD *)(v61 + 36) )
          {
            if ( !v35[3] )
            {
LABEL_94:
              v40 = v35;
              v43 = v38 > *(_DWORD *)(v34 + 36);
              goto LABEL_47;
            }
LABEL_46:
            v43 = 1;
LABEL_47:
            v95 = v43;
            v92 = v40;
            v35 = (__int64 *)sub_22077B0(0x28u);
            v35[4] = *(_QWORD *)(v34 + 32);
            sub_220F040(v95, (__int64)v35, v92, &v109);
            ++v113;
            goto LABEL_48;
          }
        }
        else
        {
          if ( v111 == v35 )
            goto LABEL_42;
          v39 = sub_220EF80((__int64)v35);
          v36 = v34 + 32;
          v40 = (__int64 *)v39;
          if ( v94 > *(_DWORD *)(v39 + 36) )
          {
            if ( *(_QWORD *)(v39 + 24) )
            {
LABEL_42:
              v41 = (__int64)v35;
              goto LABEL_43;
            }
LABEL_95:
            v42 = 0;
LABEL_44:
            if ( v40 != &v109 && !v42 )
            {
              v43 = *((_DWORD *)v40 + 9) > *(_DWORD *)(v34 + 36);
              goto LABEL_47;
            }
            goto LABEL_46;
          }
        }
      }
      v41 = sub_2B086F0((__int64)v108, v36);
      v35 = (__int64 *)v41;
      if ( v59 )
      {
        v35 = v59;
LABEL_43:
        v40 = v35;
        v42 = v41 != 0;
        goto LABEL_44;
      }
LABEL_48:
      v35 = (__int64 *)sub_220EF30((__int64)v35);
      v34 = sub_220EF30(v34);
    }
    while ( v27 != (unsigned int *)v34 );
LABEL_49:
    v7 = v88;
LABEL_50:
    sub_2B11860(*((_QWORD *)v7 + 3));
    v44 = v103;
    *((_QWORD *)v7 + 4) = v27;
    *((_QWORD *)v7 + 5) = v27;
    *((_QWORD *)v7 + 3) = 0;
    *((_QWORD *)v7 + 6) = 0;
    *v7 = v44;
    LODWORD(s) = 0;
    sub_2B08850(v99, &v103, (int *)&s);
    v45 = v97 + 1;
    sub_B48880((__int64 *)&v105, v103 - (v97 + 1), 0);
    s = v116;
    v48 = v103 - (v97 + 1);
    v115 = 0xC00000000LL;
    if ( v48 > 0xC )
    {
      sub_C8D5F0((__int64)&s, v116, v48, 4u, v46, v47);
      memset(s, 0, 4LL * v48);
      LODWORD(v115) = v48;
    }
    else
    {
      if ( v48 )
      {
        v49 = 4LL * v48;
        if ( v49 )
        {
          if ( (unsigned int)v49 >= 8 )
          {
            v69 = (unsigned int)v49;
            v70 = v49 - 1;
            *(_QWORD *)((char *)&v116[-2] + v69) = 0;
            if ( v70 >= 8 )
            {
              v71 = v70 & 0xFFFFFFF8;
              v72 = 0;
              do
              {
                v73 = v72;
                v72 += 8;
                *(_QWORD *)((char *)v116 + v73) = 0;
              }
              while ( v72 < v71 );
            }
          }
          else if ( ((4 * (_BYTE)v48) & 4) != 0 )
          {
            v116[0] = 0;
            *(_DWORD *)((char *)&v116[-1] + (unsigned int)v49) = 0;
          }
          else if ( (_DWORD)v49 )
          {
            LOBYTE(v116[0]) = 0;
          }
        }
      }
      LODWORD(v115) = v48;
    }
    v50 = v111;
    v51 = &v109;
    if ( v111 == &v109 )
      goto LABEL_64;
    while ( 2 )
    {
      v52 = sub_220EFE0((__int64)v51);
      v53 = *(_DWORD *)(v52 + 32);
      v54 = *(_QWORD *)(v101[0] + 8LL * v53);
      if ( v124 )
      {
        v55 = v121;
        v56 = &v121[8 * HIDWORD(v122)];
        if ( v121 == v56 )
          goto LABEL_77;
        while ( v54 != *(_QWORD *)v55 )
        {
          v55 += 8;
          if ( v56 == v55 )
            goto LABEL_77;
        }
LABEL_64:
        v104 = v45;
        if ( v45 >= v103 )
          goto LABEL_81;
LABEL_65:
        v57 = v45;
        while ( 2 )
        {
          v58 = v57 - v45;
          if ( (v105 & 1) != 0 )
          {
            if ( ((((v105 >> 1) & ~(-1LL << (v105 >> 58))) >> v58) & 1) != 0 )
              goto LABEL_70;
          }
          else if ( ((*(_QWORD *)(*(_QWORD *)v105 + 8LL * (v58 >> 6)) >> v58) & 1) != 0 )
          {
LABEL_70:
            sub_2B08850(v99, &v104, (int *)s + v58);
          }
          v57 = v104 + 1;
          v104 = v57;
          if ( v103 <= v57 )
            goto LABEL_81;
          continue;
        }
      }
      v98 = v52;
      if ( sub_C8CA60((__int64)&v120, v54) )
        goto LABEL_64;
      v52 = v98;
      v53 = *(_DWORD *)(v98 + 32);
LABEL_77:
      if ( (v105 & 1) != 0 )
        v105 = 2
             * ((v105 >> 58 << 57)
              | ~(-1LL << (v105 >> 58))
              & (~(-1LL << (v105 >> 58)) & (v105 >> 1) | (1LL << ((unsigned __int8)v53 - (unsigned __int8)v45))))
             + 1;
      else
        *(_QWORD *)(*(_QWORD *)v105 + 8LL * ((v53 - v45) >> 6)) |= 1LL << ((unsigned __int8)v53 - (unsigned __int8)v45);
      *((_DWORD *)s + v53 - v45) = *(_DWORD *)(v52 + 36) - v89;
      v51 = (__int64 *)sub_220EFE0((__int64)v51);
      if ( v50 != v51 )
        continue;
      break;
    }
    v104 = v45;
    if ( v45 < v103 )
      goto LABEL_65;
LABEL_81:
    if ( s != v116 )
      _libc_free((unsigned __int64)s);
    v60 = v105;
    if ( (v105 & 1) == 0 && v105 )
    {
      if ( *(_QWORD *)v105 != v105 + 16 )
        _libc_free(*(_QWORD *)v105);
      j_j___libc_free_0(v60);
    }
    sub_2B11860(v110);
LABEL_32:
    v8 = (unsigned int)v118;
    v7 = v117;
LABEL_4:
    ++v100;
    ++v96;
    if ( v91 != v100 )
      continue;
    break;
  }
  v62 = &v7[14 * v8];
  if ( v62 == v7 )
  {
    v65 = v102;
  }
  else
  {
    do
    {
      v63 = (__int64)(v7 + 2);
      v7 += 14;
      sub_2BCAEC0((__int64)v107, v63, a7, v9, v8, a5, a6);
    }
    while ( v7 != v62 );
    v64 = v117;
    v65 = v102;
    v7 = &v117[14 * (unsigned int)v118];
    if ( v117 != v7 )
    {
      do
      {
        v66 = *((_QWORD *)v7 - 4);
        v7 -= 14;
        while ( v66 )
        {
          sub_2B11860(*(_QWORD *)(v66 + 24));
          v67 = v66;
          v66 = *(_QWORD *)(v66 + 16);
          j_j___libc_free_0(v67);
        }
      }
      while ( v64 != v7 );
      v7 = v117;
    }
  }
  if ( v7 != (unsigned int *)v119 )
    _libc_free((unsigned __int64)v7);
LABEL_107:
  if ( !v124 )
    _libc_free((unsigned __int64)v121);
  return v65;
}
