// Function: sub_A27070
// Address: 0xa27070
//
__int64 __fastcall sub_A27070(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax
  _DWORD *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdi
  volatile signed __int32 *v8; // rax
  unsigned int v9; // r14d
  __int64 v10; // rdi
  volatile signed __int32 *v11; // rax
  volatile signed __int32 *v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdi
  volatile signed __int32 *v15; // rax
  __int64 v16; // r12
  _QWORD *v17; // rbx
  __int64 v18; // r14
  __int64 v19; // rdx
  unsigned int v20; // r14d
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rax
  _QWORD *v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rax
  _QWORD *v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rdi
  _QWORD *v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r12
  __int64 v43; // rdx
  __int64 v44; // r14
  _BYTE *v45; // rcx
  __int64 i; // rax
  __int64 v47; // rdi
  unsigned int v48; // r14d
  int v49; // r12d
  __int64 v50; // rax
  int v51; // esi
  __int64 v52; // rdi
  __int64 v53; // r8
  __int64 v54; // rdx
  int v55; // esi
  unsigned int v56; // ecx
  __int64 *v57; // rax
  __int64 v58; // r8
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  _QWORD *v61; // rbx
  __int64 v62; // r12
  unsigned int v63; // eax
  __int64 v64; // r12
  __int64 v65; // r14
  _QWORD *v66; // rbx
  unsigned int v67; // eax
  __int64 v68; // r12
  __int64 v69; // r15
  __int64 v70; // r13
  int v71; // r14d
  __int64 v72; // r13
  int v73; // eax
  __int64 v74; // rbx
  char *v75; // rax
  void *v76; // r8
  char *v77; // r12
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rdx
  int v81; // r9d
  __int64 v82; // r12
  _QWORD *v83; // r13
  __int64 v84; // rbx
  __int64 v85; // r13
  unsigned __int64 v86; // rbx
  unsigned int v87; // r14d
  int v88; // edx
  __int64 *v89; // rdx
  __int64 *v90; // rsi
  __int64 v91; // rcx
  __int64 v92; // r13
  __int64 v93; // rbx
  __int64 v94; // rcx
  __int64 v95; // r12
  __int64 v96; // r12
  __int64 v97; // rbx
  signed __int64 v98; // rsi
  unsigned __int64 v99; // rdx
  unsigned int v100; // [rsp+0h] [rbp-2F0h]
  void *v101; // [rsp+0h] [rbp-2F0h]
  unsigned int v102; // [rsp+8h] [rbp-2E8h]
  _QWORD *v103; // [rsp+8h] [rbp-2E8h]
  unsigned int v104; // [rsp+8h] [rbp-2E8h]
  __int64 *v105; // [rsp+8h] [rbp-2E8h]
  __int64 v106; // [rsp+20h] [rbp-2D0h]
  __int64 v107; // [rsp+20h] [rbp-2D0h]
  _QWORD *v108; // [rsp+28h] [rbp-2C8h]
  _QWORD *v109; // [rsp+28h] [rbp-2C8h]
  __int64 v110; // [rsp+30h] [rbp-2C0h] BYREF
  volatile signed __int32 *v111; // [rsp+38h] [rbp-2B8h]
  void *v112; // [rsp+40h] [rbp-2B0h] BYREF
  char *v113; // [rsp+48h] [rbp-2A8h]
  char *v114; // [rsp+50h] [rbp-2A0h]
  void *src; // [rsp+60h] [rbp-290h] BYREF
  __int64 *v116; // [rsp+68h] [rbp-288h]
  char *v117; // [rsp+70h] [rbp-280h]
  _BYTE *v118; // [rsp+80h] [rbp-270h] BYREF
  unsigned __int64 v119; // [rsp+88h] [rbp-268h]
  _BYTE v120[32]; // [rsp+90h] [rbp-260h] BYREF
  _BYTE *v121; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v122; // [rsp+B8h] [rbp-238h]
  _BYTE v123[560]; // [rsp+C0h] [rbp-230h] BYREF

  v1 = a1;
  if ( (__int64)(*(_QWORD *)(a1 + 240) - *(_QWORD *)(a1 + 232)) >> 3 <= (unsigned __int64)*(unsigned int *)(a1 + 564) )
  {
    v2 = *(_QWORD *)(a1 + 16);
    v3 = *(_QWORD *)(v2 + 72);
    result = v2 + 72;
    if ( (v3 & 0xFFFFFFFFFFFFFFF8LL) == result )
      return result;
  }
  sub_A19830(*(_QWORD *)a1, 0xFu, 4u);
  v112 = 0;
  v121 = v123;
  v122 = 0x4000000000LL;
  v113 = 0;
  v114 = 0;
  v5 = (_DWORD *)sub_22077B0(128);
  *(_QWORD *)v5 = 0;
  *((_QWORD *)v5 + 15) = 0;
  memset(
    (void *)((unsigned __int64)(v5 + 2) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v5 - (((_DWORD)v5 + 8) & 0xFFFFFFF8) + 128) >> 3));
  if ( v113 - (_BYTE *)v112 > 0 )
  {
    memmove(v5, v112, v113 - (_BYTE *)v112);
  }
  else if ( !v112 )
  {
    goto LABEL_5;
  }
  j_j___libc_free_0(v112, v114 - (_BYTE *)v112);
LABEL_5:
  v112 = v5;
  v113 = (char *)(v5 + 32);
  v114 = (char *)(v5 + 32);
  v5[1] = sub_A237E0((__int64 *)a1);
  *((_DWORD *)v112 + 4) = sub_A23C00((__int64 *)a1);
  sub_A23770(&v110);
  sub_A186C0(v110, 38, 1);
  sub_A186C0(v110, 32, 2);
  sub_A186C0(v110, 32, 2);
  v6 = v110;
  v7 = *(_QWORD *)a1;
  v110 = 0;
  v118 = (_BYTE *)v6;
  v8 = v111;
  v111 = 0;
  v119 = (unsigned __int64)v8;
  v9 = sub_A1AB30(v7, (__int64 *)&v118);
  if ( v119 )
    sub_A191D0((volatile signed __int32 *)v119);
  sub_A23770(&v118);
  v10 = (__int64)v118;
  v11 = (volatile signed __int32 *)v119;
  v118 = 0;
  v12 = v111;
  v119 = 0;
  v110 = v10;
  v111 = v11;
  if ( v12 )
  {
    sub_A191D0(v12);
    if ( v119 )
      sub_A191D0((volatile signed __int32 *)v119);
    v10 = v110;
  }
  sub_A186C0(v10, 39, 1);
  sub_A186C0(v110, 0, 6);
  sub_A186C0(v110, 6, 4);
  v13 = v110;
  v14 = *(_QWORD *)v1;
  v110 = 0;
  v118 = (_BYTE *)v13;
  v15 = v111;
  v111 = 0;
  v119 = (unsigned __int64)v15;
  v102 = sub_A1AB30(v14, (__int64 *)&v118);
  if ( v119 )
    sub_A191D0((volatile signed __int32 *)v119);
  sub_A24AF0(
    (__int64 *)v1,
    (_QWORD *)(*(_QWORD *)(v1 + 232) + 8LL * *(unsigned int *)(v1 + 564)),
    *(unsigned int *)(v1 + 568),
    (__int64)&v121);
  if ( (unsigned int)qword_4F808C8 < ((__int64)(*(_QWORD *)(v1 + 240) - *(_QWORD *)(v1 + 232)) >> 3)
                                   - (*(unsigned int *)(v1 + 568)
                                    + (unsigned __int64)*(unsigned int *)(v1 + 564)) )
  {
    v118 = 0;
    v95 = *(_QWORD *)v1;
    v119 = 0;
    if ( v9 )
    {
      sub_A1B020(v95, v9, (__int64)&v118, 2, 0, 0, 0x26u, 1);
    }
    else
    {
      sub_A17B10(v95, 3u, *(_DWORD *)(v95 + 56));
      sub_A17CC0(v95, 0x26u, 6);
      sub_A17CC0(v95, 2u, 6);
      sub_A17DE0(v95, (unsigned __int64)v118, 6);
      sub_A17DE0(v95, v119, 6);
    }
  }
  v16 = *(_QWORD *)v1;
  v17 = *(_QWORD **)(*(_QWORD *)v1 + 32LL);
  v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 24LL) + 8LL);
  v106 = v18;
  if ( v17 && (unsigned __int8)sub_CB7440(v17) )
  {
    if ( !(unsigned __int8)sub_CB7440(v17) )
      goto LABEL_113;
    v106 = (*(__int64 (__fastcall **)(_QWORD *))(*v17 + 80LL))(v17) + v18 + v17[4] - v17[2];
  }
  v19 = *(unsigned int *)(v1 + 568);
  v20 = *(_DWORD *)(v16 + 48);
  src = 0;
  v21 = *(unsigned int *)(v1 + 564);
  v22 = *(_QWORD *)(v1 + 232);
  v116 = 0;
  v117 = 0;
  v23 = v19 + v21;
  v24 = ((*(_QWORD *)(v1 + 240) - v22) >> 3) - v23;
  if ( v24 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  if ( v24 )
  {
    v74 = 8 * v24;
    v75 = (char *)sub_22077B0(8 * v24);
    v76 = src;
    v77 = v75;
    if ( (char *)v116 - (_BYTE *)src > 0 )
    {
      v101 = src;
      memmove(v75, src, (char *)v116 - (_BYTE *)src);
      v76 = v101;
      v98 = v117 - (_BYTE *)v101;
    }
    else
    {
      if ( !src )
      {
LABEL_82:
        v78 = *(unsigned int *)(v1 + 568);
        v79 = *(unsigned int *)(v1 + 564);
        src = v77;
        v117 = &v77[v74];
        v22 = *(_QWORD *)(v1 + 232);
        v23 = v78 + v79;
        v80 = *(_QWORD *)(v1 + 240);
        v116 = (__int64 *)v77;
        v24 = ((v80 - v22) >> 3) - v23;
        goto LABEL_20;
      }
      v98 = v117 - (_BYTE *)src;
    }
    j_j___libc_free_0(v76, v98);
    goto LABEL_82;
  }
LABEL_20:
  sub_A24040((__int64 *)v1, (__int64 *)(v22 + 8 * v23), v24, (__int64)&v121, (unsigned int **)&v112, (__int64)&src);
  v25 = *(unsigned int *)(v1 + 568);
  if ( (unsigned int)qword_4F808C8 < ((__int64)(*(_QWORD *)(v1 + 240) - *(_QWORD *)(v1 + 232)) >> 3)
                                   - (v25
                                    + *(unsigned int *)(v1 + 564)) )
  {
    v82 = *(_QWORD *)v1;
    v83 = *(_QWORD **)(*(_QWORD *)v1 + 32LL);
    v84 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 24LL) + 8LL);
    if ( !v83 || !(unsigned __int8)sub_CB7440(*(_QWORD *)(*(_QWORD *)v1 + 32LL)) )
    {
LABEL_89:
      v85 = v20 + 8 * v106;
      v86 = *(unsigned int *)(v82 + 48) - v85 + 8 * v84;
      sub_A177B0(v82, v85 - 64, (unsigned __int8)v86);
      v87 = WORD1(v86);
      v88 = BYTE1(v86);
      v86 >>= 32;
      sub_A177B0(v82, v85 - 56, v88);
      sub_A177B0(v82, v85 - 48, (unsigned __int8)v87);
      sub_A177B0(v82, v85 - 40, v87 >> 8);
      sub_A177B0(v82, v85 - 32, (unsigned __int8)v86);
      sub_A177B0(v82, v85 - 24, BYTE1(v86));
      LODWORD(v86) = WORD1(v86);
      sub_A177B0(v82, v85 - 16, (unsigned __int8)v86);
      sub_A177B0(v82, v85 - 8, (unsigned int)v86 >> 8);
      v89 = (__int64 *)src;
      v90 = v116;
      if ( v116 == src )
      {
        v92 = *(_QWORD *)v1;
        v94 = 0;
        if ( !v102 )
        {
          sub_A17B10(v92, 3u, *(_DWORD *)(v92 + 56));
          sub_A17CC0(v92, 0x27u, 6);
          v25 = 0;
          sub_A17CC0(v92, 0, 6);
LABEL_93:
          if ( src != v116 )
            v116 = (__int64 *)src;
          goto LABEL_21;
        }
      }
      else
      {
        do
        {
          v91 = v85;
          v85 = *v89++;
          *(v89 - 1) = v85 - v91;
        }
        while ( v90 != v89 );
        v89 = (__int64 *)src;
        v92 = *(_QWORD *)v1;
        v93 = ((char *)v116 - (_BYTE *)src) >> 3;
        v94 = v93;
        if ( !v102 )
        {
          sub_A17B10(v92, 3u, *(_DWORD *)(v92 + 56));
          sub_A17CC0(v92, 0x27u, 6);
          v25 = (unsigned int)v93;
          sub_A17CC0(v92, v93, 6);
          if ( (_DWORD)v93 )
          {
            v96 = 0;
            v97 = 8LL * (unsigned int)v93;
            do
            {
              v25 = *(_QWORD *)((char *)src + v96);
              v96 += 8;
              sub_A17DE0(v92, v25, 6);
            }
            while ( v97 != v96 );
          }
          goto LABEL_93;
        }
      }
      v25 = v102;
      sub_A1B020(v92, v102, (__int64)v89, v94, 0, 0, 0x27u, 1);
      goto LABEL_93;
    }
    if ( (unsigned __int8)sub_CB7440(v83) )
    {
      v84 += (*(__int64 (__fastcall **)(_QWORD *))(*v83 + 80LL))(v83) + v83[4] - v83[2];
      goto LABEL_89;
    }
LABEL_113:
    BUG();
  }
LABEL_21:
  v26 = *(_QWORD **)(v1 + 16);
  if ( v26 + 9 != (_QWORD *)(v26[9] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v27 = (_QWORD *)sub_22077B0(544);
    v28 = v27;
    if ( v27 )
    {
      v29 = (__int64)(v27 + 2);
      v27[1] = 0x100000001LL;
      *v27 = &unk_49D9900;
      v27[2] = v27 + 4;
      v27[3] = 0x2000000000LL;
      v30 = 0;
    }
    else
    {
      v30 = MEMORY[0x18];
      v29 = 16;
      v99 = MEMORY[0x18] + 1LL;
      if ( v99 > MEMORY[0x1C] )
      {
        sub_C8D5F0(16, 32, v99, 16);
        v30 = MEMORY[0x18];
      }
    }
    v31 = (_QWORD *)(v28[2] + 16 * v30);
    *v31 = 4;
    v31[1] = 1;
    v32 = *((unsigned int *)v28 + 7);
    v33 = (unsigned int)(*((_DWORD *)v28 + 6) + 1);
    *((_DWORD *)v28 + 6) = v33;
    if ( v33 + 1 > v32 )
    {
      sub_C8D5F0(v29, v28 + 4, v33 + 1, 16);
      v33 = *((unsigned int *)v28 + 6);
    }
    v34 = (_QWORD *)(v28[2] + 16 * v33);
    *v34 = 0;
    v34[1] = 6;
    v35 = *((unsigned int *)v28 + 7);
    v36 = (unsigned int)(*((_DWORD *)v28 + 6) + 1);
    *((_DWORD *)v28 + 6) = v36;
    if ( v36 + 1 > v35 )
    {
      sub_C8D5F0(v29, v28 + 4, v36 + 1, 16);
      v36 = *((unsigned int *)v28 + 6);
    }
    v37 = (_QWORD *)(v28[2] + 16 * v36);
    v25 = (unsigned __int64)&v118;
    *v37 = 8;
    v37[1] = 2;
    v38 = *(_QWORD *)v1;
    ++*((_DWORD *)v28 + 6);
    v118 = (_BYTE *)v29;
    v119 = (unsigned __int64)v28;
    v100 = sub_A1AB30(v38, (__int64 *)&v118);
    if ( v119 )
      sub_A191D0((volatile signed __int32 *)v119);
    v26 = *(_QWORD **)(v1 + 16);
    v103 = v26 + 9;
    v39 = (_QWORD *)v26[10];
    if ( v26 + 9 != v39 )
    {
      do
      {
        v40 = sub_B91B20(v39);
        v42 = v41;
        v43 = (unsigned int)v122;
        v44 = v40;
        if ( v42 + (unsigned __int64)(unsigned int)v122 > HIDWORD(v122) )
        {
          sub_C8D5F0(&v121, v123, v42 + (unsigned int)v122, 8);
          v43 = (unsigned int)v122;
        }
        v45 = &v121[8 * v43];
        if ( v42 > 0 )
        {
          for ( i = 0; i != v42; ++i )
            *(_QWORD *)&v45[8 * i] = *(unsigned __int8 *)(v44 + i);
          LODWORD(v43) = v122;
        }
        v47 = *(_QWORD *)v1;
        v48 = 0;
        LODWORD(v122) = v43 + v42;
        sub_A1BFB0(v47, 4u, (__int64)&v121, v100);
        LODWORD(v122) = 0;
        v49 = sub_B91A00(v39);
        if ( v49 )
        {
          do
          {
            v50 = sub_B91A10(v39, v48);
            v51 = *(_DWORD *)(v1 + 304);
            v52 = *(_QWORD *)(v1 + 288);
            v53 = 0xFFFFFFFFLL;
            v54 = v50;
            if ( v51 )
            {
              v55 = v51 - 1;
              v56 = v55 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
              v57 = (__int64 *)(v52 + 16LL * v56);
              v58 = *v57;
              if ( v54 == *v57 )
              {
LABEL_40:
                v53 = (unsigned int)(*((_DWORD *)v57 + 3) - 1);
              }
              else
              {
                v73 = 1;
                while ( v58 != -4096 )
                {
                  v81 = v73 + 1;
                  v56 = v55 & (v73 + v56);
                  v57 = (__int64 *)(v52 + 16LL * v56);
                  v58 = *v57;
                  if ( v54 == *v57 )
                    goto LABEL_40;
                  v73 = v81;
                }
                v53 = 0xFFFFFFFFLL;
              }
            }
            v59 = (unsigned int)v122;
            v60 = (unsigned int)v122 + 1LL;
            if ( v60 > HIDWORD(v122) )
            {
              v107 = v53;
              sub_C8D5F0(&v121, v123, v60, 8);
              v59 = (unsigned int)v122;
              v53 = v107;
            }
            ++v48;
            *(_QWORD *)&v121[8 * v59] = v53;
            LODWORD(v122) = v122 + 1;
          }
          while ( v49 != v48 );
        }
        v25 = 10;
        sub_A1BFB0(*(_QWORD *)v1, 0xAu, (__int64)&v121, 0);
        LODWORD(v122) = 0;
        v39 = (_QWORD *)v39[1];
      }
      while ( v103 != v39 );
      v26 = *(_QWORD **)(v1 + 16);
    }
  }
  v61 = (_QWORD *)v26[4];
  v108 = v26 + 3;
  if ( v26 + 3 != v61 )
  {
    do
    {
      while ( 1 )
      {
        v62 = (__int64)(v61 - 7);
        if ( !v61 )
          v62 = 0;
        if ( (unsigned __int8)sub_B2FC80(v62) && (*(_BYTE *)(v62 + 7) & 0x20) != 0 )
        {
          v118 = v120;
          v119 = 0x400000000LL;
          v63 = sub_A3F3B0(v1 + 24);
          sub_A188E0((__int64)&v118, v63);
          sub_A16D70(v1, (__int64)&v118, v62);
          v64 = *(_QWORD *)v1;
          v65 = 0;
          v104 = v119;
          sub_A17B10(*(_QWORD *)v1, 3u, *(_DWORD *)(*(_QWORD *)v1 + 56LL));
          sub_A17CC0(v64, 0x24u, 6);
          v25 = v104;
          sub_A17CC0(v64, v104, 6);
          if ( v104 )
          {
            do
            {
              v25 = *(_QWORD *)&v118[v65];
              v65 += 8;
              sub_A17DE0(v64, v25, 6);
            }
            while ( v65 != 8LL * v104 );
          }
          if ( v118 != v120 )
            break;
        }
        v61 = (_QWORD *)v61[1];
        if ( v108 == v61 )
          goto LABEL_57;
      }
      _libc_free(v118, v25);
      v61 = (_QWORD *)v61[1];
    }
    while ( v108 != v61 );
LABEL_57:
    v26 = *(_QWORD **)(v1 + 16);
  }
  v66 = (_QWORD *)v26[2];
  v109 = v26 + 1;
  if ( v26 + 1 != v66 )
  {
    v105 = (__int64 *)v1;
    do
    {
      while ( 1 )
      {
        if ( !v66 )
          BUG();
        if ( (*((_BYTE *)v66 - 49) & 0x20) != 0 )
        {
          v118 = v120;
          v119 = 0x400000000LL;
          v67 = sub_A3F3B0(v105 + 3);
          sub_A188E0((__int64)&v118, v67);
          sub_A16D70((__int64)v105, (__int64)&v118, (__int64)(v66 - 7));
          v68 = *v105;
          v69 = 0;
          v70 = (unsigned int)v119;
          v71 = v119;
          sub_A17B10(v68, 3u, *(_DWORD *)(v68 + 56));
          sub_A17CC0(v68, 0x24u, 6);
          v25 = (unsigned int)v70;
          sub_A17CC0(v68, v70, 6);
          v72 = 8 * v70;
          if ( v71 )
          {
            do
            {
              v25 = *(_QWORD *)&v118[v69];
              v69 += 8;
              sub_A17DE0(v68, v25, 6);
            }
            while ( v72 != v69 );
          }
          if ( v118 != v120 )
            break;
        }
        v66 = (_QWORD *)v66[1];
        if ( v109 == v66 )
          goto LABEL_67;
      }
      _libc_free(v118, v25);
      v66 = (_QWORD *)v66[1];
    }
    while ( v109 != v66 );
LABEL_67:
    v1 = (__int64)v105;
  }
  result = sub_A192A0(*(_QWORD *)v1);
  if ( src )
  {
    v25 = v117 - (_BYTE *)src;
    result = j_j___libc_free_0(src, v117 - (_BYTE *)src);
  }
  if ( v111 )
    result = sub_A191D0(v111);
  if ( v112 )
  {
    v25 = v114 - (_BYTE *)v112;
    result = j_j___libc_free_0(v112, v114 - (_BYTE *)v112);
  }
  if ( v121 != v123 )
    return _libc_free(v121, v25);
  return result;
}
