// Function: sub_1533CF0
// Address: 0x1533cf0
//
void __fastcall sub_1533CF0(__int64 *a1)
{
  __int64 *v1; // r14
  _QWORD *v2; // rax
  char *v3; // r12
  _DWORD *v4; // rbx
  _QWORD *v5; // rdi
  __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdi
  volatile signed __int32 *v10; // r8
  _QWORD *v11; // rdi
  __int64 v12; // rax
  _DWORD *v13; // rcx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r13
  unsigned int v17; // r15d
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rdi
  _QWORD *v29; // rbx
  __int64 v30; // r15
  signed __int64 v31; // rdx
  signed __int64 v32; // r12
  __int64 v33; // rdx
  _BYTE *v34; // rcx
  __int64 i; // rax
  _DWORD *v36; // rdi
  unsigned int v37; // r15d
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // r12d
  __int64 v41; // rax
  int v42; // edx
  __int64 v43; // r8
  int v44; // edi
  _DWORD *v45; // rsi
  unsigned int v46; // ecx
  _DWORD *v47; // rdx
  __int64 v48; // r8
  __int64 v49; // rax
  _QWORD *v50; // rbx
  __int64 v51; // r12
  _DWORD *v52; // r12
  __int64 v53; // r15
  unsigned __int64 v54; // rsi
  _QWORD *v55; // rbx
  _DWORD *v56; // r12
  __int64 v57; // r14
  __int64 v58; // r13
  __int32 v59; // r15d
  __int64 v60; // r13
  unsigned __int64 v61; // rsi
  int v62; // edx
  __int64 v63; // rbx
  char *v64; // rax
  void *v65; // r8
  char *v66; // rcx
  __int64 v67; // rdx
  __int64 v68; // rax
  _DWORD *v69; // r15
  __int64 v70; // rdx
  _QWORD **v71; // r8
  __int64 v72; // rax
  __int64 v73; // rcx
  unsigned __int64 v74; // rsi
  _QWORD *v75; // rdx
  char v76; // r10
  unsigned __int64 v77; // rsi
  _QWORD *v78; // rdx
  char v79; // r11
  __int64 *v80; // rdx
  __int64 *v81; // rdi
  __int64 v82; // rsi
  _DWORD *v83; // r13
  __int64 v84; // rbx
  __int64 v85; // rcx
  int v86; // r9d
  char *v87; // rax
  signed __int64 v88; // rsi
  __int64 v89; // r12
  __int64 v90; // rbx
  unsigned __int64 v91; // rsi
  unsigned int v92; // [rsp+4h] [rbp-2FCh]
  _QWORD *v93; // [rsp+8h] [rbp-2F8h]
  unsigned int v94; // [rsp+8h] [rbp-2F8h]
  __int64 *v95; // [rsp+8h] [rbp-2F8h]
  void *v96; // [rsp+8h] [rbp-2F8h]
  char *v97; // [rsp+8h] [rbp-2F8h]
  unsigned int v98; // [rsp+20h] [rbp-2E0h]
  __int64 v99; // [rsp+20h] [rbp-2E0h]
  _QWORD *v100; // [rsp+28h] [rbp-2D8h]
  _QWORD *v101; // [rsp+28h] [rbp-2D8h]
  __int64 v102; // [rsp+38h] [rbp-2C8h] BYREF
  unsigned __int128 v103; // [rsp+40h] [rbp-2C0h] BYREF
  void *v104; // [rsp+50h] [rbp-2B0h] BYREF
  char *v105; // [rsp+58h] [rbp-2A8h]
  char *v106; // [rsp+60h] [rbp-2A0h]
  void *src; // [rsp+70h] [rbp-290h] BYREF
  __int64 *v108; // [rsp+78h] [rbp-288h]
  char *v109; // [rsp+80h] [rbp-280h]
  __m128i v110; // [rsp+90h] [rbp-270h] BYREF
  _BYTE v111[32]; // [rsp+A0h] [rbp-260h] BYREF
  _BYTE *v112; // [rsp+C0h] [rbp-240h] BYREF
  __int64 v113; // [rsp+C8h] [rbp-238h]
  _BYTE v114[560]; // [rsp+D0h] [rbp-230h] BYREF

  v1 = a1;
  if ( (a1[30] - a1[29]) >> 3 <= (unsigned __int64)*((unsigned int *)a1 + 141)
    && (*(_QWORD *)(a1[2] + 72) & 0xFFFFFFFFFFFFFFF8LL) == a1[2] + 72 )
  {
    return;
  }
  sub_1526BE0((_QWORD *)*a1, 0xFu, 4u);
  v104 = 0;
  v112 = v114;
  v113 = 0x4000000000LL;
  v105 = 0;
  v106 = 0;
  v2 = (_QWORD *)sub_22077B0(124);
  v3 = (char *)v2 + 124;
  v4 = v2;
  *v2 = 0;
  *(_QWORD *)((char *)v2 + 116) = 0;
  memset(
    (void *)((unsigned __int64)(v2 + 1) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v2 - (((_DWORD)v2 + 8) & 0xFFFFFFF8) + 124) >> 3));
  if ( v105 - (_BYTE *)v104 > 0 )
  {
    memmove(v2, v104, v105 - (_BYTE *)v104);
  }
  else if ( !v104 )
  {
    goto LABEL_5;
  }
  j_j___libc_free_0(v104, v106 - (_BYTE *)v104);
LABEL_5:
  v104 = v4;
  v105 = v3;
  v106 = v3;
  v4[1] = sub_1527610((_QWORD **)a1);
  *((_DWORD *)v104 + 4) = sub_1527880((_QWORD **)a1);
  sub_1531130(&v103);
  v110.m128i_i8[8] |= 1u;
  v110.m128i_i64[0] = 38;
  sub_1525B40(v103, &v110);
  v110.m128i_i64[0] = 32;
  v110.m128i_i8[8] = 2;
  sub_1525B40(v103, &v110);
  v110.m128i_i64[0] = 32;
  v110.m128i_i8[8] = 2;
  sub_1525B40(v103, &v110);
  v5 = (_QWORD *)*a1;
  v6 = *((_QWORD *)&v103 + 1);
  v110.m128i_i64[0] = v103;
  v103 = 0u;
  v110.m128i_i64[1] = v6;
  v7 = sub_15271D0(v5, v110.m128i_i64);
  if ( v110.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v110.m128i_i64[1]);
  sub_1531130(&v110);
  v8 = v110.m128i_i64[1];
  v9 = v110.m128i_i64[0];
  v110 = 0u;
  v10 = (volatile signed __int32 *)*((_QWORD *)&v103 + 1);
  v103 = __PAIR128__(v8, v9);
  if ( v10 )
  {
    sub_A191D0(v10);
    if ( v110.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v110.m128i_i64[1]);
    v9 = v103;
  }
  v110.m128i_i8[8] |= 1u;
  v110.m128i_i64[0] = 39;
  sub_1525B40(v9, &v110);
  v110.m128i_i64[0] = 0;
  v110.m128i_i8[8] = 6;
  sub_1525B40(v103, &v110);
  v110.m128i_i64[0] = 6;
  v110.m128i_i8[8] = 4;
  sub_1525B40(v103, &v110);
  v11 = (_QWORD *)*v1;
  v12 = *((_QWORD *)&v103 + 1);
  v110.m128i_i64[0] = v103;
  v103 = 0u;
  v110.m128i_i64[1] = v12;
  v98 = sub_15271D0(v11, v110.m128i_i64);
  if ( v110.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v110.m128i_i64[1]);
  sub_152AB40(v1, (_QWORD *)(v1[29] + 8LL * *((unsigned int *)v1 + 141)), *((unsigned int *)v1 + 142), (__int64)&v112);
  v13 = (_DWORD *)v1[29];
  v14 = *((unsigned int *)v1 + 141) + (unsigned __int64)*((unsigned int *)v1 + 142);
  v15 = ((v1[30] - (__int64)v13) >> 3) - v14;
  if ( (unsigned int)dword_4F9DF40 < v15 )
  {
    v110 = 0u;
    v69 = (_DWORD *)*v1;
    if ( v7 )
    {
      src = (void *)0x100000026LL;
      sub_152A250((__int64)v69, v7, (__int64)&v110, 2, 0, 0, (__int64)&src);
    }
    else
    {
      sub_1524D80(v69, 3u, v69[4]);
      sub_1524E40(v69, 0x26u, 6);
      sub_1524E40(v69, 2u, 6);
      sub_1525280(v69, v110.m128i_u64[0], 6);
      sub_1525280(v69, v110.m128i_u64[1], 6);
    }
    v13 = (_DWORD *)v1[29];
    v14 = *((unsigned int *)v1 + 142) + (unsigned __int64)*((unsigned int *)v1 + 141);
    v16 = *(unsigned int *)(*v1 + 8);
    v70 = (v1[30] - (__int64)v13) >> 3;
    v17 = *(_DWORD *)(*(_QWORD *)*v1 + 8LL);
    src = 0;
    v108 = 0;
    v15 = v70 - v14;
    v109 = 0;
    if ( v15 > 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::reserve");
  }
  else
  {
    v16 = *(unsigned int *)(*v1 + 8);
    v17 = *(_DWORD *)(*(_QWORD *)*v1 + 8LL);
    src = 0;
    v108 = 0;
    v109 = 0;
  }
  if ( v15 )
  {
    v63 = 8 * v15;
    v64 = (char *)sub_22077B0(8 * v15);
    v65 = src;
    v66 = v64;
    if ( (char *)v108 - (_BYTE *)src > 0 )
    {
      v96 = src;
      v87 = (char *)memmove(v64, src, (char *)v108 - (_BYTE *)src);
      v65 = v96;
      v66 = v87;
      v88 = v109 - (_BYTE *)v96;
    }
    else
    {
      if ( !src )
      {
LABEL_78:
        v67 = *((unsigned int *)v1 + 141);
        v68 = *((unsigned int *)v1 + 142);
        src = v66;
        v108 = (__int64 *)v66;
        v14 = v67 + v68;
        v109 = &v66[v63];
        v13 = (_DWORD *)v1[29];
        v15 = ((v1[30] - (__int64)v13) >> 3) - v14;
        goto LABEL_16;
      }
      v88 = v109 - (_BYTE *)src;
    }
    v97 = v66;
    j_j___libc_free_0(v65, v88);
    v66 = v97;
    goto LABEL_78;
  }
LABEL_16:
  sub_15334D0((_QWORD **)v1, (__int64 *)&v13[2 * v14], v15, (__int64)&v112, (unsigned int **)&v104, (__int64)&src);
  if ( (unsigned int)dword_4F9DF40 < ((v1[30] - v1[29]) >> 3)
                                   - (*((unsigned int *)v1 + 142)
                                    + (unsigned __int64)*((unsigned int *)v1 + 141)) )
  {
    v71 = (_QWORD **)*v1;
    v72 = v16 + 8LL * v17;
    v73 = *(_QWORD *)*v1;
    v74 = *(unsigned int *)(*v1 + 8) + 8LL * *(unsigned int *)(v73 + 8) - v72;
    v75 = (_QWORD *)(*(_QWORD *)v73 + (unsigned int)((unsigned __int64)(v72 - 64) >> 3));
    if ( (((_BYTE)v16 + 8 * (_BYTE)v17) & 7) != 0 )
    {
      v76 = (v16 + 8 * v17) & 7;
      *v75 = ((((unsigned int)v74 >> (32 - v76)) & ~(-1 << v76) | HIDWORD(*v75) & (-1 << v76)) << 32)
           | (((unsigned int)v74 & ~(-1 << (32 - v76))) << v76)
           | (unsigned int)*v75 & ~(-1 << v76);
    }
    else
    {
      *(_DWORD *)v75 = v74;
    }
    v77 = HIDWORD(v74);
    v78 = (_QWORD *)(**v71 + (unsigned int)((unsigned __int64)(v72 - 32) >> 3));
    if ( (((_BYTE)v16 + 8 * (_BYTE)v17) & 7) != 0 )
    {
      v79 = (v16 + 8 * v17) & 7;
      *v78 = (unsigned int)*v78 & ~(-1 << v79)
           | (((unsigned int)v77 & ~(-1 << (32 - v79))) << v79)
           | ((~(-1 << v79) & ((unsigned int)v77 >> (32 - v79)) | HIDWORD(*v78) & (-1 << v79)) << 32);
    }
    else
    {
      *(_DWORD *)v78 = v77;
    }
    v80 = (__int64 *)src;
    v81 = v108;
    if ( v108 == src )
    {
      v83 = (_DWORD *)*v1;
      v85 = 0;
      if ( !v98 )
      {
        sub_1524D80(v83, 3u, v83[4]);
        sub_1524E40(v83, 0x27u, 6);
        sub_1524E40(v83, 0, 6);
LABEL_91:
        if ( src != v108 )
          v108 = (__int64 *)src;
        goto LABEL_17;
      }
    }
    else
    {
      do
      {
        v82 = v72;
        v72 = *v80++;
        *(v80 - 1) = v72 - v82;
      }
      while ( v81 != v80 );
      v80 = (__int64 *)src;
      v83 = (_DWORD *)*v1;
      v84 = ((char *)v108 - (_BYTE *)src) >> 3;
      v85 = v84;
      if ( !v98 )
      {
        sub_1524D80(v83, 3u, v83[4]);
        sub_1524E40(v83, 0x27u, 6);
        sub_1524E40(v83, v84, 6);
        if ( (_DWORD)v84 )
        {
          v89 = 0;
          v90 = 8LL * (unsigned int)v84;
          do
          {
            v91 = *(_QWORD *)((char *)src + v89);
            v89 += 8;
            sub_1525280(v83, v91, 6);
          }
          while ( v90 != v89 );
        }
        goto LABEL_91;
      }
    }
    v110.m128i_i64[0] = 0x100000027LL;
    sub_152A250((__int64)v83, v98, (__int64)v80, v85, 0, 0, (__int64)&v110);
    goto LABEL_91;
  }
LABEL_17:
  v18 = (_QWORD *)v1[2];
  if ( v18 + 9 != (_QWORD *)(v18[9] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v19 = (_QWORD *)sub_22077B0(544);
    v20 = (__int64)v19;
    if ( v19 )
    {
      v21 = (__int64)(v19 + 2);
      v19[1] = 0x100000001LL;
      *v19 = &unk_49ECD20;
      v19[2] = v19 + 4;
      v19[3] = 0x2000000000LL;
      v22 = 0;
    }
    else
    {
      v22 = MEMORY[0x18];
      v21 = 16;
      if ( MEMORY[0x18] >= MEMORY[0x1C] )
      {
        sub_16CD150(16, 32, 0, 16);
        v22 = MEMORY[0x18];
      }
    }
    v23 = (_QWORD *)(*(_QWORD *)(v20 + 16) + 16 * v22);
    *v23 = 4;
    v23[1] = 1;
    v24 = (unsigned int)(*(_DWORD *)(v20 + 24) + 1);
    *(_DWORD *)(v20 + 24) = v24;
    if ( *(_DWORD *)(v20 + 28) <= (unsigned int)v24 )
    {
      sub_16CD150(v21, v20 + 32, 0, 16);
      v24 = *(unsigned int *)(v20 + 24);
    }
    v25 = (_QWORD *)(*(_QWORD *)(v20 + 16) + 16 * v24);
    *v25 = 0;
    v25[1] = 6;
    v26 = (unsigned int)(*(_DWORD *)(v20 + 24) + 1);
    *(_DWORD *)(v20 + 24) = v26;
    if ( *(_DWORD *)(v20 + 28) <= (unsigned int)v26 )
    {
      sub_16CD150(v21, v20 + 32, 0, 16);
      v26 = *(unsigned int *)(v20 + 24);
    }
    v27 = (_QWORD *)(*(_QWORD *)(v20 + 16) + 16 * v26);
    *v27 = 8;
    v27[1] = 2;
    v28 = (_QWORD *)*v1;
    ++*(_DWORD *)(v20 + 24);
    v110.m128i_i64[0] = v21;
    v110.m128i_i64[1] = v20;
    v92 = sub_15271D0(v28, v110.m128i_i64);
    if ( v110.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v110.m128i_i64[1]);
    v18 = (_QWORD *)v1[2];
    v93 = v18 + 9;
    v29 = (_QWORD *)v18[10];
    if ( v18 + 9 != v29 )
    {
      do
      {
        v30 = sub_161F640(v29);
        v32 = v31;
        v33 = (unsigned int)v113;
        if ( v32 > HIDWORD(v113) - (unsigned __int64)(unsigned int)v113 )
        {
          sub_16CD150(&v112, v114, v32 + (unsigned int)v113, 8);
          v33 = (unsigned int)v113;
        }
        v34 = &v112[8 * v33];
        if ( v32 > 0 )
        {
          for ( i = 0; i != v32; ++i )
            *(_QWORD *)&v34[8 * i] = *(unsigned __int8 *)(v30 + i);
          LODWORD(v33) = v113;
        }
        v36 = (_DWORD *)*v1;
        v37 = 0;
        LODWORD(v113) = v33 + v32;
        sub_152B6B0(v36, 4u, (__int64)&v112, v92);
        LODWORD(v113) = 0;
        v40 = sub_161F520(v29, 4, v38, v39);
        if ( v40 )
        {
          do
          {
            v41 = sub_161F530(v29, v37);
            v42 = *((_DWORD *)v1 + 76);
            v43 = 0xFFFFFFFFLL;
            if ( v42 )
            {
              v44 = v42 - 1;
              v45 = (_DWORD *)v1[36];
              v46 = (v42 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v47 = &v45[4 * v46];
              v48 = *(_QWORD *)v47;
              if ( v41 == *(_QWORD *)v47 )
              {
LABEL_36:
                v43 = (unsigned int)(v47[3] - 1);
              }
              else
              {
                v62 = 1;
                while ( v48 != -4 )
                {
                  v86 = v62 + 1;
                  v46 = v44 & (v62 + v46);
                  v47 = &v45[4 * v46];
                  v48 = *(_QWORD *)v47;
                  if ( v41 == *(_QWORD *)v47 )
                    goto LABEL_36;
                  v62 = v86;
                }
                v43 = 0xFFFFFFFFLL;
              }
            }
            v49 = (unsigned int)v113;
            if ( (unsigned int)v113 >= HIDWORD(v113) )
            {
              v99 = v43;
              sub_16CD150(&v112, v114, 0, 8);
              v49 = (unsigned int)v113;
              v43 = v99;
            }
            ++v37;
            *(_QWORD *)&v112[8 * v49] = v43;
            LODWORD(v113) = v113 + 1;
          }
          while ( v37 != v40 );
        }
        sub_152B6B0((_DWORD *)*v1, 0xAu, (__int64)&v112, 0);
        LODWORD(v113) = 0;
        v29 = (_QWORD *)v29[1];
      }
      while ( v93 != v29 );
      v18 = (_QWORD *)v1[2];
    }
  }
  v50 = (_QWORD *)v18[4];
  v100 = v18 + 3;
  if ( v18 + 3 != v50 )
  {
    do
    {
      while ( 1 )
      {
        v51 = (__int64)(v50 - 7);
        if ( !v50 )
          v51 = 0;
        if ( (unsigned __int8)sub_15E4F60(v51) && (*(_BYTE *)(v51 + 34) & 0x10) != 0 )
        {
          v110.m128i_i64[0] = (__int64)v111;
          v110.m128i_i64[1] = 0x400000000LL;
          v102 = (unsigned int)sub_153E840(v1 + 3);
          sub_1525CA0((__int64)&v110, &v102);
          sub_1524A90((__int64)v1, (__int64)&v110, v51);
          v52 = (_DWORD *)*v1;
          v53 = 0;
          v94 = v110.m128i_u32[2];
          sub_1524D80((_DWORD *)*v1, 3u, *(_DWORD *)(*v1 + 16));
          sub_1524E40(v52, 0x24u, 6);
          sub_1524E40(v52, v94, 6);
          if ( v94 )
          {
            do
            {
              v54 = *(_QWORD *)(v110.m128i_i64[0] + v53);
              v53 += 8;
              sub_1525280(v52, v54, 6);
            }
            while ( 8LL * v94 != v53 );
          }
          if ( (_BYTE *)v110.m128i_i64[0] != v111 )
            break;
        }
        v50 = (_QWORD *)v50[1];
        if ( v100 == v50 )
          goto LABEL_53;
      }
      _libc_free(v110.m128i_u64[0]);
      v50 = (_QWORD *)v50[1];
    }
    while ( v100 != v50 );
LABEL_53:
    v18 = (_QWORD *)v1[2];
  }
  v55 = (_QWORD *)v18[2];
  v101 = v18 + 1;
  if ( v18 + 1 != v55 )
  {
    v95 = v1;
    do
    {
      while ( 1 )
      {
        if ( !v55 )
          BUG();
        if ( (*((_BYTE *)v55 - 22) & 0x10) != 0 )
        {
          v110.m128i_i64[0] = (__int64)v111;
          v110.m128i_i64[1] = 0x400000000LL;
          v102 = (unsigned int)sub_153E840(v95 + 3);
          sub_1525CA0((__int64)&v110, &v102);
          sub_1524A90((__int64)v95, (__int64)&v110, (__int64)(v55 - 7));
          v56 = (_DWORD *)*v95;
          v57 = 0;
          v58 = v110.m128i_u32[2];
          v59 = v110.m128i_i32[2];
          sub_1524D80(v56, 3u, v56[4]);
          sub_1524E40(v56, 0x24u, 6);
          sub_1524E40(v56, v58, 6);
          v60 = 8 * v58;
          if ( v59 )
          {
            do
            {
              v61 = *(_QWORD *)(v110.m128i_i64[0] + v57);
              v57 += 8;
              sub_1525280(v56, v61, 6);
            }
            while ( v57 != v60 );
          }
          if ( (_BYTE *)v110.m128i_i64[0] != v111 )
            break;
        }
        v55 = (_QWORD *)v55[1];
        if ( v101 == v55 )
          goto LABEL_63;
      }
      _libc_free(v110.m128i_u64[0]);
      v55 = (_QWORD *)v55[1];
    }
    while ( v101 != v55 );
LABEL_63:
    v1 = v95;
  }
  sub_15263C0((__int64 **)*v1);
  if ( src )
    j_j___libc_free_0(src, v109 - (_BYTE *)src);
  if ( *((_QWORD *)&v103 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v103 + 1));
  if ( v104 )
    j_j___libc_free_0(v104, v106 - (_BYTE *)v104);
  if ( v112 != v114 )
    _libc_free((unsigned __int64)v112);
}
