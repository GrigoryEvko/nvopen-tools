// Function: sub_29F9D60
// Address: 0x29f9d60
//
void __fastcall sub_29F9D60(_QWORD *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v3; // r13
  __int64 v4; // rax
  __int64 *v5; // rbx
  _BYTE *v6; // r12
  const char *v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // r9
  const char *v10; // r8
  size_t v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  char **v14; // r13
  __int64 v15; // rcx
  unsigned __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // rdx
  _QWORD *v19; // rdi
  _BYTE *v20; // r13
  unsigned __int8 v21; // al
  __int64 v22; // r9
  size_t v23; // r13
  void *v24; // r8
  __int64 v25; // rdx
  char **v26; // r8
  __int64 v27; // rcx
  unsigned __int64 v28; // rsi
  int v29; // eax
  __int64 v30; // rdx
  _QWORD *v31; // rdi
  __int64 v32; // rcx
  char *v33; // r8
  __int64 v34; // r9
  __int64 *v35; // r14
  __int64 *v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // r9
  unsigned __int64 v39; // r15
  unsigned __int8 *v40; // r8
  unsigned __int8 *v41; // rbx
  __int64 v42; // r14
  _BYTE *v43; // r9
  unsigned __int8 **v44; // r12
  unsigned __int8 *v45; // r8
  unsigned __int8 **v46; // r13
  int v47; // eax
  int v48; // ebx
  int *v49; // rbx
  int *v50; // rsi
  int *v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rdi
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rsi
  unsigned __int64 v57; // rcx
  int v58; // eax
  _BYTE *v59; // rsi
  int v60; // ecx
  unsigned __int64 v61; // r10
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // r10
  __int64 v65; // rax
  __int64 v66; // rdx
  unsigned __int64 *v67; // rax
  __int64 v68; // r8
  size_t v69; // r15
  size_t v70; // rdi
  void *v71; // r9
  __int64 v72; // rdi
  size_t v73; // rax
  size_t v74; // rax
  unsigned __int64 v75; // r14
  __int64 v76; // r13
  size_t v77; // rbx
  const void *v78; // r15
  _BYTE *v79; // rdi
  __int64 v80; // rdx
  __int64 v81; // rbx
  unsigned __int64 *v82; // r12
  char *v83; // r13
  _BYTE *v84; // rdi
  const char *v85; // rax
  size_t v86; // rdi
  size_t v87; // rdx
  size_t v88; // r14
  const char *v89; // r15
  unsigned __int64 v90; // rdx
  char *v91; // r13
  _BYTE *v93; // [rsp+8h] [rbp-598h]
  unsigned __int8 *v94; // [rsp+18h] [rbp-588h]
  const char *v95; // [rsp+28h] [rbp-578h]
  void *v96; // [rsp+28h] [rbp-578h]
  unsigned __int8 *v97; // [rsp+30h] [rbp-570h]
  __int64 *v98; // [rsp+38h] [rbp-568h]
  void *v99; // [rsp+38h] [rbp-568h]
  void *base; // [rsp+40h] [rbp-560h] BYREF
  __int64 v101; // [rsp+48h] [rbp-558h]
  _BYTE v102[16]; // [rsp+50h] [rbp-550h] BYREF
  _BYTE *v103; // [rsp+60h] [rbp-540h] BYREF
  unsigned __int64 v104; // [rsp+68h] [rbp-538h]
  _QWORD v105[2]; // [rsp+70h] [rbp-530h] BYREF
  __int64 v106[2]; // [rsp+80h] [rbp-520h] BYREF
  _BYTE v107[16]; // [rsp+90h] [rbp-510h] BYREF
  void *src; // [rsp+A0h] [rbp-500h] BYREF
  size_t n; // [rsp+A8h] [rbp-4F8h]
  __m128i v110; // [rsp+B0h] [rbp-4F0h] BYREF
  __int64 v111; // [rsp+C0h] [rbp-4E0h]
  __int64 v112; // [rsp+C8h] [rbp-4D8h]
  __int64 *v113; // [rsp+D0h] [rbp-4D0h]
  _BYTE *v114; // [rsp+E0h] [rbp-4C0h] BYREF
  size_t v115; // [rsp+E8h] [rbp-4B8h]
  unsigned __int64 v116; // [rsp+F0h] [rbp-4B0h]
  _BYTE v117[520]; // [rsp+F8h] [rbp-4A8h] BYREF
  __int64 v118; // [rsp+300h] [rbp-2A0h] BYREF
  __int64 v119; // [rsp+308h] [rbp-298h]
  _BYTE v120[656]; // [rsp+310h] [rbp-290h] BYREF

  v3 = a2;
  v118 = (__int64)v120;
  v119 = 0x400000000LL;
  v4 = 4LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( (a2[7] & 0x40) != 0 )
  {
    v5 = (__int64 *)*((_QWORD *)a2 - 1);
    v98 = &v5[v4];
  }
  else
  {
    v98 = (__int64 *)a2;
    v5 = (__int64 *)&a2[-(v4 * 8)];
  }
  v6 = v117;
  if ( v5 != v98 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v20 = (_BYTE *)*v5;
        v21 = *(_BYTE *)*v5;
        if ( v21 > 0x1Cu )
        {
          sub_29F9A30((__int64)a1, *v5);
          v7 = sub_BD5D20((__int64)v20);
          v114 = v117;
          v10 = v7;
          v11 = v8;
          v12 = v8;
          v115 = 0;
          v116 = 128;
          if ( v8 > 0x80 )
          {
            v95 = v10;
            sub_C8D290((__int64)&v114, v117, v8, 1u, (__int64)v10, v9);
            v10 = v95;
            v79 = &v114[v115];
          }
          else
          {
            if ( !v8 )
            {
LABEL_7:
              v13 = (unsigned int)v119;
              v14 = &v114;
              v115 = v12;
              v15 = v118;
              v16 = (unsigned int)v119 + 1LL;
              v17 = v119;
              if ( v16 > HIDWORD(v119) )
              {
                if ( v118 > (unsigned __int64)&v114
                  || (unsigned __int64)&v114 >= v118 + 152 * (unsigned __int64)(unsigned int)v119 )
                {
                  sub_29F7BD0((__int64)&v118, v16, (unsigned int)v119, v118, (__int64)v10, v9);
                  v13 = (unsigned int)v119;
                  v15 = v118;
                  v17 = v119;
                }
                else
                {
                  v83 = (char *)&v114 - v118;
                  sub_29F7BD0((__int64)&v118, v16, (unsigned int)v119, v118, (__int64)v10, v9);
                  v15 = v118;
                  v13 = (unsigned int)v119;
                  v14 = (char **)&v83[v118];
                  v17 = v119;
                }
              }
              v18 = 19 * v13;
              v19 = (_QWORD *)(v15 + 8 * v18);
              if ( v19 )
              {
                v19[1] = 0;
                *v19 = v19 + 3;
                v19[2] = 128;
                if ( v14[1] )
                  sub_29F3DD0((__int64)v19, v14, v18, v15, (__int64)v10, v9);
                v17 = v119;
              }
              LODWORD(v119) = v17 + 1;
              if ( v114 != v117 )
                _libc_free((unsigned __int64)v114);
              goto LABEL_14;
            }
            v79 = v117;
          }
          memcpy(v79, v10, v11);
          v12 = v11 + v115;
          goto LABEL_7;
        }
        if ( v21 )
          break;
LABEL_14:
        v5 += 4;
        if ( v98 == v5 )
          goto LABEL_28;
      }
      v106[1] = 0;
      v106[0] = (__int64)v107;
      v112 = 0x100000000LL;
      v107[0] = 0;
      n = 0;
      src = &unk_49DD210;
      v110 = 0u;
      v113 = v106;
      v111 = 0;
      sub_CB5980((__int64)&src, 0, 0, 0);
      sub_A5BF40((unsigned __int8 *)*v5, (__int64)&src, 0, 0);
      v23 = v113[1];
      v24 = (void *)*v113;
      v114 = v117;
      v115 = 0;
      v116 = 128;
      if ( v23 > 0x80 )
      {
        v96 = v24;
        sub_C8D290((__int64)&v114, v117, v23, 1u, (__int64)v24, v22);
        v24 = v96;
        v84 = &v114[v115];
      }
      else
      {
        if ( !v23 )
          goto LABEL_19;
        v84 = v117;
      }
      memcpy(v84, v24, v23);
      v23 += v115;
LABEL_19:
      v25 = (unsigned int)v119;
      v26 = &v114;
      v115 = v23;
      v27 = v118;
      v28 = (unsigned int)v119 + 1LL;
      v29 = v119;
      if ( v28 > HIDWORD(v119) )
      {
        if ( v118 > (unsigned __int64)&v114
          || (unsigned __int64)&v114 >= v118 + 152 * (unsigned __int64)(unsigned int)v119 )
        {
          sub_29F7BD0((__int64)&v118, v28, (unsigned int)v119, v118, (__int64)&v114, v22);
          v25 = (unsigned int)v119;
          v27 = v118;
          v26 = &v114;
          v29 = v119;
        }
        else
        {
          v91 = (char *)&v114 - v118;
          sub_29F7BD0((__int64)&v118, v28, (unsigned int)v119, v118, (__int64)&v114, v22);
          v27 = v118;
          v25 = (unsigned int)v119;
          v26 = (char **)&v91[v118];
          v29 = v119;
        }
      }
      v30 = 19 * v25;
      v31 = (_QWORD *)(v27 + 8 * v30);
      if ( v31 )
      {
        v31[1] = 0;
        *v31 = v31 + 3;
        v31[2] = 128;
        if ( v26[1] )
          sub_29F3DD0((__int64)v31, v26, v30, v27, (__int64)v26, v22);
        v29 = v119;
      }
      LODWORD(v119) = v29 + 1;
      if ( v114 != v117 )
        _libc_free((unsigned __int64)v114);
      src = &unk_49DD210;
      sub_CB5840((__int64)&src);
      if ( (_BYTE *)v106[0] == v107 )
        goto LABEL_14;
      v5 += 4;
      j_j___libc_free_0(v106[0]);
      if ( v98 == v5 )
      {
LABEL_28:
        v3 = a2;
        break;
      }
    }
  }
  if ( sub_B46D50(v3) && (unsigned int)v119 > 1 )
  {
    v35 = (__int64 *)v118;
    v36 = (__int64 *)(v118 + 304);
    sub_29F64D0(v118, v118 + 304, 2, v32, v33, v34);
    sub_29F4530(v35, v36);
  }
  v37 = (unsigned int)*v3 - 29;
  v38 = 32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF);
  v39 = 0x9DDFEA08EB382D69LL
      * ((0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v37 ^ *a1)) ^ v37 ^ ((0x9DDFEA08EB382D69LL * (v37 ^ *a1)) >> 47)))
       ^ ((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v37 ^ *a1)) ^ v37 ^ ((0x9DDFEA08EB382D69LL * (v37 ^ *a1)) >> 47))) >> 47));
  base = v102;
  v101 = 0x400000000LL;
  if ( (v3[7] & 0x40) != 0 )
  {
    v40 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
    v41 = &v40[v38];
  }
  else
  {
    v41 = v3;
    v40 = &v3[-v38];
  }
  v42 = 0;
  if ( v40 != v41 )
  {
    v43 = v117;
    v44 = (unsigned __int8 **)v40;
    v45 = v3;
    v46 = (unsigned __int8 **)v41;
    do
    {
      v47 = **v44;
      if ( (unsigned __int8)v47 > 0x1Cu )
      {
        v48 = v47 - 29;
        if ( v42 + 1 > (unsigned __int64)HIDWORD(v101) )
        {
          v93 = v43;
          v94 = v45;
          sub_C8D5F0((__int64)&base, v102, v42 + 1, 4u, (__int64)v45, (__int64)v43);
          v42 = (unsigned int)v101;
          v43 = v93;
          v45 = v94;
        }
        *((_DWORD *)base + v42) = v48;
        v42 = (unsigned int)(v101 + 1);
        LODWORD(v101) = v101 + 1;
      }
      v44 += 4;
    }
    while ( v46 != v44 );
    v49 = (int *)base;
    v3 = v45;
    v6 = v43;
    if ( sub_B46D50(v45) && (unsigned int)v42 > 1 )
    {
      qsort(v49, 2u, 4u, (__compar_fn_t)sub_29F3DB0);
      v42 = (unsigned int)v101;
      v49 = (int *)base;
    }
    v50 = &v49[v42];
    if ( v50 != v49 )
    {
      v51 = v49;
      v52 = v39;
      do
      {
        v53 = *v51++;
        v54 = 0x9DDFEA08EB382D69LL
            * ((0x9DDFEA08EB382D69LL * (v53 ^ v52)) ^ v53 ^ ((0x9DDFEA08EB382D69LL * (v53 ^ v52)) >> 47));
        v52 = 0x9DDFEA08EB382D69LL * ((v54 >> 47) ^ v54);
      }
      while ( v50 != v51 );
      v39 = v52;
    }
  }
  v114 = v6;
  v115 = 0;
  v116 = 512;
  if ( v39 <= 9 )
  {
    v103 = v105;
    sub_2240A50((__int64 *)&v103, 1u, 0);
    v59 = v103;
LABEL_61:
    *v59 = v39 + 48;
    goto LABEL_62;
  }
  if ( v39 <= 0x63 )
  {
    v103 = v105;
    sub_2240A50((__int64 *)&v103, 2u, 0);
    v59 = v103;
  }
  else
  {
    if ( v39 <= 0x3E7 )
    {
      v56 = 3;
    }
    else if ( v39 <= 0x270F )
    {
      v56 = 4;
    }
    else
    {
      v55 = v39;
      LODWORD(v56) = 1;
      while ( 1 )
      {
        v57 = v55;
        v58 = v56;
        v56 = (unsigned int)(v56 + 4);
        v55 /= 0x2710u;
        if ( v57 <= 0x1869F )
          break;
        if ( v57 <= 0xF423F )
        {
          v103 = v105;
          v56 = (unsigned int)(v58 + 5);
          goto LABEL_58;
        }
        if ( v57 <= (unsigned __int64)&loc_98967F )
        {
          v56 = (unsigned int)(v58 + 6);
          break;
        }
        if ( v57 <= 0x5F5E0FF )
        {
          v56 = (unsigned int)(v58 + 7);
          break;
        }
      }
    }
    v103 = v105;
LABEL_58:
    sub_2240A50((__int64 *)&v103, v56, 0);
    v59 = v103;
    v60 = v104 - 1;
    do
    {
      v61 = v39;
      v62 = 5 * (v39 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v39 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      v63 = v39;
      v39 /= 0x64u;
      v64 = v61 - 4 * v62;
      v59[v60] = a00010203040506_0[2 * v64 + 1];
      v65 = (unsigned int)(v60 - 1);
      v60 -= 2;
      v59[v65] = a00010203040506_0[2 * v64];
    }
    while ( v63 > 0x270F );
    if ( v63 <= 0x3E7 )
      goto LABEL_61;
  }
  v59[1] = a00010203040506_0[2 * v39 + 1];
  *v59 = a00010203040506_0[2 * v39];
LABEL_62:
  v66 = v104;
  v106[0] = (__int64)v107;
  if ( v104 > 5 )
    v66 = 5;
  sub_29F3F30(v106, v103, (__int64)&v103[v66]);
  v67 = sub_2241130((unsigned __int64 *)v106, 0, 0, "op", 2u);
  src = &v110;
  if ( (unsigned __int64 *)*v67 == v67 + 2 )
  {
    v110 = _mm_loadu_si128((const __m128i *)v67 + 1);
  }
  else
  {
    src = (void *)*v67;
    v110.m128i_i64[0] = v67[2];
  }
  n = v67[1];
  *v67 = (unsigned __int64)(v67 + 2);
  v67[1] = 0;
  *((_BYTE *)v67 + 16) = 0;
  v69 = n;
  v70 = v115;
  v71 = src;
  if ( n + v115 > v116 )
  {
    v99 = src;
    sub_C8D290((__int64)&v114, v6, n + v115, 1u, v68, (__int64)src);
    v70 = v115;
    v71 = v99;
  }
  if ( v69 )
  {
    memcpy(&v114[v70], v71, v69);
    v70 = v115;
  }
  v115 = v70 + v69;
  if ( src != &v110 )
    j_j___libc_free_0((unsigned __int64)src);
  if ( (_BYTE *)v106[0] != v107 )
    j_j___libc_free_0(v106[0]);
  if ( v103 != (_BYTE *)v105 )
    j_j___libc_free_0((unsigned __int64)v103);
  if ( *v3 == 85 && (v72 = *((_QWORD *)v3 - 4)) != 0 && !*(_BYTE *)v72 && *(_QWORD *)(v72 + 24) == *((_QWORD *)v3 + 10) )
  {
    v85 = sub_BD5D20(v72);
    v86 = v115;
    v88 = v87;
    v89 = v85;
    v90 = v115 + v87;
    if ( v90 > v116 )
    {
      sub_C8D290((__int64)&v114, v6, v90, 1u, v68, (__int64)v71);
      v86 = v115;
    }
    if ( v88 )
    {
      memcpy(&v114[v86], v89, v88);
      v86 = v115;
    }
    v73 = v88 + v86;
    v115 = v88 + v86;
  }
  else
  {
    v73 = v115;
  }
  if ( v73 + 1 > v116 )
  {
    sub_C8D290((__int64)&v114, v6, v73 + 1, 1u, v68, (__int64)v71);
    v73 = v115;
  }
  v114[v73] = 40;
  v74 = ++v115;
  if ( (_DWORD)v119 )
  {
    v75 = 0;
    v97 = v3;
    v76 = 0;
    do
    {
      while ( 1 )
      {
        v77 = *(_QWORD *)(v76 + v118 + 8);
        v78 = *(const void **)(v76 + v118);
        if ( v77 + v74 > v116 )
        {
          sub_C8D290((__int64)&v114, v6, v77 + v74, 1u, v68, (__int64)v71);
          v74 = v115;
        }
        if ( v77 )
        {
          memcpy(&v114[v74], v78, v77);
          v74 = v115;
        }
        v74 += v77;
        v115 = v74;
        if ( (unsigned __int64)(unsigned int)v119 - 1 > v75 )
          break;
        ++v75;
        v76 += 152;
        if ( v75 >= (unsigned int)v119 )
          goto LABEL_99;
      }
      if ( v74 + 2 > v116 )
      {
        sub_C8D290((__int64)&v114, v6, v74 + 2, 1u, v68, (__int64)v71);
        v74 = v115;
      }
      ++v75;
      v76 += 152;
      *(_WORD *)&v114[v74] = 8236;
      v74 = v115 + 2;
      v115 += 2LL;
    }
    while ( v75 < (unsigned int)v119 );
LABEL_99:
    v3 = v97;
  }
  if ( v74 + 1 > v116 )
  {
    sub_C8D290((__int64)&v114, v6, v74 + 1, 1u, v68, (__int64)v71);
    v74 = v115;
  }
  v114[v74] = 41;
  ++v115;
  sub_BD5D20((__int64)v3);
  if ( (!v80 || (_BYTE)qword_5009888) && *(_BYTE *)(*((_QWORD *)v3 + 1) + 8LL) != 7 )
  {
    LOWORD(v111) = 261;
    src = v114;
    n = v115;
    sub_BD6B50(v3, (const char **)&src);
  }
  if ( v114 != v6 )
    _libc_free((unsigned __int64)v114);
  if ( base != v102 )
    _libc_free((unsigned __int64)base);
  v81 = v118;
  v82 = (unsigned __int64 *)(v118 + 152LL * (unsigned int)v119);
  if ( (unsigned __int64 *)v118 != v82 )
  {
    do
    {
      v82 -= 19;
      if ( (unsigned __int64 *)*v82 != v82 + 3 )
        _libc_free(*v82);
    }
    while ( (unsigned __int64 *)v81 != v82 );
    v82 = (unsigned __int64 *)v118;
  }
  if ( v82 != (unsigned __int64 *)v120 )
    _libc_free((unsigned __int64)v82);
}
