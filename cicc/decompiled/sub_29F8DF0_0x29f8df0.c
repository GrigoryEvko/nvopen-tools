// Function: sub_29F8DF0
// Address: 0x29f8df0
//
void __fastcall sub_29F8DF0(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rax
  unsigned __int8 **v5; // r13
  unsigned __int8 **v6; // r8
  unsigned __int8 **v7; // r12
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // rdx
  void *v11; // r10
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  char **v15; // r10
  int v16; // eax
  __int64 v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  char *v21; // r8
  __int64 v22; // r9
  int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // r14
  unsigned __int64 *v27; // rcx
  unsigned __int64 *v28; // rsi
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rcx
  int v35; // eax
  _BYTE *v36; // rsi
  int v37; // ecx
  unsigned __int64 v38; // r10
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // r10
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 *v44; // rax
  __int64 v45; // r9
  size_t v46; // r8
  __int64 v47; // rdi
  void *v48; // r10
  __int64 v49; // r8
  __int64 v50; // rdi
  const char *v51; // rax
  __int64 v52; // rdi
  size_t v53; // rdx
  size_t v54; // r14
  const char *v55; // r15
  unsigned __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rbx
  unsigned __int64 *v59; // r12
  unsigned __int64 v60; // r15
  __int64 v61; // r12
  unsigned __int64 v62; // rdx
  size_t v63; // r13
  const void *v64; // r14
  int *v65; // rdi
  __int64 v66; // rax
  void *v67; // [rsp+8h] [rbp-4B8h]
  unsigned __int8 *v68; // [rsp+10h] [rbp-4B0h]
  size_t v69; // [rsp+20h] [rbp-4A0h]
  size_t v70; // [rsp+20h] [rbp-4A0h]
  char *v71; // [rsp+20h] [rbp-4A0h]
  size_t v73; // [rsp+28h] [rbp-498h]
  size_t v74; // [rsp+30h] [rbp-490h]
  unsigned __int8 *v75; // [rsp+30h] [rbp-490h]
  void *v76; // [rsp+30h] [rbp-490h]
  _BYTE *v77; // [rsp+40h] [rbp-480h] BYREF
  unsigned __int64 v78; // [rsp+48h] [rbp-478h]
  _QWORD v79[2]; // [rsp+50h] [rbp-470h] BYREF
  __int64 v80[2]; // [rsp+60h] [rbp-460h] BYREF
  void *v81[2]; // [rsp+70h] [rbp-450h] BYREF
  void *src; // [rsp+80h] [rbp-440h] BYREF
  size_t n; // [rsp+88h] [rbp-438h]
  __m128i v84; // [rsp+90h] [rbp-430h] BYREF
  __int16 v85; // [rsp+A0h] [rbp-420h]
  __int64 *v86; // [rsp+B0h] [rbp-410h] BYREF
  __int64 v87; // [rsp+B8h] [rbp-408h]
  __int64 v88; // [rsp+C0h] [rbp-400h] BYREF
  unsigned int v89; // [rsp+C8h] [rbp-3F8h]
  unsigned __int64 *v90; // [rsp+D0h] [rbp-3F0h]
  unsigned int v91; // [rsp+D8h] [rbp-3E8h]
  _QWORD *v92; // [rsp+E0h] [rbp-3E0h] BYREF
  __int64 v93; // [rsp+E8h] [rbp-3D8h]
  unsigned __int64 v94; // [rsp+F0h] [rbp-3D0h]
  _QWORD v95[3]; // [rsp+F8h] [rbp-3C8h] BYREF
  void **v96; // [rsp+110h] [rbp-3B0h]
  unsigned __int64 v97; // [rsp+200h] [rbp-2C0h] BYREF
  char *v98; // [rsp+208h] [rbp-2B8h]
  __int64 v99; // [rsp+210h] [rbp-2B0h]
  int v100; // [rsp+218h] [rbp-2A8h] BYREF
  char v101; // [rsp+21Ch] [rbp-2A4h]
  char v102; // [rsp+220h] [rbp-2A0h] BYREF
  __int64 v103; // [rsp+320h] [rbp-1A0h] BYREF
  __int64 v104; // [rsp+328h] [rbp-198h]
  _BYTE v105[400]; // [rsp+330h] [rbp-190h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 7 )
    return;
  v2 = (unsigned __int8 *)a2;
  sub_BD5D20(a2);
  if ( v3 )
  {
    if ( !(_BYTE)qword_5009888 )
      return;
  }
  v103 = (__int64)v105;
  v104 = 0x400000000LL;
  v4 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(unsigned __int8 ***)(a2 - 8);
    v5 = &v6[v4];
  }
  else
  {
    v5 = (unsigned __int8 **)a2;
    v6 = (unsigned __int8 **)(a2 - v4 * 8);
  }
  if ( v6 != v5 )
  {
    v68 = (unsigned __int8 *)a2;
    v7 = v6;
    while ( 1 )
    {
      while ( !**v7 )
      {
LABEL_8:
        v7 += 4;
        if ( v5 == v7 )
          goto LABEL_21;
      }
      v95[2] = 0x100000000LL;
      v86 = &v88;
      v87 = 0;
      v92 = &unk_49DD210;
      LOBYTE(v88) = 0;
      v96 = (void **)&v86;
      v93 = 0;
      v94 = 0;
      v95[0] = 0;
      v95[1] = 0;
      sub_CB5980((__int64)&v92, 0, 0, 0);
      sub_A5BF40(*v7, (__int64)&v92, 0, 0);
      v10 = (unsigned __int64)v96[1];
      v11 = *v96;
      v97 = (unsigned __int64)&v100;
      v98 = 0;
      v99 = 64;
      if ( v10 > 0x40 )
      {
        v67 = v11;
        v69 = v10;
        sub_C8D290((__int64)&v97, &v100, v10, 1u, v8, v9);
        v10 = v69;
        v11 = v67;
        v65 = (int *)&v98[v97];
      }
      else
      {
        if ( !v10 )
          goto LABEL_12;
        v65 = &v100;
      }
      v70 = v10;
      memcpy(v65, v11, v10);
      v10 = (unsigned __int64)&v98[v70];
LABEL_12:
      v98 = (char *)v10;
      v12 = (unsigned int)v104;
      v13 = v103;
      v14 = (unsigned int)v104 + 1LL;
      v15 = (char **)&v97;
      v16 = v104;
      if ( v14 > HIDWORD(v104) )
      {
        if ( v103 > (unsigned __int64)&v97 || (unsigned __int64)&v97 >= v103 + 88 * (unsigned __int64)(unsigned int)v104 )
        {
          sub_29F7AB0((__int64)&v103, v14, (unsigned int)v104, v103, v8, v9);
          v12 = (unsigned int)v104;
          v13 = v103;
          v15 = (char **)&v97;
          v16 = v104;
        }
        else
        {
          v71 = (char *)&v97 - v103;
          sub_29F7AB0((__int64)&v103, v14, (unsigned int)v104, v103, v8, v9);
          v13 = v103;
          v12 = (unsigned int)v104;
          v15 = (char **)&v71[v103];
          v16 = v104;
        }
      }
      a2 = 5 * v12;
      v17 = 11 * v12;
      v18 = (_QWORD *)(v13 + 8 * v17);
      if ( v18 )
      {
        v18[1] = 0;
        *v18 = v18 + 3;
        v18[2] = 64;
        if ( v15[1] )
        {
          a2 = (__int64)v15;
          sub_29F3DD0((__int64)v18, v15, v17, v13, v8, v9);
        }
        v16 = v104;
      }
      LODWORD(v104) = v16 + 1;
      if ( (int *)v97 != &v100 )
        _libc_free(v97);
      v92 = &unk_49DD210;
      sub_CB5840((__int64)&v92);
      if ( v86 == &v88 )
        goto LABEL_8;
      v7 += 4;
      a2 = v88 + 1;
      j_j___libc_free_0((unsigned __int64)v86);
      if ( v5 == v7 )
      {
LABEL_21:
        v2 = v68;
        break;
      }
    }
  }
  if ( sub_B46D50(v2) && (unsigned int)v104 > 1 )
    sub_29F8720(&v103, a2, v19, v20, v21, v22);
  v23 = *v2;
  v97 = 0;
  v101 = 1;
  v100 = 0;
  v24 = *a1;
  v25 = (unsigned int)(v23 - 29);
  v99 = 32;
  v98 = &v102;
  v26 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v25 ^ v24)) ^ v25 ^ ((0x9DDFEA08EB382D69LL * (v25 ^ v24)) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v25 ^ v24)) ^ v25 ^ ((0x9DDFEA08EB382D69LL * (v25 ^ v24)) >> 47))));
  sub_29F8B90((__int64)&v86, (__int64)v2, (__int64)&v97, v20, (__int64)v21, v22);
  v27 = v90;
  v28 = (unsigned __int64 *)((char *)v90 + 4 * v91);
  if ( v28 != v90 )
  {
    v29 = v26;
    do
    {
      v30 = *(int *)v27;
      v27 = (unsigned __int64 *)((char *)v27 + 4);
      v31 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v30 ^ v29)) ^ v30 ^ ((0x9DDFEA08EB382D69LL * (v30 ^ v29)) >> 47));
      v29 = 0x9DDFEA08EB382D69LL * ((v31 >> 47) ^ v31);
    }
    while ( v28 != v27 );
    v26 = v29;
  }
  v93 = 0;
  v92 = v95;
  v94 = 256;
  if ( v26 > 9 )
  {
    if ( v26 <= 0x63 )
    {
      v77 = v79;
      sub_2240A50((__int64 *)&v77, 2u, 0);
      v36 = v77;
    }
    else
    {
      if ( v26 <= 0x3E7 )
      {
        v33 = 3;
      }
      else if ( v26 <= 0x270F )
      {
        v33 = 4;
      }
      else
      {
        v32 = v26;
        LODWORD(v33) = 1;
        while ( 1 )
        {
          v34 = v32;
          v35 = v33;
          v33 = (unsigned int)(v33 + 4);
          v32 /= 0x2710u;
          if ( v34 <= 0x1869F )
            break;
          if ( v34 <= 0xF423F )
          {
            v77 = v79;
            v33 = (unsigned int)(v35 + 5);
            goto LABEL_39;
          }
          if ( v34 <= (unsigned __int64)&loc_98967F )
          {
            v33 = (unsigned int)(v35 + 6);
            break;
          }
          if ( v34 <= 0x5F5E0FF )
          {
            v33 = (unsigned int)(v35 + 7);
            break;
          }
        }
      }
      v77 = v79;
LABEL_39:
      sub_2240A50((__int64 *)&v77, v33, 0);
      v36 = v77;
      v37 = v78 - 1;
      do
      {
        v38 = v26;
        v39 = 5 * (v26 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v26 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
        v40 = v26;
        v26 /= 0x64u;
        v41 = v38 - 4 * v39;
        v36[v37] = a00010203040506_0[2 * v41 + 1];
        v42 = (unsigned int)(v37 - 1);
        v37 -= 2;
        v36[v42] = a00010203040506_0[2 * v41];
      }
      while ( v40 > 0x270F );
      if ( v40 <= 0x3E7 )
        goto LABEL_42;
    }
    v36[1] = a00010203040506_0[2 * v26 + 1];
    *v36 = a00010203040506_0[2 * v26];
    goto LABEL_43;
  }
  v77 = v79;
  sub_2240A50((__int64 *)&v77, 1u, 0);
  v36 = v77;
LABEL_42:
  *v36 = v26 + 48;
LABEL_43:
  v43 = v78;
  v80[0] = (__int64)v81;
  if ( v78 > 5 )
    v43 = 5;
  sub_29F3F30(v80, v77, (__int64)&v77[v43]);
  v44 = sub_2241130((unsigned __int64 *)v80, 0, 0, "vl", 2u);
  src = &v84;
  if ( (unsigned __int64 *)*v44 == v44 + 2 )
  {
    v84 = _mm_loadu_si128((const __m128i *)v44 + 1);
  }
  else
  {
    src = (void *)*v44;
    v84.m128i_i64[0] = v44[2];
  }
  n = v44[1];
  *v44 = (unsigned __int64)(v44 + 2);
  v44[1] = 0;
  *((_BYTE *)v44 + 16) = 0;
  v46 = n;
  v47 = v93;
  v48 = src;
  if ( n + v93 > v94 )
  {
    v73 = n;
    v76 = src;
    sub_C8D290((__int64)&v92, v95, n + v93, 1u, n, v45);
    v47 = v93;
    v46 = v73;
    v48 = v76;
  }
  if ( v46 )
  {
    v74 = v46;
    memcpy((char *)v92 + v47, v48, v46);
    v47 = v93;
    v46 = v74;
  }
  v49 = v47 + v46;
  v93 = v49;
  if ( src != &v84 )
    j_j___libc_free_0((unsigned __int64)src);
  if ( (void **)v80[0] != v81 )
    j_j___libc_free_0(v80[0]);
  if ( v77 != (_BYTE *)v79 )
    j_j___libc_free_0((unsigned __int64)v77);
  if ( *v2 == 85 && (v50 = *((_QWORD *)v2 - 4)) != 0 && !*(_BYTE *)v50 && *(_QWORD *)(v50 + 24) == *((_QWORD *)v2 + 10) )
  {
    v51 = sub_BD5D20(v50);
    v52 = v93;
    v54 = v53;
    v55 = v51;
    v56 = v93 + v53;
    if ( v56 > v94 )
    {
      sub_C8D290((__int64)&v92, v95, v56, 1u, v49, v45);
      v52 = v93;
    }
    if ( v54 )
    {
      memcpy((char *)v92 + v52, v55, v54);
      v52 = v93;
    }
    v57 = v54 + v52;
    v93 = v54 + v52;
  }
  else
  {
    v57 = v93;
  }
  if ( v57 + 1 > v94 )
  {
    sub_C8D290((__int64)&v92, v95, v57 + 1, 1u, v49, v45);
    v57 = v93;
  }
  *((_BYTE *)v92 + v57) = 40;
  v66 = ++v93;
  if ( (_DWORD)v104 )
  {
    v75 = v2;
    v60 = 0;
    v61 = 0;
    do
    {
      v63 = *(_QWORD *)(v61 + v103 + 8);
      v64 = *(const void **)(v61 + v103);
      if ( v63 + v66 > v94 )
      {
        sub_C8D290((__int64)&v92, v95, v63 + v66, 1u, v49, v45);
        v66 = v93;
      }
      if ( v63 )
      {
        memcpy((char *)v92 + v66, v64, v63);
        v66 = v93;
      }
      v62 = (unsigned int)v104;
      v66 += v63;
      v93 = v66;
      if ( (unsigned __int64)(unsigned int)v104 - 1 > v60 )
      {
        if ( v66 + 2 > v94 )
        {
          sub_C8D290((__int64)&v92, v95, v66 + 2, 1u, v49, v45);
          v66 = v93;
        }
        *(_WORD *)((char *)v92 + v66) = 8236;
        v62 = (unsigned int)v104;
        v66 = v93 + 2;
        v93 += 2;
      }
      ++v60;
      v61 += 88;
    }
    while ( v60 < v62 );
    v2 = v75;
  }
  if ( v66 + 1 > v94 )
  {
    sub_C8D290((__int64)&v92, v95, v66 + 1, 1u, v49, v45);
    v66 = v93;
  }
  *((_BYTE *)v92 + v66) = 41;
  v85 = 261;
  ++v93;
  src = v92;
  n = v93;
  sub_BD6B50(v2, (const char **)&src);
  if ( v92 != v95 )
    _libc_free((unsigned __int64)v92);
  if ( v90 != (unsigned __int64 *)&v92 )
    _libc_free((unsigned __int64)v90);
  sub_C7D6A0(v87, 4LL * v89, 4);
  if ( !v101 )
    _libc_free((unsigned __int64)v98);
  v58 = v103;
  v59 = (unsigned __int64 *)(v103 + 88LL * (unsigned int)v104);
  if ( (unsigned __int64 *)v103 != v59 )
  {
    do
    {
      v59 -= 11;
      if ( (unsigned __int64 *)*v59 != v59 + 3 )
        _libc_free(*v59);
    }
    while ( (unsigned __int64 *)v58 != v59 );
    v59 = (unsigned __int64 *)v103;
  }
  if ( v59 != (unsigned __int64 *)v105 )
    _libc_free((unsigned __int64)v59);
}
