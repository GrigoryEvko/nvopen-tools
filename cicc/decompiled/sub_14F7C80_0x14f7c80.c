// Function: sub_14F7C80
// Address: 0x14f7c80
//
__int64 __fastcall sub_14F7C80(_QWORD *a1, unsigned __int64 a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v5; // r14
  __int64 v7; // rdx
  _BYTE *v8; // rax
  __int64 v9; // r13
  _QWORD *v10; // r12
  __m128i *v11; // rsi
  _QWORD *v12; // r14
  size_t v13; // r12
  size_t v14; // rbx
  size_t v15; // rdx
  signed __int64 v16; // rax
  size_t v17; // rcx
  __m128i *v18; // r15
  size_t v19; // r13
  size_t v20; // rdx
  signed __int64 v21; // rax
  __int64 result; // rax
  unsigned __int64 v23; // r15
  _QWORD *v24; // r13
  unsigned __int64 v25; // r8
  __int64 v26; // rbx
  unsigned __int64 *v27; // r12
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  __int64 v31; // rdx
  _BYTE *v32; // rsi
  __m128i *v33; // rdi
  size_t v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rax
  _QWORD *v38; // rsi
  __int64 v39; // rbx
  signed __int64 v40; // r14
  _QWORD *v41; // r15
  _QWORD *v42; // rcx
  unsigned __int64 *v43; // r12
  unsigned __int64 *v44; // rsi
  char *v45; // rdi
  char *v46; // rax
  _QWORD *v47; // rdx
  _QWORD *v48; // rdx
  signed __int64 v49; // rax
  _QWORD *v50; // rsi
  _QWORD *v51; // rax
  __int64 v52; // rax
  _QWORD *v53; // rax
  unsigned __int64 *v54; // rdx
  _QWORD *v55; // rcx
  _BOOL8 v56; // rdi
  _QWORD *v57; // rax
  signed __int64 v58; // rsi
  unsigned __int64 *v59; // rsi
  unsigned __int64 *v60; // rax
  unsigned __int64 *v61; // rdx
  unsigned __int64 *v62; // r14
  _BOOL8 v63; // rdi
  size_t v64; // rdx
  unsigned __int64 v65; // rdi
  unsigned __int64 *v66; // rdi
  _QWORD *v67; // r13
  size_t v68; // r8
  __int64 v69; // rax
  _QWORD *v70; // rdx
  __int64 v71; // r13
  _QWORD *v72; // r15
  void *v73; // r8
  __int64 v74; // rdi
  __int64 v75; // rdi
  size_t v76; // r13
  size_t v77; // rdx
  int v78; // eax
  unsigned int v79; // edi
  __int64 v80; // r8
  unsigned __int64 *v83; // [rsp+28h] [rbp-A8h]
  _QWORD *v84; // [rsp+30h] [rbp-A0h]
  _QWORD *v85; // [rsp+40h] [rbp-90h]
  unsigned __int64 *src; // [rsp+48h] [rbp-88h]
  _QWORD *srca; // [rsp+48h] [rbp-88h]
  unsigned __int64 *srcb; // [rsp+48h] [rbp-88h]
  __int64 v89; // [rsp+50h] [rbp-80h]
  __int64 v90; // [rsp+58h] [rbp-78h]
  unsigned __int64 *v91; // [rsp+60h] [rbp-70h]
  unsigned __int64 *v92; // [rsp+68h] [rbp-68h]
  _QWORD *s2; // [rsp+70h] [rbp-60h]
  void *s2a; // [rsp+70h] [rbp-60h]
  void *s2b; // [rsp+70h] [rbp-60h]
  _QWORD *v96; // [rsp+78h] [rbp-58h]
  size_t v97; // [rsp+78h] [rbp-58h]
  __m128i *v98; // [rsp+78h] [rbp-58h]
  __int64 v99; // [rsp+78h] [rbp-58h]
  void *v100; // [rsp+78h] [rbp-58h]
  void *s1; // [rsp+80h] [rbp-50h] BYREF
  size_t n; // [rsp+88h] [rbp-48h]
  __m128i v103[4]; // [rsp+90h] [rbp-40h] BYREF

  v5 = a1;
  v7 = a1[1];
  v8 = (_BYTE *)(*a1 + a3);
  if ( v8 )
  {
    s1 = v103;
    sub_14E9CA0((__int64 *)&s1, v8, (__int64)&v8[v7]);
  }
  else
  {
    n = 0;
    s1 = v103;
    v103[0].m128i_i8[0] = 0;
  }
  v9 = a4[12];
  v10 = a4 + 11;
  if ( !v9 )
  {
    v90 = (__int64)(a4 + 11);
    goto LABEL_110;
  }
  v90 = (__int64)(a4 + 11);
  v96 = a4;
  v11 = (__m128i *)s1;
  v12 = a4 + 11;
  v13 = n;
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v9 + 40);
      v15 = v13;
      if ( v14 <= v13 )
        v15 = *(_QWORD *)(v9 + 40);
      if ( v15 )
      {
        LODWORD(v16) = memcmp(*(const void **)(v9 + 32), v11, v15);
        if ( (_DWORD)v16 )
          goto LABEL_12;
      }
      v16 = v14 - v13;
      if ( (__int64)(v14 - v13) >= 0x80000000LL )
        break;
      if ( v16 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_12:
        if ( (int)v16 >= 0 )
          break;
      }
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v9 )
        goto LABEL_14;
    }
    v90 = v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v9 );
LABEL_14:
  v17 = v13;
  v10 = v12;
  a4 = v96;
  v5 = a1;
  v18 = v11;
  if ( v10 == (_QWORD *)v90 )
    goto LABEL_110;
  v19 = *(_QWORD *)(v90 + 40);
  v20 = v17;
  if ( v19 <= v17 )
    v20 = *(_QWORD *)(v90 + 40);
  if ( v20 && (v97 = v17, LODWORD(v21) = memcmp(v11, *(const void **)(v90 + 32), v20), v17 = v97, (_DWORD)v21) )
  {
LABEL_21:
    if ( (int)v21 < 0 )
      goto LABEL_110;
  }
  else
  {
    v21 = v17 - v19;
    if ( (__int64)(v17 - v19) <= 0x7FFFFFFF )
    {
      if ( v21 >= (__int64)0xFFFFFFFF80000000LL )
        goto LABEL_21;
LABEL_110:
      v67 = (_QWORD *)v90;
      v90 = sub_22077B0(152);
      *(_QWORD *)(v90 + 32) = v90 + 48;
      if ( s1 == v103 )
      {
        *(__m128i *)(v90 + 48) = _mm_load_si128(v103);
      }
      else
      {
        *(_QWORD *)(v90 + 32) = s1;
        *(_QWORD *)(v90 + 48) = v103[0].m128i_i64[0];
      }
      v103[0].m128i_i8[0] = 0;
      v68 = n;
      n = 0;
      s1 = v103;
      *(_QWORD *)(v90 + 40) = v68;
      memset((void *)(v90 + 64), 0, 0x58u);
      s2b = (void *)v68;
      *(_QWORD *)(v90 + 128) = v90 + 112;
      *(_QWORD *)(v90 + 136) = v90 + 112;
      v69 = sub_14F61B0(a4 + 10, v67, v90 + 32);
      v71 = v69;
      v72 = v70;
      if ( v70 )
      {
        if ( v69 || (v73 = s2b, v10 == v70) )
        {
LABEL_118:
          v74 = 1;
          goto LABEL_119;
        }
        v77 = v70[5];
        v76 = v77;
        if ( (unsigned __int64)s2b <= v77 )
          v77 = (size_t)s2b;
        if ( v77 && (v78 = memcmp(*(const void **)(v90 + 32), (const void *)v72[4], v77), v73 = s2b, (v79 = v78) != 0) )
        {
LABEL_131:
          v74 = v79 >> 31;
        }
        else
        {
          v80 = (__int64)v73 - v76;
          v74 = 0;
          if ( v80 <= 0x7FFFFFFF )
          {
            if ( v80 < (__int64)0xFFFFFFFF80000000LL )
              goto LABEL_118;
            v79 = v80;
            goto LABEL_131;
          }
        }
LABEL_119:
        sub_220F040(v74, v90, v72, v10);
        ++a4[15];
      }
      else
      {
        sub_14EAD40(0);
        v75 = *(_QWORD *)(v90 + 32);
        if ( v90 + 48 != v75 )
          j_j___libc_free_0(v75, *(_QWORD *)(v90 + 48) + 1LL);
        j_j___libc_free_0(v90, 152);
        v90 = v71;
      }
      v18 = (__m128i *)s1;
    }
  }
  if ( v18 != v103 )
    j_j___libc_free_0(v18, v103[0].m128i_i64[0] + 1);
  *(_DWORD *)(v90 + 64) = v5[2];
  *(_DWORD *)(v90 + 68) = v5[3];
  *(_QWORD *)(v90 + 72) = v5[4];
  *(_QWORD *)(v90 + 80) = v5[5];
  *(_BYTE *)(v90 + 88) = v5[6];
  *(_QWORD *)(v90 + 96) = v5[7];
  result = v90 + 104;
  v83 = (unsigned __int64 *)(v90 + 112);
  if ( a2 <= 8 )
    return result;
  v23 = 8;
  v24 = v5;
  while ( 2 )
  {
    v25 = v24[v23];
    v26 = v23;
    v27 = (unsigned __int64 *)(v90 + 112);
    v28 = *(unsigned __int64 **)(v90 + 120);
    if ( !v28 )
      goto LABEL_92;
    do
    {
      while ( 1 )
      {
        v29 = v28[2];
        v30 = v28[3];
        if ( v25 <= v28[4] )
          break;
        v28 = (unsigned __int64 *)v28[3];
        if ( !v30 )
          goto LABEL_31;
      }
      v27 = v28;
      v28 = (unsigned __int64 *)v28[2];
    }
    while ( v29 );
LABEL_31:
    if ( v27 == v83 || v25 < v27[4] )
    {
LABEL_92:
      v100 = (void *)v24[v23];
      v59 = v27;
      v27 = (unsigned __int64 *)sub_22077B0(128);
      v27[4] = (unsigned __int64)v100;
      memset(v27 + 5, 0, 0x58u);
      s2a = v100;
      v98 = (__m128i *)(v27 + 8);
      v27[6] = (unsigned __int64)(v27 + 8);
      v27[13] = (unsigned __int64)(v27 + 11);
      v27[14] = (unsigned __int64)(v27 + 11);
      v60 = sub_14F7B80((_QWORD *)(v90 + 104), v59, v27 + 4);
      v62 = v60;
      if ( v61 )
      {
        v63 = v83 == v61 || v60 || (unsigned __int64)s2a < v61[4];
        sub_220F040(v63, v27, v61, v83);
        ++*(_QWORD *)(v90 + 144);
      }
      else
      {
        sub_14EAA70(0);
        v65 = v27[6];
        if ( v98 != (__m128i *)v65 )
          j_j___libc_free_0(v65, v27[8] + 1);
        v66 = v27;
        v27 = v62;
        j_j___libc_free_0(v66, 128);
        v98 = (__m128i *)(v62 + 8);
      }
    }
    else
    {
      v98 = (__m128i *)(v27 + 8);
    }
    *((_DWORD *)v27 + 10) = v24[v26 + 1];
    v31 = v24[v26 + 3];
    v32 = (_BYTE *)(v24[v26 + 2] + a3);
    s1 = v103;
    sub_14E9CA0((__int64 *)&s1, v32, (__int64)&v32[v31]);
    v33 = (__m128i *)v27[6];
    if ( s1 == v103 )
    {
      v64 = n;
      if ( n )
      {
        if ( n == 1 )
          v33->m128i_i8[0] = v103[0].m128i_i8[0];
        else
          memcpy(v33, v103, n);
        v64 = n;
        v33 = (__m128i *)v27[6];
      }
      v27[7] = v64;
      v33->m128i_i8[v64] = 0;
      v33 = (__m128i *)s1;
    }
    else
    {
      v34 = n;
      v35 = v103[0].m128i_i64[0];
      if ( v33 == v98 )
      {
        v27[6] = (unsigned __int64)s1;
        v27[7] = v34;
        v27[8] = v35;
      }
      else
      {
        v36 = v27[8];
        v27[6] = (unsigned __int64)s1;
        v27[7] = v34;
        v27[8] = v35;
        if ( v33 )
        {
          s1 = v33;
          v103[0].m128i_i64[0] = v36;
          goto LABEL_38;
        }
      }
      s1 = v103;
      v33 = v103;
    }
LABEL_38:
    n = 0;
    v33->m128i_i8[0] = 0;
    if ( s1 != v103 )
      j_j___libc_free_0(s1, v103[0].m128i_i64[0] + 1);
    result = v23 + 5;
    v89 = v24[v26 + 4];
    if ( !v89 )
    {
      v23 += 5LL;
      goto LABEL_82;
    }
    v37 = v26 * 8 + 48;
    v38 = &v24[v26 + 6];
    v39 = v23 + v24[v26 + 5] + 6;
    v40 = 8 * v39 - v37;
    s2 = &v24[v39];
    if ( (unsigned __int64)v40 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_72;
    v99 = 0;
    v84 = v27 + 10;
    v92 = v27 + 11;
    v91 = v27;
    while ( 2 )
    {
      v41 = 0;
      if ( v40 )
        v41 = (_QWORD *)sub_22077B0(v40);
      v42 = (_QWORD *)((char *)v41 + v40);
      if ( s2 != v38 )
      {
        memcpy(v41, v38, v40);
        v42 = (_QWORD *)((char *)v41 + v40);
      }
      v43 = v92;
      v44 = (unsigned __int64 *)v91[12];
      if ( !v44 )
        goto LABEL_65;
      while ( 2 )
      {
        v45 = (char *)v44[5];
        v46 = (char *)v44[4];
        if ( v45 - v46 > v40 )
          v45 = &v46[v40];
        v47 = v41;
        if ( v46 != v45 )
        {
          while ( *(_QWORD *)v46 >= *v47 )
          {
            if ( *(_QWORD *)v46 > *v47 )
              goto LABEL_74;
            v46 += 8;
            ++v47;
            if ( v45 == v46 )
              goto LABEL_73;
          }
          goto LABEL_55;
        }
LABEL_73:
        if ( v47 != v42 )
        {
LABEL_55:
          v44 = (unsigned __int64 *)v44[3];
          goto LABEL_56;
        }
LABEL_74:
        v43 = v44;
        v44 = (unsigned __int64 *)v44[2];
LABEL_56:
        if ( v44 )
          continue;
        break;
      }
      if ( v43 == v92 )
        goto LABEL_65;
      v48 = (_QWORD *)v43[4];
      v49 = v43[5] - (_QWORD)v48;
      v50 = (_QWORD *)((char *)v41 + v49);
      if ( v49 >= v40 )
        v50 = v42;
      if ( v41 == v50 )
      {
LABEL_75:
        if ( (_QWORD *)v43[5] != v48 )
          goto LABEL_65;
LABEL_76:
        if ( v41 )
          j_j___libc_free_0(v41, v40);
      }
      else
      {
        v51 = v41;
        while ( *v51 >= *v48 )
        {
          if ( *v51 > *v48 )
            goto LABEL_76;
          ++v51;
          ++v48;
          if ( v50 == v51 )
            goto LABEL_75;
        }
LABEL_65:
        v85 = v42;
        src = v43;
        v52 = sub_22077B0(80);
        *(_QWORD *)(v52 + 32) = v41;
        v43 = (unsigned __int64 *)v52;
        *(_QWORD *)(v52 + 40) = v85;
        *(_QWORD *)(v52 + 48) = v85;
        *(_DWORD *)(v52 + 56) = 0;
        *(_QWORD *)(v52 + 64) = 0;
        *(_QWORD *)(v52 + 72) = 0;
        v53 = sub_14F7820(v84, src, (char **)(v52 + 32));
        if ( v54 )
        {
          if ( v53 )
            goto LABEL_68;
          v55 = v85;
          if ( v92 == v54 )
            goto LABEL_68;
          v57 = (_QWORD *)v54[4];
          v58 = v54[5] - (_QWORD)v57;
          if ( v40 > v58 )
            v55 = (_QWORD *)((char *)v41 + v58);
          if ( v41 == v55 )
          {
LABEL_102:
            v56 = v54[5] != (_QWORD)v57;
          }
          else
          {
            while ( *v41 >= *v57 )
            {
              if ( *v41 > *v57 )
              {
                v56 = 0;
                goto LABEL_69;
              }
              ++v41;
              ++v57;
              if ( v55 == v41 )
                goto LABEL_102;
            }
LABEL_68:
            v56 = 1;
          }
LABEL_69:
          sub_220F040(v56, v43, v54, v92);
          ++v91[15];
        }
        else
        {
          if ( v41 )
          {
            srca = v53;
            j_j___libc_free_0(v41, v40);
            v53 = srca;
          }
          srcb = v53;
          j_j___libc_free_0(v43, 80);
          v43 = srcb;
        }
      }
      ++v99;
      v23 = v39 + 4;
      *((_DWORD *)v43 + 14) = *s2;
      result = 8 * (v39 + 1);
      v43[8] = *(_QWORD *)((char *)v24 + result);
      *((_DWORD *)v43 + 18) = *(_QWORD *)((char *)v24 + result + 8);
      *((_DWORD *)v43 + 19) = *(_QWORD *)((char *)v24 + result + 16);
      if ( v89 != v99 )
      {
        v38 = (_QWORD *)((char *)v24 + result + 32);
        v39 += *(_QWORD *)((char *)v24 + result + 24) + 5LL;
        v40 = 8 * v39 - (result + 32);
        s2 = &v24[v39];
        if ( (unsigned __int64)v40 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_72:
          sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
        continue;
      }
      break;
    }
LABEL_82:
    if ( a2 > v23 )
      continue;
    return result;
  }
}
