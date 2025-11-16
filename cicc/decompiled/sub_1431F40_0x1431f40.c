// Function: sub_1431F40
// Address: 0x1431f40
//
void __fastcall sub_1431F40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rbx
  __m128i *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rsi
  size_t *v14; // rdi
  void *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  char v18; // al
  size_t *v19; // rbx
  _BYTE *v20; // r13
  unsigned int v21; // esi
  __int64 v22; // r9
  unsigned int v23; // r8d
  _QWORD *v24; // rax
  __int64 v25; // r11
  __int64 v26; // rdi
  int v27; // r12d
  _QWORD *v28; // rcx
  int v29; // eax
  unsigned int v30; // eax
  __int64 v31; // r9
  unsigned __int32 v32; // r8d
  int v33; // edx
  __int64 v34; // rdi
  __int64 v35; // r10
  int v36; // r12d
  _QWORD *v37; // rbx
  __m128i *v38; // rsi
  unsigned int v39; // esi
  int v40; // eax
  int v41; // eax
  void **v42; // rax
  void **p_src; // rsi
  size_t v44; // rcx
  size_t v45; // r15
  size_t v46; // rdx
  signed __int64 v47; // r14
  __int64 v48; // rax
  void *v49; // rdi
  bool v50; // r15
  __m128i *v51; // r14
  bool v52; // zf
  unsigned __int64 v53; // rbx
  void *v54; // rcx
  signed __int64 v55; // rbx
  int v56; // eax
  int v57; // eax
  unsigned int v58; // eax
  __int64 v59; // r11
  unsigned __int32 v60; // r9d
  __int64 v61; // r8
  __int64 v62; // rsi
  int v63; // r12d
  __m128i *v64; // [rsp+8h] [rbp-A8h]
  size_t v65; // [rsp+8h] [rbp-A8h]
  __int64 v67; // [rsp+18h] [rbp-98h] BYREF
  size_t *v68; // [rsp+28h] [rbp-88h] BYREF
  void *v69; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v70; // [rsp+38h] [rbp-78h]
  _BYTE *v71; // [rsp+40h] [rbp-70h]
  __m128i v72; // [rsp+50h] [rbp-60h] BYREF
  void *src; // [rsp+60h] [rbp-50h] BYREF
  _BYTE *v74; // [rsp+68h] [rbp-48h]
  _BYTE *v75; // [rsp+70h] [rbp-40h]

  v67 = a2;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v8 = sub_1389B50(&v67);
  v9 = (v67 & 0xFFFFFFFFFFFFFFF8LL) + 24 * (1LL - (*(_DWORD *)((v67 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  if ( v8 == v9 )
  {
LABEL_10:
    v72.m128i_i64[0] = a3;
    v14 = (size_t *)a5;
    v72.m128i_i64[1] = a1;
    v15 = v69;
    v69 = 0;
    src = v15;
    v16 = v70;
    v70 = 0;
    v74 = v16;
    v17 = v71;
    v71 = 0;
    v75 = v17;
    v18 = sub_142F320((_DWORD *)a5, (const void **)&v72, &v68);
    v19 = v68;
    if ( v18 )
    {
LABEL_11:
      v20 = src;
      goto LABEL_12;
    }
    v39 = *(_DWORD *)(a5 + 24);
    v40 = *(_DWORD *)(a5 + 16);
    ++*(_QWORD *)a5;
    v41 = v40 + 1;
    if ( 4 * v41 >= 3 * v39 )
    {
      v39 *= 2;
    }
    else if ( v39 - *(_DWORD *)(a5 + 20) - v41 > v39 >> 3 )
    {
      goto LABEL_50;
    }
    sub_1431E40(a5, v39);
    v14 = (size_t *)a5;
    sub_142F320((_DWORD *)a5, (const void **)&v72, &v68);
    v19 = v68;
    v41 = *(_DWORD *)(a5 + 16) + 1;
LABEL_50:
    *(_DWORD *)(a5 + 16) = v41;
    v42 = (void **)v19[3];
    p_src = (void **)v19[2];
    v44 = *v19;
    v45 = v19[1];
    v46 = (char *)v42 - (char *)p_src;
    v47 = (char *)v42 - (char *)p_src;
    if ( v42 == p_src )
    {
      v49 = 0;
    }
    else
    {
      if ( v46 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_97;
      v65 = *v19;
      v48 = sub_22077B0(v46);
      p_src = (void **)v19[2];
      v44 = v65;
      v49 = (void *)v48;
      v42 = (void **)v19[3];
      v46 = (char *)v42 - (char *)p_src;
    }
    v50 = __PAIR128__(v44, v46) == 0 && v45 == -1;
    if ( p_src == v42 )
    {
      if ( !v50 )
      {
        if ( !v49 )
          goto LABEL_56;
        goto LABEL_55;
      }
      if ( !v49 )
        goto LABEL_57;
    }
    else
    {
      v49 = memmove(v49, p_src, v46);
      if ( !v50 )
      {
LABEL_55:
        j_j___libc_free_0(v49, v47);
LABEL_56:
        --*(_DWORD *)(a5 + 20);
LABEL_57:
        p_src = &src;
        v14 = v19 + 2;
        *(__m128i *)v19 = _mm_loadu_si128(&v72);
        sub_142B670((__int64)(v19 + 2), (char **)&src);
        v51 = *(__m128i **)(a5 + 40);
        if ( v51 == *(__m128i **)(a5 + 48) )
        {
          sub_142E3D0(a5 + 32, *(const __m128i **)(a5 + 40), &v72);
          goto LABEL_11;
        }
        if ( !v51 )
        {
          v20 = src;
LABEL_65:
          *(_QWORD *)(a5 + 40) = (char *)v51 + 40;
LABEL_12:
          if ( v20 )
            j_j___libc_free_0(v20, v75 - v20);
          goto LABEL_14;
        }
        *v51 = _mm_loadu_si128(&v72);
        v53 = v74 - (_BYTE *)src;
        v52 = v74 == src;
        v51[1].m128i_i64[0] = 0;
        v51[1].m128i_i64[1] = 0;
        v51[2].m128i_i64[0] = 0;
        if ( v52 )
        {
          v54 = 0;
          goto LABEL_62;
        }
        if ( v53 <= 0x7FFFFFFFFFFFFFF8LL )
        {
          v54 = (void *)sub_22077B0(v53);
LABEL_62:
          v51[1].m128i_i64[0] = (__int64)v54;
          v51[2].m128i_i64[0] = (__int64)v54 + v53;
          v51[1].m128i_i64[1] = (__int64)v54;
          v20 = src;
          v55 = v74 - (_BYTE *)src;
          if ( v74 != src )
            v54 = memmove(v54, src, v74 - (_BYTE *)src);
          v51[1].m128i_i64[1] = (__int64)v54 + v55;
          v51 = *(__m128i **)(a5 + 40);
          goto LABEL_65;
        }
LABEL_97:
        sub_4261EA(v14, p_src, v46);
      }
    }
    j_j___libc_free_0(v49, v47);
    goto LABEL_57;
  }
  v10 = &v72;
  while ( 1 )
  {
    v11 = *(_QWORD *)v9;
    if ( *(_BYTE *)(*(_QWORD *)v9 + 16LL) != 13 || *(_DWORD *)(v11 + 32) > 0x40u )
      break;
    v12 = *(_QWORD *)(v11 + 24);
    v13 = v70;
    v72.m128i_i64[0] = v12;
    if ( v70 == v71 )
    {
      v64 = v10;
      sub_A235E0((__int64)&v69, v70, v10);
      v10 = v64;
    }
    else
    {
      if ( v70 )
      {
        *(_QWORD *)v70 = v12;
        v13 = v70;
      }
      v70 = v13 + 8;
    }
    v9 += 24LL;
    if ( v8 == v9 )
      goto LABEL_10;
  }
  v21 = *(_DWORD *)(a4 + 24);
  v72.m128i_i64[0] = a3;
  v72.m128i_i64[1] = a1;
  if ( !v21 )
  {
    ++*(_QWORD *)a4;
LABEL_32:
    sub_14314B0(a4, 2 * v21);
    v29 = *(_DWORD *)(a4 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a4 + 8);
      v32 = v72.m128i_i32[0] & v30;
      v33 = *(_DWORD *)(a4 + 16) + 1;
      v28 = (_QWORD *)(v31 + 16LL * (v72.m128i_i32[0] & v30));
      v34 = *v28;
      v35 = v28[1];
      if ( *(_OWORD *)&v72 == *(_OWORD *)v28 )
        goto LABEL_40;
      v36 = 1;
      v37 = 0;
      while ( 1 )
      {
        if ( !v34 )
        {
          if ( v35 == -1 )
          {
LABEL_38:
            if ( v37 )
              v28 = v37;
            goto LABEL_40;
          }
          if ( v35 == -2 && !v37 )
            v37 = v28;
        }
        v32 = v30 & (v36 + v32);
        v28 = (_QWORD *)(v31 + 16LL * v32);
        v35 = v28[1];
        v34 = *v28;
        if ( *(_OWORD *)v28 == *(_OWORD *)&v72 )
          goto LABEL_40;
        ++v36;
      }
    }
LABEL_98:
    ++*(_DWORD *)(a4 + 16);
    BUG();
  }
  v22 = *(_QWORD *)(a4 + 8);
  v23 = a3 & (v21 - 1);
  v24 = (_QWORD *)(v22 + 16LL * v23);
  v25 = v24[1];
  v26 = *v24;
  if ( a1 == v25 && a3 == v26 )
    goto LABEL_14;
  v27 = 1;
  v28 = 0;
  while ( 1 )
  {
    if ( v26 )
      goto LABEL_22;
    if ( v25 == -1 )
      break;
    if ( v25 == -2 && !v28 )
      v28 = v24;
LABEL_22:
    v23 = (v21 - 1) & (v27 + v23);
    v24 = (_QWORD *)(v22 + 16LL * v23);
    v26 = *v24;
    v25 = v24[1];
    if ( a3 == *v24 && a1 == v25 )
      goto LABEL_14;
    ++v27;
  }
  if ( !v28 )
    v28 = v24;
  v56 = *(_DWORD *)(a4 + 16);
  ++*(_QWORD *)a4;
  v33 = v56 + 1;
  if ( 4 * (v56 + 1) >= 3 * v21 )
    goto LABEL_32;
  if ( v21 - *(_DWORD *)(a4 + 20) - v33 > v21 >> 3 )
    goto LABEL_40;
  sub_14314B0(a4, v21);
  v57 = *(_DWORD *)(a4 + 24);
  if ( !v57 )
    goto LABEL_98;
  v58 = v57 - 1;
  v59 = *(_QWORD *)(a4 + 8);
  v60 = v72.m128i_i32[0] & v58;
  v33 = *(_DWORD *)(a4 + 16) + 1;
  v28 = (_QWORD *)(v59 + 16LL * (v72.m128i_i32[0] & v58));
  v61 = v28[1];
  v62 = *v28;
  if ( *(_OWORD *)&v72 != *(_OWORD *)v28 )
  {
    v63 = 1;
    v37 = 0;
    while ( 1 )
    {
      if ( !v62 )
      {
        if ( v61 == -1 )
          goto LABEL_38;
        if ( v61 == -2 && !v37 )
          v37 = v28;
      }
      v60 = v58 & (v63 + v60);
      v28 = (_QWORD *)(v59 + 16LL * v60);
      v61 = v28[1];
      v62 = *v28;
      if ( *(_OWORD *)&v72 == *(_OWORD *)v28 )
        break;
      ++v63;
    }
  }
LABEL_40:
  *(_DWORD *)(a4 + 16) = v33;
  if ( v28[1] != -1 || *v28 )
    --*(_DWORD *)(a4 + 20);
  *(__m128i *)v28 = _mm_loadu_si128(&v72);
  v38 = *(__m128i **)(a4 + 40);
  if ( v38 == *(__m128i **)(a4 + 48) )
  {
    sub_142E240(a4 + 32, v38, &v72);
  }
  else
  {
    if ( v38 )
    {
      *v38 = _mm_loadu_si128(&v72);
      v38 = *(__m128i **)(a4 + 40);
    }
    *(_QWORD *)(a4 + 40) = v38 + 1;
  }
LABEL_14:
  if ( v69 )
    j_j___libc_free_0(v69, v71 - (_BYTE *)v69);
}
