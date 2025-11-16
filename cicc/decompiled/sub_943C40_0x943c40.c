// Function: sub_943C40
// Address: 0x943c40
//
const char *__fastcall sub_943C40(__int64 a1, __int64 a2)
{
  const char *v4; // rax
  __int64 v5; // rdx
  char *v6; // r13
  const char *v7; // rax
  __int64 v8; // rdx
  char *v9; // r14
  __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // r10
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  __m128i *v17; // rax
  void **v18; // rcx
  __m128i *v19; // rdx
  size_t v20; // rcx
  __m128i *v21; // rdi
  size_t v22; // rdx
  __int64 v23; // rsi
  unsigned int v24; // r15d
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r14
  int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned int v39; // esi
  __int64 v40; // r8
  int v41; // r10d
  unsigned int v42; // r14d
  __int64 *v43; // rdx
  unsigned int v44; // edi
  __int64 *v45; // rax
  __int64 v46; // rcx
  __int64 *v47; // rax
  const char *result; // rax
  unsigned __int64 v49; // rcx
  __int8 *v50; // r14
  size_t v51; // r8
  _QWORD *v52; // rax
  unsigned __int64 v53; // rax
  char *v54; // rdi
  __int64 v55; // rax
  _QWORD *v56; // rdi
  int v57; // eax
  int v58; // ecx
  int v59; // eax
  int v60; // esi
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // r8
  int v64; // r10d
  __int64 *v65; // r9
  int v66; // eax
  int v67; // eax
  __int64 v68; // rdi
  __int64 v69; // r11
  int v70; // r9d
  __int64 *v71; // r8
  __int64 v72; // rsi
  __int64 v73; // [rsp+8h] [rbp-138h]
  size_t v74; // [rsp+10h] [rbp-130h]
  __int64 v75; // [rsp+20h] [rbp-120h]
  const __m128i *v76; // [rsp+28h] [rbp-118h]
  unsigned __int8 v77; // [rsp+37h] [rbp-109h] BYREF
  __int64 v78; // [rsp+38h] [rbp-108h] BYREF
  void *dest; // [rsp+40h] [rbp-100h] BYREF
  size_t v80; // [rsp+48h] [rbp-F8h]
  _QWORD v81[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v82[2]; // [rsp+60h] [rbp-E0h] BYREF
  _QWORD v83[2]; // [rsp+70h] [rbp-D0h] BYREF
  __m128i *v84; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v85; // [rsp+88h] [rbp-B8h]
  __m128i v86; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v87[2]; // [rsp+A0h] [rbp-A0h] BYREF
  _QWORD v88[2]; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD *v89; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v90; // [rsp+C8h] [rbp-78h]
  _QWORD v91[2]; // [rsp+D0h] [rbp-70h] BYREF
  void **p_src; // [rsp+E0h] [rbp-60h] BYREF
  size_t n; // [rsp+E8h] [rbp-58h]
  __m128i src; // [rsp+F0h] [rbp-50h] BYREF
  __int16 v95; // [rsp+100h] [rbp-40h]

  if ( sub_9439D0(a1, a2) )
    sub_91B8A0("unexpected: declaration for variable already exists!", (_DWORD *)(a2 + 64), 1);
  v80 = 0;
  dest = v81;
  LOBYTE(v81[0]) = 0;
  sub_72F9F0(a2, 0, &v77, &v78);
  v76 = sub_740200(a2);
  if ( v77 != 3 && v77 > 1u && !v76 && (v77 != 2 || *(_BYTE *)(*(_QWORD *)v78 + 48LL) > 1u) )
    sub_91B8A0("function-scope static variable is initialized with non-constant initializer!", (_DWORD *)(a2 + 64), 1);
  if ( !*(_QWORD *)(a2 + 8) )
  {
    v49 = sub_737880(a2);
    if ( !v49 )
    {
      src.m128i_i8[4] = 48;
      v50 = &src.m128i_i8[4];
      v87[0] = (__int64)v88;
LABEL_69:
      v51 = 1;
      LOBYTE(v88[0]) = *v50;
      v52 = v88;
LABEL_83:
      v87[1] = v51;
      *((_BYTE *)v52 + v51) = 0;
      goto LABEL_9;
    }
    v50 = &src.m128i_i8[5];
    do
    {
      *--v50 = v49 % 0xA + 48;
      v53 = v49;
      v49 /= 0xAu;
    }
    while ( v53 > 9 );
    v54 = (char *)(&src.m128i_u8[5] - (unsigned __int8 *)v50);
    v51 = &src.m128i_u8[5] - (unsigned __int8 *)v50;
    v87[0] = (__int64)v88;
    v89 = (_QWORD *)(&src.m128i_u8[5] - (unsigned __int8 *)v50);
    if ( (unsigned __int64)(&src.m128i_u8[5] - (unsigned __int8 *)v50) <= 0xF )
    {
      if ( v54 == (char *)1 )
        goto LABEL_69;
      if ( !v54 )
      {
        v52 = v88;
        goto LABEL_83;
      }
      v56 = v88;
    }
    else
    {
      v55 = sub_22409D0(v87, &v89, 0);
      v51 = &src.m128i_u8[5] - (unsigned __int8 *)v50;
      v87[0] = v55;
      v56 = (_QWORD *)v55;
      v88[0] = v89;
    }
    memcpy(v56, v50, v51);
    v51 = (size_t)v89;
    v52 = (_QWORD *)v87[0];
    goto LABEL_83;
  }
  v4 = (const char *)sub_91B6B0(a2);
  v5 = -1;
  v6 = (char *)v4;
  v87[0] = (__int64)v88;
  if ( v4 )
    v5 = (__int64)&v4[strlen(v4)];
  sub_943740(v87, v6, v5);
LABEL_9:
  sub_91B9C0((__int64 *)&v89, v87, a2);
  v7 = (const char *)sub_91B6C0(*(_QWORD *)(*(_QWORD *)(a1 + 528) + 32LL));
  v82[0] = (__int64)v83;
  v8 = -1;
  v9 = (char *)v7;
  if ( v7 )
    v8 = (__int64)&v7[strlen(v7)];
  sub_943740(v82, v9, v8);
  if ( v82[1] == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v11 = (__m128i *)sub_2241490(v82, "$", 1, v10);
  v84 = &v86;
  if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
  {
    v86 = _mm_loadu_si128(v11 + 1);
  }
  else
  {
    v84 = (__m128i *)v11->m128i_i64[0];
    v86.m128i_i64[0] = v11[1].m128i_i64[0];
  }
  v12 = v11->m128i_i64[1];
  v11[1].m128i_i8[0] = 0;
  v85 = v12;
  v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
  v13 = v84;
  v11->m128i_i64[1] = 0;
  v14 = 15;
  v15 = 15;
  if ( v13 != &v86 )
    v15 = v86.m128i_i64[0];
  v16 = v85 + v90;
  if ( v85 + v90 <= v15 )
    goto LABEL_20;
  if ( v89 != v91 )
    v14 = v91[0];
  if ( v16 <= v14 )
  {
    v17 = (__m128i *)sub_2241130(&v89, 0, 0, v13, v85);
    p_src = (void **)&src;
    v18 = (void **)v17->m128i_i64[0];
    v19 = v17 + 1;
    if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
      goto LABEL_21;
  }
  else
  {
LABEL_20:
    v17 = (__m128i *)sub_2241490(&v84, v89, v90, v16);
    p_src = (void **)&src;
    v18 = (void **)v17->m128i_i64[0];
    v19 = v17 + 1;
    if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
    {
LABEL_21:
      p_src = v18;
      src.m128i_i64[0] = v17[1].m128i_i64[0];
      goto LABEL_22;
    }
  }
  src = _mm_loadu_si128(v17 + 1);
LABEL_22:
  v20 = v17->m128i_u64[1];
  n = v20;
  v17->m128i_i64[0] = (__int64)v19;
  v17->m128i_i64[1] = 0;
  v17[1].m128i_i8[0] = 0;
  v21 = (__m128i *)dest;
  v22 = n;
  if ( p_src == (void **)&src )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src.m128i_i8[0];
      else
        memcpy(dest, &src, n);
      v22 = n;
      v21 = (__m128i *)dest;
    }
    v80 = v22;
    v21->m128i_i8[v22] = 0;
    v21 = (__m128i *)p_src;
  }
  else
  {
    v20 = src.m128i_i64[0];
    if ( dest == v81 )
    {
      dest = p_src;
      v80 = n;
      v81[0] = src.m128i_i64[0];
    }
    else
    {
      v23 = v81[0];
      dest = p_src;
      v80 = n;
      v81[0] = src.m128i_i64[0];
      if ( v21 )
      {
        p_src = (void **)v21;
        src.m128i_i64[0] = v23;
        goto LABEL_26;
      }
    }
    p_src = (void **)&src;
    v21 = &src;
  }
LABEL_26:
  n = 0;
  v21->m128i_i8[0] = 0;
  if ( p_src != (void **)&src )
    j_j___libc_free_0(p_src, src.m128i_i64[0] + 1);
  if ( v84 != &v86 )
    j_j___libc_free_0(v84, v86.m128i_i64[0] + 1);
  if ( (_QWORD *)v82[0] != v83 )
    j_j___libc_free_0(v82[0], v83[0] + 1LL);
  if ( v89 != v91 )
    j_j___libc_free_0(v89, v91[0] + 1LL);
  if ( (_QWORD *)v87[0] != v88 )
    j_j___libc_free_0(v87[0], v88[0] + 1LL);
  v24 = *(_BYTE *)(*(_QWORD *)(a1 + 192) + 32LL) & 0xF;
  if ( ((*(_BYTE *)(*(_QWORD *)(a1 + 192) + 32LL) + 14) & 0xFu) > 3 && v24 != 10 && v24 != 9 )
    v24 = 7;
  v75 = sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(a2 + 120), v22, v20);
  if ( (*(_BYTE *)(a2 + 156) & 2) != 0 )
  {
    v27 = v75;
    v76 = (const __m128i *)sub_91DAF0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 120), v25, v26);
  }
  else
  {
    v27 = v75;
    if ( v76 )
    {
      v76 = (const __m128i *)sub_91FFE0(a1, v76, *(_QWORD *)(a2 + 120), v26);
      if ( (unsigned __int8)sub_918D80(*(_QWORD *)(a1 + 32) + 8LL, *(_QWORD *)(a2 + 120)) )
        v27 = v76->m128i_i64[1];
    }
  }
  v73 = **(_QWORD **)(a1 + 32);
  v74 = sub_AD6530(v27);
  v95 = 260;
  p_src = &dest;
  v28 = sub_91C310(a2);
  BYTE4(v89) = 1;
  LODWORD(v89) = v28;
  v29 = unk_3F0FAE8;
  v30 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v30 )
  {
    v29 = v73;
    sub_B30000(v30, v73, v27, 0, v24, v74, &p_src, 0, 0, v89, 0);
  }
  v31 = (__int64)v76;
  if ( v76 )
  {
    v29 = v30;
    sub_90A710(*(_QWORD *)(a1 + 32), v30, (__int64)v76, a2);
  }
  LODWORD(v32) = sub_91CB50(a2, v29, v31);
  v33 = 0;
  if ( (_DWORD)v32 )
  {
    _BitScanReverse64((unsigned __int64 *)&v32, (unsigned int)v32);
    LOBYTE(v33) = 63 - (v32 ^ 0x3F);
    BYTE1(v33) = 1;
  }
  v34 = v30;
  sub_B2F740(v30, v33);
  if ( v27 != v75 )
  {
    v35 = sub_BCE760(v75, 0);
    v34 = sub_AD4C90(v30, v35, 0, v36, v37, v38);
  }
  v39 = *(_DWORD *)(a1 + 24);
  if ( !v39 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_105;
  }
  v40 = *(_QWORD *)(a1 + 8);
  v41 = 1;
  v42 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v43 = 0;
  v44 = (v39 - 1) & v42;
  v45 = (__int64 *)(v40 + 16LL * v44);
  v46 = *v45;
  if ( a2 == *v45 )
  {
LABEL_54:
    v47 = v45 + 1;
    goto LABEL_55;
  }
  while ( v46 != -4096 )
  {
    if ( !v43 && v46 == -8192 )
      v43 = v45;
    v44 = (v39 - 1) & (v41 + v44);
    v45 = (__int64 *)(v40 + 16LL * v44);
    v46 = *v45;
    if ( a2 == *v45 )
      goto LABEL_54;
    ++v41;
  }
  if ( !v43 )
    v43 = v45;
  v57 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v58 = v57 + 1;
  if ( 4 * (v57 + 1) >= 3 * v39 )
  {
LABEL_105:
    sub_9437F0(a1, 2 * v39);
    v59 = *(_DWORD *)(a1 + 24);
    if ( v59 )
    {
      v60 = v59 - 1;
      v61 = *(_QWORD *)(a1 + 8);
      LODWORD(v62) = (v59 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v58 = *(_DWORD *)(a1 + 16) + 1;
      v43 = (__int64 *)(v61 + 16LL * (unsigned int)v62);
      v63 = *v43;
      if ( a2 != *v43 )
      {
        v64 = 1;
        v65 = 0;
        while ( v63 != -4096 )
        {
          if ( !v65 && v63 == -8192 )
            v65 = v43;
          v62 = v60 & (unsigned int)(v62 + v64);
          v43 = (__int64 *)(v61 + 16 * v62);
          v63 = *v43;
          if ( a2 == *v43 )
            goto LABEL_97;
          ++v64;
        }
        if ( v65 )
          v43 = v65;
      }
      goto LABEL_97;
    }
    goto LABEL_130;
  }
  if ( v39 - *(_DWORD *)(a1 + 20) - v58 <= v39 >> 3 )
  {
    sub_9437F0(a1, v39);
    v66 = *(_DWORD *)(a1 + 24);
    if ( v66 )
    {
      v67 = v66 - 1;
      v68 = *(_QWORD *)(a1 + 8);
      LODWORD(v69) = v67 & v42;
      v70 = 1;
      v71 = 0;
      v58 = *(_DWORD *)(a1 + 16) + 1;
      v43 = (__int64 *)(v68 + 16LL * (v67 & v42));
      v72 = *v43;
      if ( a2 != *v43 )
      {
        while ( v72 != -4096 )
        {
          if ( !v71 && v72 == -8192 )
            v71 = v43;
          v69 = v67 & (unsigned int)(v69 + v70);
          v43 = (__int64 *)(v68 + 16 * v69);
          v72 = *v43;
          if ( a2 == *v43 )
            goto LABEL_97;
          ++v70;
        }
        if ( v71 )
          v43 = v71;
      }
      goto LABEL_97;
    }
LABEL_130:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_97:
  *(_DWORD *)(a1 + 16) = v58;
  if ( *v43 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v43 = a2;
  v47 = v43 + 1;
  v43[1] = 0;
LABEL_55:
  *v47 = v34;
  if ( sub_91B5A0(a2) )
    sub_914140(*(_QWORD *)(a1 + 32), v30);
  result = (const char *)dword_4D046B4;
  if ( dword_4D046B4 )
    result = sub_943430(*(_QWORD **)(*(_QWORD *)(a1 + 32) + 368LL), v30, a2, 0);
  if ( dest != v81 )
    return (const char *)j_j___libc_free_0(dest, v81[0] + 1LL);
  return result;
}
