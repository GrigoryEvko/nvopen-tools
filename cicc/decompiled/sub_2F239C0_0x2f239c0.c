// Function: sub_2F239C0
// Address: 0x2f239c0
//
void __fastcall sub_2F239C0(__int64 a1, __int8 *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 *v6; // r13
  __int64 v7; // r14
  _BYTE *v8; // rsi
  __int64 v9; // rdx
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __m128i v12; // xmm7
  __int64 v13; // rax
  __m128i v14; // xmm7
  bool v15; // bl
  const __m128i *v16; // rbx
  __int8 *v17; // r13
  __int64 v18; // r14
  _BYTE *v19; // rsi
  __int64 v20; // rdx
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __m128i v23; // xmm2
  __int64 v24; // rax
  __m128i v25; // xmm3
  bool v26; // r15
  __int64 v27; // rax
  unsigned __int64 v28; // r14
  __int64 v29; // rdx
  _BYTE *v30; // rsi
  int v31; // edx
  _BYTE *v32; // rsi
  __m128i v33; // xmm0
  __int64 v34; // rdx
  __m128i v35; // xmm1
  bool v36; // bl
  _BYTE *v37; // rsi
  __int64 v38; // rdx
  _BYTE *v39; // rsi
  __int64 v40; // rdx
  __m128i v41; // xmm7
  __m128i v42; // xmm5
  bool v43; // bl
  __int64 v44; // rax
  size_t v45; // rax
  __m128i v46; // xmm4
  __m128i *v47; // r15
  __int64 v48; // rax
  __int64 v49; // rax
  _BYTE *v50; // rax
  __m128i *v51; // rdi
  size_t v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // r8
  int v55; // eax
  __m128i *v56; // rdi
  size_t v57; // rdx
  size_t v58; // rdx
  const __m128i *v59; // r14
  __int64 v60; // r13
  __int64 v61; // r9
  size_t v62; // rdi
  __m128i v63; // xmm6
  __int32 v64; // ecx
  __m128i v65; // xmm7
  __int64 v66; // rsi
  __m128i *v67; // rdx
  __m128i v68; // xmm4
  __m128i v69; // xmm5
  const void *v70; // r12
  __m128i *v71; // r14
  size_t v72; // rax
  __m128i v73; // xmm7
  __int64 v74; // rax
  const void *v75; // rax
  _BYTE *v76; // rax
  __int64 v77; // r15
  __m128i v78; // xmm7
  __int64 v79; // rax
  size_t v80; // rdx
  __int64 *v81; // [rsp+0h] [rbp-110h]
  const __m128i *v82; // [rsp+8h] [rbp-108h]
  __int64 v83; // [rsp+18h] [rbp-F8h]
  __int8 *v84; // [rsp+20h] [rbp-F0h]
  __m128i *dest; // [rsp+28h] [rbp-E8h]
  __int8 *v86; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v87; // [rsp+40h] [rbp-D0h]
  __int64 v88; // [rsp+58h] [rbp-B8h]
  __int64 v89; // [rsp+60h] [rbp-B0h]
  __int64 v90; // [rsp+60h] [rbp-B0h]
  __int64 v91; // [rsp+60h] [rbp-B0h]
  __int64 v92; // [rsp+60h] [rbp-B0h]
  __int64 v93; // [rsp+60h] [rbp-B0h]
  __m128i *v94; // [rsp+68h] [rbp-A8h] BYREF
  size_t v95; // [rsp+70h] [rbp-A0h]
  __m128i v96; // [rsp+78h] [rbp-98h] BYREF
  __m128i v97; // [rsp+88h] [rbp-88h] BYREF
  __int32 v98; // [rsp+98h] [rbp-78h]
  __int64 v99; // [rsp+A0h] [rbp-70h] BYREF
  __m128i *p_src; // [rsp+A8h] [rbp-68h] BYREF
  size_t n; // [rsp+B0h] [rbp-60h]
  __m128i src; // [rsp+B8h] [rbp-58h] BYREF
  __m128i v103; // [rsp+C8h] [rbp-48h] BYREF
  int v104; // [rsp+D8h] [rbp-38h]

  v3 = (__int64)&a2[-a1];
  v83 = a3;
  v84 = a2;
  if ( (__int64)&a2[-a1] <= 1024 )
    return;
  v4 = a1;
  if ( !a3 )
  {
    v86 = a2;
    goto LABEL_72;
  }
  v5 = a1;
  v81 = (__int64 *)(a1 + 64);
  v82 = (const __m128i *)(a1 + 88);
  while ( 2 )
  {
    p_src = &src;
    --v83;
    v6 = (__int64 *)(v84 - 64);
    v7 = v5 + ((__int64)(((unsigned __int64)&v84[-v5] >> 63) + ((__int64)&v84[-v5] >> 6)) >> 1 << 6);
    v8 = *(_BYTE **)(v7 + 8);
    v9 = (__int64)&v8[*(_QWORD *)(v7 + 16)];
    v99 = *(_QWORD *)v7;
    sub_2F07250((__int64 *)&p_src, v8, v9);
    v10 = *(_BYTE **)(v5 + 72);
    v11 = *(_QWORD *)(v5 + 80);
    v12 = _mm_loadu_si128((const __m128i *)(v7 + 40));
    v104 = *(_DWORD *)(v7 + 56);
    v13 = *(_QWORD *)(v5 + 64);
    v103 = v12;
    v89 = v13;
    v94 = &v96;
    sub_2F07250((__int64 *)&v94, v10, (__int64)&v10[v11]);
    v14 = _mm_loadu_si128((const __m128i *)(v5 + 104));
    v98 = *(_DWORD *)(v5 + 120);
    v97 = v14;
    v15 = (unsigned int)v89 < (unsigned int)v99;
    if ( (_DWORD)v89 == (_DWORD)v99 )
      v15 = HIDWORD(v89) < HIDWORD(v99);
    if ( v94 != &v96 )
      j_j___libc_free_0((unsigned __int64)v94);
    if ( p_src != &src )
      j_j___libc_free_0((unsigned __int64)p_src);
    if ( !v15 )
    {
      p_src = &src;
      v37 = (_BYTE *)*((_QWORD *)v84 - 7);
      v38 = (__int64)&v37[*((_QWORD *)v84 - 6)];
      v99 = *((_QWORD *)v84 - 8);
      sub_2F07250((__int64 *)&p_src, v37, v38);
      v39 = *(_BYTE **)(v5 + 72);
      v40 = *(_QWORD *)(v5 + 80);
      v41 = _mm_loadu_si128((const __m128i *)(v84 - 24));
      v104 = *((_DWORD *)v84 - 2);
      v92 = *(_QWORD *)(v5 + 64);
      v103 = v41;
      v94 = &v96;
      sub_2F07250((__int64 *)&v94, v39, (__int64)&v39[v40]);
      v42 = _mm_loadu_si128((const __m128i *)(v5 + 104));
      v98 = *(_DWORD *)(v5 + 120);
      v97 = v42;
      v43 = (unsigned int)v92 < (unsigned int)v99;
      if ( (_DWORD)v92 == (_DWORD)v99 )
        v43 = HIDWORD(v92) < HIDWORD(v99);
      if ( v94 != &v96 )
        j_j___libc_free_0((unsigned __int64)v94);
      if ( p_src != &src )
        j_j___libc_free_0((unsigned __int64)p_src);
      if ( v43 )
        goto LABEL_13;
      if ( (unsigned __int8)sub_2F07850((__int64 *)v7, v6) )
      {
LABEL_38:
        sub_2F237C0(v5, v6);
        goto LABEL_14;
      }
LABEL_67:
      sub_2F237C0(v5, (__int64 *)v7);
      goto LABEL_14;
    }
    if ( (unsigned __int8)sub_2F07850((__int64 *)v7, v6) )
      goto LABEL_67;
    if ( (unsigned __int8)sub_2F07850(v81, v6) )
      goto LABEL_38;
LABEL_13:
    sub_2F237C0(v5, v81);
LABEL_14:
    v16 = v82;
    v17 = v84;
    v18 = v5;
    while ( 1 )
    {
      v19 = *(_BYTE **)(v18 + 8);
      v20 = *(_QWORD *)(v18 + 16);
      p_src = &src;
      v86 = &v16[-2].m128i_i8[8];
      v99 = *(_QWORD *)v18;
      sub_2F07250((__int64 *)&p_src, v19, (__int64)&v19[v20]);
      v21 = (_BYTE *)v16[-1].m128i_i64[0];
      v22 = v16[-1].m128i_i64[1];
      v23 = _mm_loadu_si128((const __m128i *)(v18 + 40));
      v104 = *(_DWORD *)(v18 + 56);
      v24 = v16[-2].m128i_i64[1];
      v103 = v23;
      v90 = v24;
      v94 = &v96;
      sub_2F07250((__int64 *)&v94, v21, (__int64)&v21[v22]);
      v25 = _mm_loadu_si128(v16 + 1);
      v98 = v16[2].m128i_i32[0];
      v97 = v25;
      v26 = (unsigned int)v90 < (unsigned int)v99;
      if ( (_DWORD)v90 == (_DWORD)v99 )
        v26 = HIDWORD(v90) < HIDWORD(v99);
      if ( v94 != &v96 )
        j_j___libc_free_0((unsigned __int64)v94);
      if ( p_src != &src )
        j_j___libc_free_0((unsigned __int64)p_src);
      if ( !v26 )
        break;
LABEL_50:
      v16 += 4;
    }
    v27 = v18;
    dest = (__m128i *)v16;
    v28 = (unsigned __int64)(v17 - 64);
    v5 = v27;
    do
    {
      v29 = *(_QWORD *)v28;
      v30 = *(_BYTE **)(v28 + 8);
      p_src = &src;
      v87 = v28;
      v99 = v29;
      sub_2F07250((__int64 *)&p_src, v30, (__int64)&v30[*(_QWORD *)(v28 + 16)]);
      v31 = *(_DWORD *)(v28 + 56);
      v32 = *(_BYTE **)(v5 + 8);
      v94 = &v96;
      v33 = _mm_loadu_si128((const __m128i *)(v28 + 40));
      v104 = v31;
      v34 = *(_QWORD *)v5;
      v103 = v33;
      v91 = v34;
      sub_2F07250((__int64 *)&v94, v32, (__int64)&v32[*(_QWORD *)(v5 + 16)]);
      v35 = _mm_loadu_si128((const __m128i *)(v5 + 40));
      v98 = *(_DWORD *)(v5 + 56);
      v97 = v35;
      v36 = (unsigned int)v91 < (unsigned int)v99;
      if ( (_DWORD)v91 == (_DWORD)v99 )
        v36 = HIDWORD(v91) < HIDWORD(v99);
      if ( v94 != &v96 )
        j_j___libc_free_0((unsigned __int64)v94);
      if ( p_src != &src )
        j_j___libc_free_0((unsigned __int64)p_src);
      v28 -= 64LL;
    }
    while ( v36 );
    v16 = dest;
    v17 = (__int8 *)v87;
    v18 = v5;
    if ( (unsigned __int64)v86 < v87 )
    {
      v44 = dest[-2].m128i_i64[1];
      p_src = &src;
      v99 = v44;
      if ( dest == (__m128i *)dest[-1].m128i_i64[0] )
      {
        src = _mm_loadu_si128(dest);
      }
      else
      {
        p_src = (__m128i *)dest[-1].m128i_i64[0];
        src.m128i_i64[0] = dest->m128i_i64[0];
      }
      v45 = dest[-1].m128i_u64[1];
      v46 = _mm_loadu_si128(dest + 1);
      dest->m128i_i8[0] = 0;
      v47 = (__m128i *)(v87 + 24);
      dest[-1].m128i_i64[0] = (__int64)dest;
      n = v45;
      LODWORD(v45) = dest[2].m128i_i32[0];
      dest[-1].m128i_i64[1] = 0;
      v104 = v45;
      v48 = *(_QWORD *)v87;
      v103 = v46;
      dest[-2].m128i_i64[1] = v48;
      v49 = *(_QWORD *)(v87 + 8);
      if ( v49 == v87 + 24 )
      {
        v58 = *(_QWORD *)(v87 + 16);
        if ( v58 )
        {
          if ( v58 == 1 )
            dest->m128i_i8[0] = *(_BYTE *)(v87 + 24);
          else
            memcpy(dest, (const void *)(v87 + 24), v58);
          v58 = *(_QWORD *)(v87 + 16);
        }
        dest[-1].m128i_i64[1] = v58;
        dest->m128i_i8[v58] = 0;
        v50 = *(_BYTE **)(v87 + 8);
      }
      else
      {
        dest[-1].m128i_i64[0] = v49;
        dest[-1].m128i_i64[1] = *(_QWORD *)(v87 + 16);
        dest->m128i_i64[0] = *(_QWORD *)(v87 + 24);
        v50 = (_BYTE *)(v87 + 24);
        *(_QWORD *)(v87 + 8) = v47;
      }
      *(_QWORD *)(v87 + 16) = 0;
      *v50 = 0;
      dest[1] = _mm_loadu_si128((const __m128i *)(v87 + 40));
      dest[2].m128i_i32[0] = *(_DWORD *)(v87 + 56);
      v51 = *(__m128i **)(v87 + 8);
      *(_QWORD *)v87 = v99;
      if ( p_src == &src )
      {
        v57 = n;
        if ( n )
        {
          if ( n == 1 )
            v51->m128i_i8[0] = src.m128i_i8[0];
          else
            memcpy(v51, &src, n);
          v57 = n;
          v51 = *(__m128i **)(v87 + 8);
        }
        *(_QWORD *)(v87 + 16) = v57;
        v51->m128i_i8[v57] = 0;
        v51 = p_src;
        goto LABEL_48;
      }
      v52 = n;
      v53 = src.m128i_i64[0];
      if ( v51 == v47 )
      {
        *(_QWORD *)(v87 + 8) = p_src;
        *(_QWORD *)(v87 + 16) = v52;
        *(_QWORD *)(v87 + 24) = v53;
      }
      else
      {
        v54 = *(_QWORD *)(v87 + 24);
        *(_QWORD *)(v87 + 8) = p_src;
        *(_QWORD *)(v87 + 16) = v52;
        *(_QWORD *)(v87 + 24) = v53;
        if ( v51 )
        {
          p_src = v51;
          src.m128i_i64[0] = v54;
          goto LABEL_48;
        }
      }
      v51 = &src;
      p_src = &src;
LABEL_48:
      n = 0;
      v51->m128i_i8[0] = 0;
      v55 = v104;
      v56 = p_src;
      *(__m128i *)(v87 + 40) = _mm_loadu_si128(&v103);
      *(_DWORD *)(v87 + 56) = v55;
      if ( v56 != &src )
        j_j___libc_free_0((unsigned __int64)v56);
      goto LABEL_50;
    }
    v3 = (__int64)&v86[-v5];
    sub_2F239C0(v86, v84, v83);
    if ( (__int64)&v86[-v5] <= 1024 )
      return;
    if ( v83 )
    {
      v84 = v86;
      continue;
    }
    break;
  }
  v4 = v5;
LABEL_72:
  v88 = v4;
  v59 = (const __m128i *)(v4 + (((v3 >> 6) - 2) >> 1 << 6) + 24);
  v60 = ((v3 >> 6) - 2) >> 1;
  while ( 2 )
  {
    v66 = v59[-2].m128i_i64[1];
    v67 = (__m128i *)v59[-1].m128i_i64[0];
    if ( v67 == v59 )
    {
      v64 = v59[2].m128i_i32[0];
      v68 = _mm_loadu_si128(v59);
      v99 = v59[-2].m128i_i64[1];
      v62 = v59[-1].m128i_u64[1];
      v69 = _mm_loadu_si128(v59 + 1);
      v59->m128i_i8[0] = 0;
      v59[-1].m128i_i64[1] = 0;
      v98 = v64;
      p_src = &src;
      v96 = v68;
      v97 = v69;
LABEL_83:
      src = _mm_loadu_si128(&v96);
    }
    else
    {
      v61 = v59->m128i_i64[0];
      v62 = v59[-1].m128i_u64[1];
      v59[-1].m128i_i64[0] = (__int64)v59;
      v63 = _mm_loadu_si128(v59 + 1);
      v64 = v59[2].m128i_i32[0];
      v59->m128i_i8[0] = 0;
      v94 = v67;
      v96.m128i_i64[0] = v61;
      v95 = v62;
      v59[-1].m128i_i64[1] = 0;
      v98 = v64;
      v99 = v66;
      p_src = &src;
      v97 = v63;
      if ( v67 == &v96 )
        goto LABEL_83;
      p_src = v67;
      src.m128i_i64[0] = v61;
    }
    n = v62;
    v65 = _mm_loadu_si128(&v97);
    v104 = v64;
    v94 = &v96;
    v95 = 0;
    v96.m128i_i8[0] = 0;
    v103 = v65;
    sub_2F0DBF0(v88, v60, v3 >> 6, &v99);
    if ( p_src != &src )
      j_j___libc_free_0((unsigned __int64)p_src);
    if ( v60 )
    {
      --v60;
      if ( v94 != &v96 )
        j_j___libc_free_0((unsigned __int64)v94);
      v59 -= 4;
      continue;
    }
    break;
  }
  if ( v94 != &v96 )
    j_j___libc_free_0((unsigned __int64)v94);
  v70 = (const void *)(v88 + 24);
  v71 = (__m128i *)(v86 - 40);
  do
  {
    v79 = v71[-2].m128i_i64[1];
    v94 = &v96;
    v93 = v79;
    if ( (__m128i *)v71[-1].m128i_i64[0] == v71 )
    {
      v96 = _mm_loadu_si128(v71);
    }
    else
    {
      v94 = (__m128i *)v71[-1].m128i_i64[0];
      v96.m128i_i64[0] = v71->m128i_i64[0];
    }
    v72 = v71[-1].m128i_u64[1];
    v73 = _mm_loadu_si128(v71 + 1);
    v71[-1].m128i_i64[0] = (__int64)v71;
    v71[-1].m128i_i64[1] = 0;
    v95 = v72;
    LODWORD(v72) = v71[2].m128i_i32[0];
    v71->m128i_i8[0] = 0;
    v98 = v72;
    v74 = *(_QWORD *)v88;
    v97 = v73;
    v71[-2].m128i_i64[1] = v74;
    v75 = *(const void **)(v88 + 8);
    if ( v75 == v70 )
    {
      v80 = *(_QWORD *)(v88 + 16);
      if ( v80 )
      {
        if ( v80 == 1 )
          v71->m128i_i8[0] = *(_BYTE *)(v88 + 24);
        else
          memcpy(v71, v70, v80);
        v80 = *(_QWORD *)(v88 + 16);
      }
      v71[-1].m128i_i64[1] = v80;
      v71->m128i_i8[v80] = 0;
      v76 = *(_BYTE **)(v88 + 8);
    }
    else
    {
      v71[-1].m128i_i64[0] = (__int64)v75;
      v71[-1].m128i_i64[1] = *(_QWORD *)(v88 + 16);
      v71->m128i_i64[0] = *(_QWORD *)(v88 + 24);
      v76 = (_BYTE *)(v88 + 24);
      *(_QWORD *)(v88 + 8) = v70;
    }
    *(_QWORD *)(v88 + 16) = 0;
    *v76 = 0;
    v71[1] = _mm_loadu_si128((const __m128i *)(v88 + 40));
    v71[2].m128i_i32[0] = *(_DWORD *)(v88 + 56);
    v99 = v93;
    p_src = &src;
    if ( v94 == &v96 )
    {
      src = _mm_loadu_si128(&v96);
    }
    else
    {
      p_src = v94;
      src.m128i_i64[0] = v96.m128i_i64[0];
    }
    v77 = (__int64)&v71[-2].m128i_i64[1] - v88;
    v78 = _mm_loadu_si128(&v97);
    v94 = &v96;
    n = v95;
    v95 = 0;
    v96.m128i_i8[0] = 0;
    v104 = v98;
    v103 = v78;
    sub_2F0DBF0(v88, 0, v77 >> 6, &v99);
    if ( p_src != &src )
      j_j___libc_free_0((unsigned __int64)p_src);
    if ( v94 != &v96 )
      j_j___libc_free_0((unsigned __int64)v94);
    v71 -= 4;
  }
  while ( v77 > 64 );
}
