// Function: sub_1E81750
// Address: 0x1e81750
//
void __fastcall sub_1E81750(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  const __m128i *v8; // rsi
  char v9; // al
  const __m128i *v10; // rdi
  const __m128i *v11; // rdx
  __m128i *v12; // rax
  const __m128i *v13; // rcx
  __m128i *v14; // r12
  const __m128i *v15; // rcx
  __int64 v16; // rax
  __m128i *v17; // rdi
  const __m128i *v18; // rax
  const __m128i *v19; // rsi
  __int64 v20; // r15
  _QWORD *v21; // r14
  __int64 v22; // r14
  __int64 *v23; // rax
  unsigned int v24; // eax
  char v25; // al
  __m128i *v26; // rax
  const __m128i *v27; // rdx
  __m128i *v28; // r12
  unsigned __int64 v29; // r15
  __m128i *v30; // rbx
  __m128i *v31; // rdi
  const __m128i *v32; // rax
  const __m128i *v33; // rsi
  __int64 v34; // r15
  __int64 v35; // r14
  __int64 v36; // rdx
  __int64 v37; // rcx
  int v38; // r8d
  __int64 v39; // r15
  __int64 *v40; // rax
  __int64 v41; // rax
  __m128i *v42; // rsi
  __int64 v43; // rax
  __m128i *v44; // rsi
  __int64 v45; // rax
  const __m128i *v46; // rax
  __int64 *v47; // rax
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rax
  const __m128i *v51; // rax
  __int64 *v52; // rax
  __int64 v53; // r15
  __int64 v54; // rax
  const __m128i *v55; // [rsp+0h] [rbp-230h]
  const __m128i *v56; // [rsp+8h] [rbp-228h]
  const __m128i *v57; // [rsp+8h] [rbp-228h]
  signed __int64 v58; // [rsp+18h] [rbp-218h]
  const __m128i *v60; // [rsp+20h] [rbp-210h]
  __m128i *v61; // [rsp+28h] [rbp-208h]
  signed __int64 v62; // [rsp+28h] [rbp-208h]
  __int64 v63; // [rsp+30h] [rbp-200h] BYREF
  __m128i *v64; // [rsp+38h] [rbp-1F8h] BYREF
  __m128i *v65; // [rsp+40h] [rbp-1F0h]
  const __m128i *v66; // [rsp+48h] [rbp-1E8h]
  __int64 v67; // [rsp+50h] [rbp-1E0h] BYREF
  __m128i *v68; // [rsp+58h] [rbp-1D8h] BYREF
  __m128i *v69; // [rsp+60h] [rbp-1D0h]
  const __m128i *v70; // [rsp+68h] [rbp-1C8h]
  __int64 v71; // [rsp+70h] [rbp-1C0h]
  _QWORD *v72; // [rsp+90h] [rbp-1A0h] BYREF
  const __m128i *v73; // [rsp+98h] [rbp-198h] BYREF
  __m128i *v74; // [rsp+A0h] [rbp-190h]
  const __m128i *v75; // [rsp+A8h] [rbp-188h]
  __int64 v76; // [rsp+B0h] [rbp-180h]
  _QWORD *v77; // [rsp+D0h] [rbp-160h] BYREF
  const __m128i *v78; // [rsp+D8h] [rbp-158h] BYREF
  __m128i *v79; // [rsp+E0h] [rbp-150h]
  const __m128i *v80; // [rsp+E8h] [rbp-148h]
  __m128i v81; // [rsp+F0h] [rbp-140h] BYREF
  __m128i v82; // [rsp+130h] [rbp-100h] BYREF
  _QWORD v83[2]; // [rsp+170h] [rbp-C0h] BYREF
  __int64 v84; // [rsp+180h] [rbp-B0h] BYREF
  _BYTE *v85; // [rsp+188h] [rbp-A8h]
  void *s; // [rsp+190h] [rbp-A0h]
  _BYTE v87[12]; // [rsp+198h] [rbp-98h]
  _BYTE v88[64]; // [rsp+1A8h] [rbp-88h] BYREF
  __int64 v89; // [rsp+1E8h] [rbp-48h]
  char v90; // [rsp+1F0h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 440);
  v90 = 0;
  v4 = *(_QWORD *)(v3 + 264);
  v5 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)v87 = 8;
  *(_DWORD *)&v87[8] = 0;
  v83[0] = v5;
  v6 = *(unsigned int *)(a1 + 16);
  v89 = v4;
  v8 = &v82;
  v83[1] = v6;
  v85 = v88;
  s = v88;
  v84 = 1;
  v72 = v83;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v82.m128i_i8[8] = 0;
  v9 = sub_1E7F580((__int64 *)&v72, (__int64)&v82, a2);
  v10 = v74;
  if ( v9 )
  {
    v45 = *(_QWORD *)(a2 + 64);
    v82.m128i_i64[0] = a2;
    v82.m128i_i64[1] = v45;
    v46 = v74;
    if ( v74 == v75 )
    {
      v8 = v74;
      sub_1DE02F0(&v73, v74, &v82);
      v10 = v74;
    }
    else
    {
      if ( v74 )
      {
        *v74 = _mm_loadu_si128(&v82);
        v46 = v74;
      }
      v10 = v46 + 1;
      v74 = (__m128i *)&v46[1];
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v47 = (__int64 *)v10[-1].m128i_i64[1];
        if ( *(__int64 **)(v10[-1].m128i_i64[0] + 72) == v47 )
          goto LABEL_2;
        v8 = &v82;
        v10[-1].m128i_i64[1] = (__int64)(v47 + 1);
        v48 = *v47;
        v82.m128i_i8[8] = 1;
        v82.m128i_i64[0] = v74[-1].m128i_i64[0];
        if ( (unsigned __int8)sub_1E7F580((__int64 *)&v72, (__int64)&v82, v48) )
          break;
LABEL_93:
        v10 = v74;
      }
      v49 = *(_QWORD *)(v48 + 64);
      v8 = v74;
      v82.m128i_i64[0] = v48;
      v82.m128i_i64[1] = v49;
      if ( v74 == v75 )
      {
        sub_1DE02F0(&v73, v74, &v82);
        goto LABEL_93;
      }
      if ( v74 )
      {
        *v74 = _mm_loadu_si128(&v82);
        v8 = v74;
      }
      v10 = v8 + 1;
      v74 = (__m128i *)&v8[1];
    }
  }
LABEL_2:
  v11 = (const __m128i *)((char *)v10 - (char *)v73);
  v71 = (__int64)v72;
  v58 = (char *)v10 - (char *)v73;
  if ( v10 == v73 )
  {
    v61 = 0;
LABEL_83:
    v14 = v61;
    v11 = 0;
    v15 = 0;
    goto LABEL_10;
  }
  if ( (unsigned __int64)v11 > 0x7FFFFFFFFFFFFFF0LL )
    goto LABEL_119;
  v12 = (__m128i *)sub_22077B0(v58);
  v10 = v74;
  v13 = v73;
  v61 = v12;
  if ( v73 == v74 )
    goto LABEL_83;
  v11 = (const __m128i *)((char *)v74 - (char *)v73);
  v14 = (__m128i *)((char *)v12 + (char *)v74 - (char *)v73);
  do
  {
    if ( v12 )
      *v12 = _mm_loadu_si128(v13);
    ++v12;
    ++v13;
  }
  while ( v12 != v14 );
  v10 = v73;
  v15 = v11;
LABEL_10:
  if ( v10 )
  {
    v55 = v15;
    v56 = v11;
    v8 = (const __m128i *)((char *)v75 - (char *)v10);
    j_j___libc_free_0(v10, (char *)v75 - (char *)v10);
    v15 = v55;
    v11 = v56;
  }
  v64 = 0;
  v65 = 0;
  v82.m128i_i64[0] = v71;
  v81.m128i_i64[0] = v71;
  v63 = v71;
  v66 = 0;
  if ( v15 )
  {
    if ( (unsigned __int64)v11 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_119;
    v57 = v11;
    v16 = sub_22077B0(v11);
    v11 = v57;
    v17 = (__m128i *)v16;
  }
  else
  {
    v17 = 0;
  }
  v64 = v17;
  v65 = v17;
  v66 = (const __m128i *)((char *)v11 + (_QWORD)v17);
  v18 = v61;
  if ( v14 == v61 )
  {
    v19 = v17;
  }
  else
  {
    v19 = (__m128i *)((char *)v17 + (char *)v14 - (char *)v61);
    do
    {
      if ( v17 )
        *v17 = _mm_loadu_si128(v18);
      ++v17;
      ++v18;
    }
    while ( v17 != v19 );
    v17 = v64;
  }
  v65 = (__m128i *)v19;
  if ( v19 != v17 )
  {
LABEL_22:
    v20 = v19[-1].m128i_i64[0];
    v21 = (_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL * *(int *)(v20 + 48));
    *v21 = (**(__int64 (__fastcall ***)(__int64, __int64))a1)(a1, v20);
    sub_1E80550((_QWORD *)a1, v20);
    --v65;
    v17 = v64;
    v19 = v65;
    if ( v65 == v64 )
      goto LABEL_28;
    while ( 1 )
    {
      while ( 1 )
      {
        v23 = (__int64 *)v19[-1].m128i_i64[1];
        if ( *(__int64 **)(v19[-1].m128i_i64[0] + 72) == v23 )
        {
          v17 = v64;
          if ( v19 == v64 )
            goto LABEL_28;
          goto LABEL_22;
        }
        v19[-1].m128i_i64[1] = (__int64)(v23 + 1);
        v22 = *v23;
        v82.m128i_i8[8] = 1;
        v82.m128i_i64[0] = v65[-1].m128i_i64[0];
        if ( (unsigned __int8)sub_1E7F580(&v63, (__int64)&v82, v22) )
          break;
LABEL_25:
        v19 = v65;
      }
      v41 = *(_QWORD *)(v22 + 64);
      v42 = v65;
      v82.m128i_i64[0] = v22;
      v82.m128i_i64[1] = v41;
      if ( v65 == v66 )
      {
        sub_1DE02F0((const __m128i **)&v64, v65, &v82);
        goto LABEL_25;
      }
      if ( v65 )
      {
        *v65 = _mm_loadu_si128(&v82);
        v42 = v65;
      }
      v19 = v42 + 1;
      v65 = (__m128i *)v19;
    }
  }
LABEL_28:
  if ( v17 )
    j_j___libc_free_0(v17, (char *)v66 - (char *)v17);
  if ( v61 )
    j_j___libc_free_0(v61, v58);
  ++v84;
  v90 = 1;
  if ( s == v85 )
    goto LABEL_37;
  v24 = 4 * (*(_DWORD *)&v87[4] - *(_DWORD *)&v87[8]);
  if ( v24 < 0x20 )
    v24 = 32;
  if ( v24 >= *(_DWORD *)v87 )
  {
    memset(s, -1, 8LL * *(unsigned int *)v87);
LABEL_37:
    *(_QWORD *)&v87[4] = 0;
    goto LABEL_38;
  }
  sub_16CC920((__int64)&v84);
LABEL_38:
  v8 = &v82;
  v78 = 0;
  v77 = v83;
  v79 = 0;
  v80 = 0;
  v82.m128i_i8[8] = 0;
  v25 = sub_1E7F580((__int64 *)&v77, (__int64)&v82, a2);
  v10 = v79;
  if ( v25 )
  {
    v50 = *(_QWORD *)(a2 + 88);
    v82.m128i_i64[0] = a2;
    v82.m128i_i64[1] = v50;
    v51 = v79;
    if ( v79 == v80 )
    {
      v8 = v79;
      sub_1DE02F0(&v78, v79, &v82);
      v10 = v79;
    }
    else
    {
      if ( v79 )
      {
        *v79 = _mm_loadu_si128(&v82);
        v51 = v79;
      }
      v10 = v51 + 1;
      v79 = (__m128i *)&v51[1];
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v52 = (__int64 *)v10[-1].m128i_i64[1];
        if ( *(__int64 **)(v10[-1].m128i_i64[0] + 96) == v52 )
          goto LABEL_39;
        v8 = &v82;
        v10[-1].m128i_i64[1] = (__int64)(v52 + 1);
        v53 = *v52;
        v82.m128i_i8[8] = 1;
        v82.m128i_i64[0] = v79[-1].m128i_i64[0];
        if ( (unsigned __int8)sub_1E7F580((__int64 *)&v77, (__int64)&v82, v53) )
          break;
LABEL_105:
        v10 = v79;
      }
      v54 = *(_QWORD *)(v53 + 88);
      v8 = v79;
      v82.m128i_i64[0] = v53;
      v82.m128i_i64[1] = v54;
      if ( v79 == v80 )
      {
        sub_1DE02F0(&v78, v79, &v82);
        goto LABEL_105;
      }
      if ( v79 )
      {
        *v79 = _mm_loadu_si128(&v82);
        v8 = v79;
      }
      v10 = v8 + 1;
      v79 = (__m128i *)&v8[1];
    }
  }
LABEL_39:
  v11 = v78;
  v76 = (__int64)v77;
  v62 = (char *)v10 - (char *)v78;
  if ( v10 == v78 )
  {
    v28 = 0;
LABEL_86:
    v30 = v28;
    v11 = 0;
    v29 = 0;
    goto LABEL_47;
  }
  if ( (unsigned __int64)((char *)v10 - (char *)v78) > 0x7FFFFFFFFFFFFFF0LL )
    goto LABEL_119;
  v26 = (__m128i *)sub_22077B0((char *)v10 - (char *)v78);
  v10 = v79;
  v27 = v78;
  v28 = v26;
  if ( v78 == v79 )
    goto LABEL_86;
  v29 = (char *)v79 - (char *)v78;
  v30 = (__m128i *)((char *)v26 + (char *)v79 - (char *)v78);
  do
  {
    if ( v26 )
      *v26 = _mm_loadu_si128(v27);
    ++v26;
    ++v27;
  }
  while ( v26 != v30 );
  v10 = v78;
  v11 = (const __m128i *)v29;
LABEL_47:
  if ( v10 )
  {
    v60 = v11;
    v8 = (const __m128i *)((char *)v80 - (char *)v10);
    j_j___libc_free_0(v10, (char *)v80 - (char *)v10);
    v11 = v60;
  }
  v68 = 0;
  v69 = 0;
  v81.m128i_i64[0] = v76;
  v82.m128i_i64[0] = v76;
  v67 = v76;
  v70 = 0;
  if ( v11 )
  {
    if ( v29 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v31 = (__m128i *)sub_22077B0(v29);
      goto LABEL_52;
    }
LABEL_119:
    sub_4261EA(v10, v8, v11);
  }
  v31 = 0;
LABEL_52:
  v68 = v31;
  v69 = v31;
  v70 = (__m128i *)((char *)v31 + v29);
  if ( v28 == v30 )
  {
    v33 = v31;
  }
  else
  {
    v32 = v28;
    v33 = (__m128i *)((char *)v31 + (char *)v30 - (char *)v28);
    do
    {
      if ( v31 )
        *v31 = _mm_loadu_si128(v32);
      ++v31;
      ++v32;
    }
    while ( v31 != v33 );
    v31 = v68;
  }
  v69 = (__m128i *)v33;
  if ( v33 != v31 )
  {
LABEL_59:
    v34 = v33[-1].m128i_i64[0];
    v35 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(v34 + 48);
    *(_QWORD *)(v35 + 8) = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 8LL))(a1, v34);
    sub_1E806A0((_QWORD *)a1, v34, v36, v37, v38);
    --v69;
    v31 = v68;
    v33 = v69;
    if ( v69 == v68 )
      goto LABEL_65;
    while ( 1 )
    {
      while ( 1 )
      {
        v40 = (__int64 *)v33[-1].m128i_i64[1];
        if ( *(__int64 **)(v33[-1].m128i_i64[0] + 96) == v40 )
        {
          v31 = v68;
          if ( v33 == v68 )
            goto LABEL_65;
          goto LABEL_59;
        }
        v33[-1].m128i_i64[1] = (__int64)(v40 + 1);
        v39 = *v40;
        v81.m128i_i8[8] = 1;
        v81.m128i_i64[0] = v69[-1].m128i_i64[0];
        if ( (unsigned __int8)sub_1E7F580(&v67, (__int64)&v81, v39) )
          break;
LABEL_62:
        v33 = v69;
      }
      v43 = *(_QWORD *)(v39 + 88);
      v44 = v69;
      v81.m128i_i64[0] = v39;
      v81.m128i_i64[1] = v43;
      if ( v69 == v70 )
      {
        sub_1DE02F0((const __m128i **)&v68, v69, &v81);
        goto LABEL_62;
      }
      if ( v69 )
      {
        *v69 = _mm_loadu_si128(&v81);
        v44 = v69;
      }
      v33 = v44 + 1;
      v69 = (__m128i *)v33;
    }
  }
LABEL_65:
  if ( v31 )
    j_j___libc_free_0(v31, (char *)v70 - (char *)v31);
  if ( v28 )
    j_j___libc_free_0(v28, v62);
  if ( s != v85 )
    _libc_free((unsigned __int64)s);
}
