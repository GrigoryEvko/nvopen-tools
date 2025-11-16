// Function: sub_95D990
// Address: 0x95d990
//
_QWORD *__fastcall sub_95D990(_QWORD *a1, __m128i *a2, size_t a3, int *a4, __m128i *a5, char a6, char a7)
{
  _QWORD *v7; // r13
  __int64 v8; // r12
  _BYTE *v10; // rdi
  __m128i *v12; // rsi
  __int64 v13; // r15
  _BYTE *v14; // r13
  size_t v15; // r12
  char *v16; // r14
  __int64 v17; // rax
  size_t v18; // rdi
  int v19; // edx
  __m128i *v20; // rax
  __int64 v21; // rdi
  __m128i *v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rdi
  unsigned __int64 v27; // rax
  unsigned int v28; // ebx
  const __m128i *v29; // r12
  __m128i *v30; // rax
  const char *v31; // r10
  __int64 v32; // r14
  size_t v33; // rax
  const char *v34; // r10
  __m128i *v35; // r8
  size_t v36; // r11
  _QWORD *v37; // rdx
  __int64 v38; // rax
  size_t v39; // rdx
  __m128i *v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rax
  _QWORD *v43; // rdi
  __int64 v44; // rsi
  const __m128i *v45; // r12
  __m128i *v46; // rdx
  const __m128i *v47; // rcx
  int v48; // eax
  size_t v49; // rax
  size_t v50; // r12
  char *v51; // rsi
  __int64 v52; // rax
  size_t v53; // rdi
  char *v54; // r12
  __int64 v55; // r12
  const __m128i *v56; // r14
  char *v57; // r12
  size_t na; // [rsp+10h] [rbp-220h]
  size_t n; // [rsp+10h] [rbp-220h]
  __int64 v60; // [rsp+20h] [rbp-210h]
  __int64 *v62; // [rsp+40h] [rbp-1F0h]
  const char *v63; // [rsp+40h] [rbp-1F0h]
  __int64 v66; // [rsp+58h] [rbp-1D8h]
  __int64 v67; // [rsp+60h] [rbp-1D0h]
  size_t v69; // [rsp+78h] [rbp-1B8h] BYREF
  _QWORD *v70; // [rsp+80h] [rbp-1B0h] BYREF
  size_t v71; // [rsp+88h] [rbp-1A8h]
  _QWORD v72[2]; // [rsp+90h] [rbp-1A0h] BYREF
  size_t v73; // [rsp+A0h] [rbp-190h] BYREF
  __int64 v74; // [rsp+A8h] [rbp-188h]
  _BYTE v75[64]; // [rsp+B0h] [rbp-180h] BYREF
  _BYTE *v76; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v77; // [rsp+F8h] [rbp-138h]
  _BYTE dest[304]; // [rsp+100h] [rbp-130h] BYREF

  v7 = a1;
  v8 = *a4;
  if ( (int)v8 <= 0 )
  {
    *a1 = a1 + 2;
    a1[1] = 0x200000000LL;
    return v7;
  }
  v10 = dest;
  v76 = dest;
  v77 = 0x2000000000LL;
  if ( (unsigned __int64)(8 * v8) > 0x100 )
  {
    sub_C8D5F0(&v76, dest, v8, 8);
    v10 = &v76[8 * (unsigned int)v77];
  }
  v12 = a5;
  memcpy(v10, a5, 8 * v8);
  v66 = 0;
  v73 = (size_t)v75;
  LODWORD(v77) = v77 + v8;
  v74 = 0x200000000LL;
  if ( !(_DWORD)v77 )
  {
    *a4 = 0;
    *v7 = v7 + 2;
    v7[1] = 0x200000000LL;
    goto LABEL_37;
  }
  v60 = (__int64)v7;
  v13 = 0;
  do
  {
    while ( 1 )
    {
      v14 = v76;
      v15 = 0;
      v67 = v13;
      v16 = *(char **)&v76[8 * v13];
      if ( v16 )
        v15 = strlen(*(const char **)&v76[8 * v13]);
      if ( v15 < a3 )
        goto LABEL_10;
      if ( a3 )
      {
        v12 = a2;
        if ( memcmp(v16, a2, a3) )
          goto LABEL_10;
      }
      if ( v15 == a3 )
        break;
      if ( a6 )
      {
        if ( &v16[a3] )
        {
          v70 = v72;
          sub_95BA30((__int64 *)&v70, &v16[a3], (__int64)&v16[v15]);
        }
        else
        {
          LOBYTE(v72[0]) = 0;
          v70 = v72;
          v71 = 0;
        }
        v17 = (unsigned int)v74;
        v12 = (__m128i *)&v70;
        v18 = v73;
        v19 = v74;
        if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
        {
          if ( v73 > (unsigned __int64)&v70 || (unsigned __int64)&v70 >= v73 + 32LL * (unsigned int)v74 )
          {
            sub_95D880((__int64)&v73, (unsigned int)v74 + 1LL);
            v17 = (unsigned int)v74;
            v18 = v73;
            v19 = v74;
            v12 = (__m128i *)&v70;
          }
          else
          {
            v54 = (char *)&v70 - v73;
            sub_95D880((__int64)&v73, (unsigned int)v74 + 1LL);
            v18 = v73;
            v17 = (unsigned int)v74;
            v12 = (__m128i *)&v54[v73];
            v19 = v74;
          }
        }
        v20 = (__m128i *)(v18 + 32 * v17);
        if ( v20 )
        {
          v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
          v21 = v12->m128i_i64[0];
          v22 = v12 + 1;
          if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
          {
LABEL_24:
            v20->m128i_i64[0] = v21;
            v20[1].m128i_i64[0] = v12[1].m128i_i64[0];
LABEL_25:
            v20->m128i_i64[1] = v12->m128i_i64[1];
            v12->m128i_i64[0] = (__int64)v22;
            v12->m128i_i64[1] = 0;
            v12[1].m128i_i8[0] = 0;
            v19 = v74;
            goto LABEL_26;
          }
LABEL_77:
          v20[1] = _mm_loadu_si128(v12 + 1);
          goto LABEL_25;
        }
        goto LABEL_26;
      }
      if ( a7 && a7 == v16[a3] )
      {
        v49 = a3 + 1;
        if ( v15 >= a3 + 1 )
        {
          v50 = v15 - v49;
          v51 = &v16[v49];
          goto LABEL_73;
        }
        v51 = &v16[v15];
        if ( &v16[v15] )
        {
          v50 = 0;
LABEL_73:
          v70 = v72;
          sub_95BA30((__int64 *)&v70, v51, (__int64)&v51[v50]);
        }
        else
        {
          LOBYTE(v72[0]) = 0;
          v70 = v72;
          v71 = 0;
        }
        v52 = (unsigned int)v74;
        v12 = (__m128i *)&v70;
        v53 = v73;
        v19 = v74;
        if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
        {
          if ( v73 > (unsigned __int64)&v70 || (unsigned __int64)&v70 >= v73 + 32LL * (unsigned int)v74 )
          {
            sub_95D880((__int64)&v73, (unsigned int)v74 + 1LL);
            v52 = (unsigned int)v74;
            v53 = v73;
            v19 = v74;
            v12 = (__m128i *)&v70;
          }
          else
          {
            v57 = (char *)&v70 - v73;
            sub_95D880((__int64)&v73, (unsigned int)v74 + 1LL);
            v53 = v73;
            v52 = (unsigned int)v74;
            v12 = (__m128i *)&v57[v73];
            v19 = v74;
          }
        }
        v20 = (__m128i *)(v53 + 32 * v52);
        if ( v20 )
        {
          v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
          v21 = v12->m128i_i64[0];
          v22 = v12 + 1;
          if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
            goto LABEL_24;
          goto LABEL_77;
        }
LABEL_26:
        LODWORD(v74) = v19 + 1;
        if ( v70 != v72 )
        {
          v12 = (__m128i *)(v72[0] + 1LL);
          j_j___libc_free_0(v70, v72[0] + 1LL);
        }
        v23 = &a5->m128i_i64[v67];
        v24 = v13;
        v25 = a5->m128i_i64[v67];
        if ( !v25 )
          goto LABEL_11;
LABEL_29:
        v62 = v23;
        j_j___libc_free_0_0(v25);
        v23 = v62;
        goto LABEL_30;
      }
LABEL_10:
      a5->m128i_i64[v66++] = (__int64)v16;
LABEL_11:
      if ( ++v13 >= (unsigned __int64)(unsigned int)v77 )
        goto LABEL_34;
    }
    v24 = v13 + 1;
    v31 = *(const char **)&v14[v67 * 8 + 8];
    v32 = v67 * 8 + 8;
    v70 = v72;
    if ( !v31 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v63 = v31;
    v33 = strlen(v31);
    v34 = v63;
    v69 = v33;
    v35 = (__m128i *)&v70;
    v36 = v33;
    if ( v33 > 0xF )
    {
      na = v33;
      v42 = sub_22409D0(&v70, &v69, 0);
      v34 = v63;
      v70 = (_QWORD *)v42;
      v43 = (_QWORD *)v42;
      v36 = na;
      v72[0] = v69;
    }
    else
    {
      if ( v33 == 1 )
      {
        LOBYTE(v72[0]) = *v63;
        v37 = v72;
        goto LABEL_43;
      }
      if ( !v33 )
      {
        v37 = v72;
        goto LABEL_43;
      }
      v43 = v72;
    }
    memcpy(v43, v34, v36);
    v33 = v69;
    v37 = v70;
    v35 = (__m128i *)&v70;
LABEL_43:
    v71 = v33;
    *((_BYTE *)v37 + v33) = 0;
    v38 = (unsigned int)v74;
    v12 = (__m128i *)(unsigned int)v74;
    if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
    {
      if ( v73 > (unsigned __int64)&v70 || (n = v73, (unsigned __int64)&v70 >= v73 + 32LL * (unsigned int)v74) )
      {
        sub_95D880((__int64)&v73, (unsigned int)v74 + 1LL);
        v38 = (unsigned int)v74;
        v39 = v73;
        v35 = (__m128i *)&v70;
        v12 = (__m128i *)(unsigned int)v74;
      }
      else
      {
        sub_95D880((__int64)&v73, (unsigned int)v74 + 1LL);
        v39 = v73;
        v38 = (unsigned int)v74;
        v35 = (__m128i *)((char *)&v70 + v73 - n);
        v12 = (__m128i *)(unsigned int)v74;
      }
    }
    else
    {
      v39 = v73;
    }
    v40 = (__m128i *)(v39 + 32 * v38);
    if ( v40 )
    {
      v40->m128i_i64[0] = (__int64)v40[1].m128i_i64;
      if ( (__m128i *)v35->m128i_i64[0] == &v35[1] )
      {
        v40[1] = _mm_loadu_si128(v35 + 1);
      }
      else
      {
        v40->m128i_i64[0] = v35->m128i_i64[0];
        v40[1].m128i_i64[0] = v35[1].m128i_i64[0];
      }
      v41 = v35->m128i_i64[1];
      v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
      v35->m128i_i64[1] = 0;
      v40->m128i_i64[1] = v41;
      v12 = (__m128i *)(unsigned int)v74;
      v35[1].m128i_i8[0] = 0;
    }
    LODWORD(v74) = (_DWORD)v12 + 1;
    if ( v70 != v72 )
    {
      v12 = (__m128i *)(v72[0] + 1LL);
      j_j___libc_free_0(v70, v72[0] + 1LL);
    }
    v23 = (__int64 *)((char *)a5->m128i_i64 + v32);
    v25 = *(__int64 *)((char *)a5->m128i_i64 + v32);
    if ( v25 )
      goto LABEL_29;
LABEL_30:
    *v23 = 0;
    if ( v13 == v24 )
      goto LABEL_11;
    v26 = a5->m128i_i64[v13];
    if ( v26 )
      j_j___libc_free_0_0(v26);
    v27 = (unsigned int)v77;
    a5->m128i_i64[v13] = 0;
    v13 = v24 + 1;
  }
  while ( v24 + 1 < v27 );
LABEL_34:
  v7 = (_QWORD *)v60;
  v28 = v74;
  v29 = (const __m128i *)v73;
  *a4 = v66;
  v30 = (__m128i *)(v60 + 16);
  *(_QWORD *)v60 = v60 + 16;
  *(_QWORD *)(v60 + 8) = 0x200000000LL;
  if ( !v28 )
    goto LABEL_35;
  if ( v29 == (const __m128i *)v75 )
  {
    v44 = v28;
    if ( v28 > 2 )
    {
      sub_95D880(v60, v28);
      v30 = *(__m128i **)v60;
      v29 = (const __m128i *)v73;
      v44 = (unsigned int)v74;
    }
    v12 = (__m128i *)(32 * v44);
    if ( v12 )
    {
      v45 = v29 + 1;
      v46 = (__m128i *)((char *)v12 + (_QWORD)v30);
      do
      {
        if ( v30 )
        {
          v30->m128i_i64[0] = (__int64)v30[1].m128i_i64;
          v47 = (const __m128i *)v45[-1].m128i_i64[0];
          if ( v47 == v45 )
          {
            v30[1] = _mm_loadu_si128(v45);
          }
          else
          {
            v30->m128i_i64[0] = (__int64)v47;
            v30[1].m128i_i64[0] = v45->m128i_i64[0];
          }
          v30->m128i_i64[1] = v45[-1].m128i_i64[1];
          v45[-1].m128i_i64[0] = (__int64)v45;
          v45[-1].m128i_i64[1] = 0;
          v45->m128i_i8[0] = 0;
        }
        v30 += 2;
        v45 += 2;
      }
      while ( v30 != v46 );
      v55 = (unsigned int)v74;
      v56 = (const __m128i *)v73;
      *(_DWORD *)(v60 + 8) = v28;
      v29 = &v56[2 * v55];
      if ( v56 != v29 )
      {
        do
        {
          v29 -= 2;
          if ( (const __m128i *)v29->m128i_i64[0] != &v29[1] )
          {
            v12 = (__m128i *)(v29[1].m128i_i64[0] + 1);
            j_j___libc_free_0(v29->m128i_i64[0], v12);
          }
        }
        while ( v56 != v29 );
        v29 = (const __m128i *)v73;
      }
    }
    else
    {
      *(_DWORD *)(v60 + 8) = v28;
    }
LABEL_35:
    if ( v29 != (const __m128i *)v75 )
      _libc_free(v29, v12);
  }
  else
  {
    v48 = HIDWORD(v74);
    *(_QWORD *)v60 = v29;
    *(_DWORD *)(v60 + 8) = v28;
    *(_DWORD *)(v60 + 12) = v48;
  }
LABEL_37:
  if ( v76 != dest )
    _libc_free(v76, v12);
  return v7;
}
