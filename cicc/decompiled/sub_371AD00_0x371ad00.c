// Function: sub_371AD00
// Address: 0x371ad00
//
unsigned __int64 __fastcall sub_371AD00(unsigned __int64 *base, unsigned __int64 a2)
{
  unsigned __int64 v2; // r8
  unsigned __int64 v3; // rax
  char *v4; // r15
  unsigned __int64 *v5; // rbx
  unsigned __int8 v6; // r13
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int8 v9; // cl
  unsigned __int64 *v10; // rdx
  void **v11; // r9
  unsigned __int64 *v12; // rdi
  __int64 v13; // rsi
  unsigned __int64 v14; // r10
  unsigned __int64 **v15; // r11
  char *v16; // r12
  char v17; // cl
  void *v18; // rax
  void *v19; // rdx
  unsigned __int64 v20; // rdx
  const __m128i *p_src; // r14
  __m128i *v22; // rax
  unsigned __int64 *v23; // rdx
  unsigned __int64 v24; // r12
  unsigned __int64 *v26; // r12
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  const __m128i *v29; // rax
  char *v30; // rcx
  unsigned __int64 v31; // r8
  __m128i *v32; // rdx
  _BYTE *v33; // rbx
  char *v34; // r14
  char *v35; // rdx
  unsigned __int64 v36; // [rsp+8h] [rbp-438h]
  unsigned __int64 **v37; // [rsp+10h] [rbp-430h]
  unsigned __int64 **v38; // [rsp+10h] [rbp-430h]
  signed __int64 v39; // [rsp+18h] [rbp-428h]
  unsigned __int64 **v40; // [rsp+18h] [rbp-428h]
  __int64 n; // [rsp+30h] [rbp-410h]
  unsigned __int64 **v43; // [rsp+48h] [rbp-3F8h] BYREF
  unsigned __int64 v44; // [rsp+50h] [rbp-3F0h] BYREF
  _QWORD v45[3]; // [rsp+58h] [rbp-3E8h] BYREF
  _QWORD v46[4]; // [rsp+70h] [rbp-3D0h] BYREF
  unsigned __int64 **v47[4]; // [rsp+90h] [rbp-3B0h] BYREF
  unsigned __int64 v48; // [rsp+B0h] [rbp-390h] BYREF
  __int64 v49; // [rsp+B8h] [rbp-388h]
  _BYTE v50[192]; // [rsp+C0h] [rbp-380h] BYREF
  void *src; // [rsp+180h] [rbp-2C0h] BYREF
  __int64 v52; // [rsp+188h] [rbp-2B8h]
  _BYTE v53[688]; // [rsp+190h] [rbp-2B0h] BYREF

  v2 = a2;
  v3 = a2;
  v4 = (char *)&base[5 * a2];
  n = 5 * a2;
  if ( base == (unsigned __int64 *)v4 )
  {
LABEL_49:
    if ( a2 )
      return base[n - 4] + base[n - 5];
    return v3;
  }
  v5 = base;
  v6 = 0;
  while ( *v5 != -1 )
  {
    if ( v6 < *((_BYTE *)v5 + 32) )
      v6 = *((_BYTE *)v5 + 32);
    v5 += 5;
    if ( v4 == (char *)v5 )
      goto LABEL_49;
  }
  v7 = v5;
  v8 = 0;
  do
  {
    v9 = *((_BYTE *)v7 + 32);
    v7[3] = v8;
    if ( v6 < v9 )
      v6 = v9;
    v7 += 5;
    ++v8;
  }
  while ( v7 != (unsigned __int64 *)v4 );
  if ( v4 - (char *)v5 > 40 )
  {
    qsort(v5, 0xCCCCCCCCCCCCCCCDLL * ((v4 - (char *)v5) >> 3), 0x28u, (__compar_fn_t)sub_371AA20);
    v2 = a2;
  }
  v10 = base;
  v3 = 0;
  if ( base == v5 )
  {
LABEL_29:
    v23 = v5;
    while ( ((v3 + (1LL << *((_BYTE *)v23 + 32)) - 1) & -(1LL << *((_BYTE *)v23 + 32))) == v3 )
    {
      *v23 = v3;
      v3 += v23[1];
      v23 += 5;
      if ( v23 == (unsigned __int64 *)v4 )
        return v3;
    }
  }
  else
  {
    while ( *v10 == v3 )
    {
      v3 += v10[1];
      v10 += 5;
      if ( v10 == v5 )
        goto LABEL_29;
    }
  }
  v11 = (void **)v50;
  v12 = v5;
  v13 = 0;
  v49 = 0x800000000LL;
  v14 = 8;
  v15 = (unsigned __int64 **)&v48;
  v48 = (unsigned __int64)v50;
  while ( 1 )
  {
    v16 = (char *)(v12 + 5);
    v17 = *((_BYTE *)v12 + 32);
    v18 = (void *)v12[1];
    if ( v12 + 5 == (unsigned __int64 *)v4 )
    {
      v35 = (char *)v12;
    }
    else
    {
      while ( 1 )
      {
        v35 = v16 - 40;
        if ( v16[32] != v17 )
          break;
        v19 = (void *)*((_QWORD *)v16 + 1);
        *((_QWORD *)v16 - 2) = v16;
        if ( v18 > v19 )
          v18 = v19;
        if ( v16 + 40 == v4 )
        {
          v35 = v16;
          v16 = v4;
          break;
        }
        v16 += 40;
      }
    }
    *((_QWORD *)v35 + 3) = 0;
    v20 = v13 + 1;
    p_src = (const __m128i *)&src;
    src = v18;
    v52 = (__int64)v12;
    v53[0] = v17;
    if ( v13 + 1 > v14 )
    {
      if ( v11 > &src )
      {
        v36 = v2;
      }
      else
      {
        v36 = v2;
        if ( &src < &v11[3 * v13] )
        {
          v34 = (char *)((char *)&src - (char *)v11);
          v37 = v15;
          sub_C8D5F0((__int64)v15, v50, v20, 0x18u, v2, (__int64)v11);
          v11 = (void **)v48;
          v13 = (unsigned int)v49;
          v2 = v36;
          v15 = v37;
          p_src = (const __m128i *)&v34[v48];
          goto LABEL_26;
        }
      }
      v38 = v15;
      sub_C8D5F0((__int64)v15, v50, v20, 0x18u, v2, (__int64)v11);
      v11 = (void **)v48;
      v13 = (unsigned int)v49;
      v15 = v38;
      v2 = v36;
    }
LABEL_26:
    v22 = (__m128i *)&v11[3 * v13];
    *v22 = _mm_loadu_si128(p_src);
    v22[1].m128i_i64[0] = p_src[1].m128i_i64[0];
    v13 = (unsigned int)(v49 + 1);
    LODWORD(v49) = v49 + 1;
    if ( v4 == v16 )
      break;
    v14 = HIDWORD(v49);
    v11 = (void **)v48;
    v12 = (unsigned __int64 *)v16;
  }
  v43 = v15;
  src = v53;
  v52 = 0x1000000000LL;
  if ( v2 > 0x10 )
  {
    v40 = v15;
    sub_C8D5F0((__int64)&src, v53, v2, 0x28u, v2, (__int64)v11);
    v15 = v40;
  }
  v44 = 0;
  v46[0] = &v43;
  v45[0] = v46;
  v26 = base;
  v46[1] = &src;
  v46[2] = &v44;
  v47[1] = (unsigned __int64 **)&v44;
  v47[2] = (unsigned __int64 **)v45;
  v27 = 0;
  for ( v47[0] = v15; v26 != v5; v44 = v27 )
  {
    while ( *v26 != v27 && (unsigned __int8)sub_371AA60(v47, *v26, 1) )
      v27 = v44;
    v28 = (unsigned int)v52;
    v29 = (const __m128i *)v26;
    v30 = (char *)src;
    v31 = (unsigned int)v52 + 1LL;
    if ( v31 > HIDWORD(v52) )
    {
      if ( src > v26 || (char *)src + 40 * (unsigned int)v52 <= (char *)v26 )
      {
        sub_C8D5F0((__int64)&src, v53, (unsigned int)v52 + 1LL, 0x28u, v31, (__int64)v11);
        v30 = (char *)src;
        v28 = (unsigned int)v52;
        v29 = (const __m128i *)v26;
      }
      else
      {
        v39 = (char *)v26 - (_BYTE *)src;
        sub_C8D5F0((__int64)&src, v53, (unsigned int)v52 + 1LL, 0x28u, v31, (__int64)v11);
        v30 = (char *)src;
        v28 = (unsigned int)v52;
        v29 = (const __m128i *)((char *)src + v39);
      }
    }
    v26 += 5;
    v32 = (__m128i *)&v30[40 * v28];
    *v32 = _mm_loadu_si128(v29);
    v32[1] = _mm_loadu_si128(v29 + 1);
    v32[2].m128i_i64[0] = v29[2].m128i_i64[0];
    v27 = *(v26 - 5) + *(v26 - 4);
    LODWORD(v52) = v52 + 1;
  }
  while ( (_DWORD)v49 )
    sub_371AA60(v47, v45[1], 0);
  v33 = src;
  memcpy(base, src, n * 8);
  v24 = v44;
  if ( v33 != v53 )
    _libc_free((unsigned __int64)v33);
  if ( (_BYTE *)v48 != v50 )
    _libc_free(v48);
  return v24;
}
