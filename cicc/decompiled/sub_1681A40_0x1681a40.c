// Function: sub_1681A40
// Address: 0x1681a40
//
__m128i *__fastcall sub_1681A40(
        __m128i *a1,
        void **a2,
        _DWORD *a3,
        size_t a4,
        char *a5,
        __int64 a6,
        const char **a7,
        __int64 a8)
{
  char *v12; // r14
  char *v13; // rsi
  size_t v14; // rdx
  const __m128i *v17; // rax
  __int64 v18; // rdx
  const __m128i *v19; // r11
  const char **v20; // r14
  const char **v21; // r9
  __m128i *v22; // r10
  __int64 v23; // rdx
  const char *v24; // rax
  const char *v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rax
  const char *v28; // r10
  __int64 v29; // r8
  _BYTE *v30; // rax
  void *v31; // rdi
  _QWORD *v32; // r8
  __m128i *v33; // r14
  __m128i si128; // xmm0
  __int64 v35; // rdx
  __m128i v36; // xmm0
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  char *src; // [rsp+18h] [rbp-78h]
  _QWORD *srcb; // [rsp+18h] [rbp-78h]
  __m128i v45; // [rsp+20h] [rbp-70h] BYREF
  __int64 v46; // [rsp+30h] [rbp-60h]
  __m128i v47; // [rsp+40h] [rbp-50h]
  __int64 v48; // [rsp+50h] [rbp-40h]

  if ( !a6 || !a8 )
  {
    a1->m128i_i64[0] = 0;
    a1->m128i_i64[1] = 0;
    a1[1].m128i_i64[0] = 0;
    return a1;
  }
  v45 = 0u;
  v46 = 0;
  if ( a4 == 4 )
  {
    if ( *a3 == 1886152040 )
    {
      sub_1680D70(a5, a6, a7, a8);
      goto LABEL_7;
    }
LABEL_13:
    v17 = (const __m128i *)sub_1680B50(a3, a4, (const char **)a5, a6);
    v19 = v17;
    if ( v17 )
    {
      v20 = &a7[8 * a8];
      v46 = v17[2].m128i_i64[0];
      v45 = _mm_loadu_si128(v17 + 1);
      if ( a7 != v20 )
      {
        v21 = a7;
        v22 = &v45;
        do
        {
          v23 = v19[2].m128i_i64[0];
          v24 = v21[3];
          v25 = v21[2];
          v47 = _mm_loadu_si128(v19 + 1);
          v48 = v23;
          if ( (unsigned __int64)v21[4] & v23
             | v47.m128i_i64[0] & (unsigned __int64)v25
             | v47.m128i_i64[1] & (unsigned __int64)v24 )
          {
            sub_16809A0(v22, (__int64)v21, a7, a8);
          }
          v21 += 8;
        }
        while ( v21 != v20 );
      }
    }
    else
    {
      v27 = sub_16E8CB0(a3, a4, v18);
      v28 = (const char *)a3;
      v29 = v27;
      v30 = *(_BYTE **)(v27 + 24);
      if ( *(_BYTE **)(v29 + 16) == v30 )
      {
        v37 = sub_16E7EE0(v29, "'", 1);
        v28 = (const char *)a3;
        v31 = *(void **)(v37 + 24);
        v29 = v37;
      }
      else
      {
        *v30 = 39;
        v31 = (void *)(*(_QWORD *)(v29 + 24) + 1LL);
        *(_QWORD *)(v29 + 24) = v31;
      }
      if ( a4 > *(_QWORD *)(v29 + 16) - (_QWORD)v31 )
      {
        v39 = sub_16E7EE0(v29, v28, a4);
        v33 = *(__m128i **)(v39 + 24);
        v32 = (_QWORD *)v39;
      }
      else
      {
        srcb = (_QWORD *)v29;
        memcpy(v31, v28, a4);
        v32 = srcb;
        v33 = (__m128i *)(srcb[3] + a4);
        srcb[3] = v33;
      }
      if ( v32[2] - (_QWORD)v33 <= 0x2Eu )
      {
        v38 = sub_16E7EE0(v32, "' is not a recognized processor for this target", 47);
        v35 = *(_QWORD *)(v38 + 24);
        v32 = (_QWORD *)v38;
      }
      else
      {
        *v33 = _mm_load_si128((const __m128i *)&xmmword_3F82940);
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F82970);
        qmemcpy(&v33[2], "for this target", 15);
        v33[1] = si128;
        v35 = v32[3] + 47LL;
        v32[3] = v35;
      }
      if ( (unsigned __int64)(v32[2] - v35) <= 0x15 )
      {
        sub_16E7EE0(v32, " (ignoring processor)\n", 22);
      }
      else
      {
        v36 = _mm_load_si128((const __m128i *)&xmmword_3F82980);
        *(_DWORD *)(v35 + 16) = 1919906675;
        *(_WORD *)(v35 + 20) = 2601;
        *(__m128i *)v35 = v36;
        v32[3] += 22LL;
      }
    }
    goto LABEL_7;
  }
  if ( a4 )
    goto LABEL_13;
LABEL_7:
  v12 = (char *)*a2;
  src = (char *)a2[1];
  if ( src != *a2 )
  {
    do
    {
      if ( !(unsigned int)sub_2241AC0(v12, "+help") )
        sub_1680D70(a5, a6, a7, a8);
      v13 = *(char **)v12;
      v14 = *((_QWORD *)v12 + 1);
      v12 += 32;
      sub_1681780(&v45, v13, v14, a7, a8);
    }
    while ( src != v12 );
  }
  v26 = v46;
  *a1 = _mm_loadu_si128(&v45);
  a1[1].m128i_i64[0] = v26;
  return a1;
}
