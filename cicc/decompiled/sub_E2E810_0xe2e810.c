// Function: sub_E2E810
// Address: 0xe2e810
//
__int64 __fastcall sub_E2E810(__int64 a1, __int64 *a2, unsigned int a3)
{
  bool v6; // zf
  char *v7; // rsi
  unsigned __int64 v8; // rax
  char *v9; // rdi
  __m128i v10; // xmm0
  __m128i *v11; // rdi
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  char *v15; // rdi
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  char *v19; // rsi
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rax
  __int64 result; // rax
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  __m128i si128; // xmm0
  __m128i *v28; // rdi
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 *v32; // rdi
  unsigned __int64 (__fastcall *v33)(__int64, char **, unsigned int); // rax
  char *v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // rdi
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rsi
  unsigned __int64 v41; // rax
  __int64 v42; // rax

  v6 = *(_BYTE *)(a1 + 40) == 0;
  v7 = (char *)a2[1];
  v8 = a2[2];
  v9 = (char *)*a2;
  if ( v6 )
  {
    if ( v8 < (unsigned __int64)(v7 + 25) )
    {
      v24 = (unsigned __int64)(v7 + 1017);
      v25 = 2 * v8;
      if ( v24 > v25 )
        a2[2] = v24;
      else
        a2[2] = v25;
      v26 = realloc(v9);
      *a2 = v26;
      v9 = (char *)v26;
      if ( !v26 )
        goto LABEL_47;
      v7 = (char *)a2[1];
    }
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7CA80);
    v28 = (__m128i *)&v9[(_QWORD)v7];
    v28[1].m128i_i64[0] = 0x726F662072657A69LL;
    v28[1].m128i_i8[8] = 32;
    *v28 = si128;
    v12 = a2[1] + 25;
    a2[1] = v12;
  }
  else
  {
    if ( v8 < (unsigned __int64)(v7 + 31) )
    {
      v40 = (unsigned __int64)(v7 + 1023);
      v41 = 2 * v8;
      if ( v40 > v41 )
        a2[2] = v40;
      else
        a2[2] = v41;
      v42 = realloc(v9);
      *a2 = v42;
      v9 = (char *)v42;
      if ( !v42 )
        goto LABEL_47;
      v7 = (char *)a2[1];
    }
    v10 = _mm_load_si128((const __m128i *)&xmmword_3F7CA70);
    v11 = (__m128i *)&v9[(_QWORD)v7];
    qmemcpy(&v11[1], "destructor for ", 15);
    *v11 = v10;
    v12 = a2[1] + 31;
    a2[1] = v12;
  }
  v13 = a2[2];
  v14 = v12 + 1;
  v15 = (char *)*a2;
  if ( *(_QWORD *)(a1 + 24) )
  {
    if ( v14 > v13 )
    {
      v16 = v12 + 993;
      v17 = 2 * v13;
      if ( v16 > v17 )
        a2[2] = v16;
      else
        a2[2] = v17;
      v18 = realloc(v15);
      *a2 = v18;
      v15 = (char *)v18;
      if ( !v18 )
        goto LABEL_47;
      v12 = a2[1];
    }
    v15[v12] = 96;
    ++a2[1];
    (*(void (__fastcall **)(_QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 24) + 16LL))(*(_QWORD *)(a1 + 24), a2, a3);
    v19 = (char *)a2[1];
    v20 = a2[2];
    if ( (unsigned __int64)(v19 + 2) <= v20 )
    {
      result = *a2;
      goto LABEL_22;
    }
    v21 = (unsigned __int64)(v19 + 994);
    v22 = 2 * v20;
    if ( v21 > v22 )
      a2[2] = v21;
    else
      a2[2] = v22;
    result = realloc((void *)*a2);
    *a2 = result;
    if ( result )
    {
      v19 = (char *)a2[1];
LABEL_22:
      *(_WORD *)&v19[result] = 10023;
      a2[1] += 2;
      return result;
    }
LABEL_47:
    abort();
  }
  if ( v14 > v13 )
  {
    v29 = v12 + 993;
    v30 = 2 * v13;
    if ( v29 > v30 )
      a2[2] = v29;
    else
      a2[2] = v30;
    v31 = realloc(v15);
    *a2 = v31;
    v15 = (char *)v31;
    if ( !v31 )
      goto LABEL_47;
    v12 = a2[1];
  }
  v15[v12] = 39;
  ++a2[1];
  v32 = *(__int64 **)(a1 + 32);
  v33 = *(unsigned __int64 (__fastcall **)(__int64, char **, unsigned int))(*v32 + 16);
  if ( v33 == sub_E2CA10 )
    sub_E2C8E0(v32[2], (char **)a2, a3, 2u, "::");
  else
    v33((__int64)v32, (char **)a2, a3);
  v34 = (char *)a2[1];
  v35 = a2[2];
  v36 = *a2;
  if ( (unsigned __int64)(v34 + 2) > v35 )
  {
    v37 = (unsigned __int64)(v34 + 994);
    v38 = 2 * v35;
    if ( v37 > v38 )
      a2[2] = v37;
    else
      a2[2] = v38;
    v39 = realloc((void *)v36);
    *a2 = v39;
    v36 = v39;
    if ( !v39 )
      goto LABEL_47;
    v34 = (char *)a2[1];
  }
  *(_WORD *)&v34[v36] = 10023;
  a2[1] += 2;
  return 10023;
}
