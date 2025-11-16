// Function: sub_E2A930
// Address: 0xe2a930
//
unsigned __int64 __fastcall sub_E2A930(__int64 a1, char **a2)
{
  bool v4; // zf
  char *v5; // rsi
  unsigned __int64 v6; // rax
  char *v7; // rdi
  __m128i si128; // xmm0
  __m128i *v9; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __m128i v14; // xmm0
  __m128i *v15; // rdi
  char *v16; // rsi
  unsigned __int64 v17; // rax
  char *v18; // rdi
  __int64 v19; // r8
  unsigned __int64 v20; // rcx
  _BYTE *v21; // r12
  unsigned __int64 v22; // rax
  _BYTE *v23; // r9
  size_t v24; // r13
  char *v25; // rdi
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  char *v34; // rdi
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  _BYTE v38[43]; // [rsp+15h] [rbp-2Bh] BYREF

  v4 = *(_BYTE *)(a1 + 24) == 0;
  v5 = a2[1];
  v6 = (unsigned __int64)a2[2];
  v7 = *a2;
  if ( !v4 )
  {
    if ( (unsigned __int64)(v5 + 27) > v6 )
    {
      v27 = (unsigned __int64)(v5 + 1019);
      v28 = 2 * v6;
      if ( v27 > v28 )
        a2[2] = (char *)v27;
      else
        a2[2] = (char *)v28;
      v29 = realloc(v7);
      *a2 = (char *)v29;
      v7 = (char *)v29;
      if ( !v29 )
        goto LABEL_40;
      v5 = a2[1];
    }
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7CA20);
    v9 = (__m128i *)&v7[(_QWORD)v5];
    qmemcpy(&v9[1], "read guard'", 11);
    *v9 = si128;
    a2[1] += 27;
    result = *(unsigned int *)(a1 + 28);
    if ( !(_DWORD)result )
      return result;
    goto LABEL_11;
  }
  if ( (unsigned __int64)(v5 + 20) > v6 )
  {
    v11 = (unsigned __int64)(v5 + 1012);
    v12 = 2 * v6;
    if ( v11 > v12 )
      a2[2] = (char *)v11;
    else
      a2[2] = (char *)v12;
    v13 = realloc(v7);
    *a2 = (char *)v13;
    v7 = (char *)v13;
    if ( !v13 )
      goto LABEL_40;
    v5 = a2[1];
  }
  v14 = _mm_load_si128((const __m128i *)&xmmword_3F7CA30);
  v15 = (__m128i *)&v7[(_QWORD)v5];
  v15[1].m128i_i32[0] = 660894305;
  *v15 = v14;
  a2[1] += 20;
  result = *(unsigned int *)(a1 + 28);
  if ( (_DWORD)result )
  {
LABEL_11:
    v16 = a2[1];
    v17 = (unsigned __int64)a2[2];
    v18 = *a2;
    if ( (unsigned __int64)(v16 + 1) > v17 )
    {
      v30 = (unsigned __int64)(v16 + 993);
      v31 = 2 * v17;
      if ( v30 > v31 )
        a2[2] = (char *)v30;
      else
        a2[2] = (char *)v31;
      v32 = realloc(v18);
      *a2 = (char *)v32;
      v18 = (char *)v32;
      if ( !v32 )
        goto LABEL_40;
      v16 = a2[1];
    }
    v16[(_QWORD)v18] = 123;
    v19 = (__int64)(a2[1] + 1);
    a2[1] = (char *)v19;
    v20 = *(unsigned int *)(a1 + 28);
    v21 = v38;
    do
    {
      *--v21 = v20 % 0xA + 48;
      v22 = v20;
      v20 /= 0xAu;
    }
    while ( v22 > 9 );
    v23 = (_BYTE *)(v38 - v21);
    v24 = v38 - v21;
    if ( v38 != v21 )
    {
      v33 = (unsigned __int64)a2[2];
      v34 = *a2;
      if ( (unsigned __int64)&v23[v19] > v33 )
      {
        v35 = (unsigned __int64)&v23[v19 + 992];
        v36 = 2 * v33;
        if ( v35 > v36 )
          a2[2] = (char *)v35;
        else
          a2[2] = (char *)v36;
        v37 = realloc(v34);
        *a2 = (char *)v37;
        v34 = (char *)v37;
        if ( !v37 )
          goto LABEL_40;
        v19 = (__int64)a2[1];
      }
      memcpy(&v34[v19], v21, v24);
      v19 = (__int64)&a2[1][v24];
      a2[1] = (char *)v19;
    }
    result = (unsigned __int64)a2[2];
    v25 = *a2;
    if ( v19 + 1 <= result )
    {
LABEL_20:
      v25[v19] = 125;
      ++a2[1];
      return result;
    }
    v26 = 2 * result;
    if ( v19 + 993 > v26 )
      a2[2] = (char *)(v19 + 993);
    else
      a2[2] = (char *)v26;
    result = realloc(v25);
    *a2 = (char *)result;
    v25 = (char *)result;
    if ( result )
    {
      v19 = (__int64)a2[1];
      goto LABEL_20;
    }
LABEL_40:
    abort();
  }
  return result;
}
