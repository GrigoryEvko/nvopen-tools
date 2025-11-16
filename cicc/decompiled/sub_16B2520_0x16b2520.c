// Function: sub_16B2520
// Address: 0x16b2520
//
__int64 __fastcall sub_16B2520(__int128 a1, unsigned int a2, int a3)
{
  __m128i *v5; // rdi
  char *v6; // rsi
  void *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rax
  void *v15; // r13
  void *v16; // rdi
  __int64 v17; // r12
  __int64 result; // rax
  char *v19; // rax
  unsigned __int64 v20; // rcx
  char *v21; // rdx
  void *v22; // rsi
  unsigned __int64 v23; // rdi
  char *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r12
  char *v27; // rdi
  char *v28; // rdx
  __m128i v29; // xmm0
  char *v30; // [rsp+8h] [rbp-78h]
  __int128 v31; // [rsp+10h] [rbp-70h] BYREF
  char v32; // [rsp+2Fh] [rbp-51h] BYREF
  void *src[2]; // [rsp+30h] [rbp-50h]
  __int64 v34; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v35; // [rsp+48h] [rbp-38h]

  v31 = a1;
  v5 = (__m128i *)&v31;
  v6 = &v32;
  v32 = 10;
  v7 = (void *)sub_16D20C0(&v31, &v32, 1, 0);
  if ( v7 == (void *)-1LL )
  {
    v29 = _mm_loadu_si128((const __m128i *)&v31);
    v34 = 0;
    v35 = 0;
    *(__m128i *)src = v29;
  }
  else
  {
    v9 = *((_QWORD *)&v31 + 1);
    v10 = (__int64)v7 + 1;
    v6 = (char *)v31;
    if ( (unsigned __int64)v7 + 1 > *((_QWORD *)&v31 + 1) )
      v10 = *((_QWORD *)&v31 + 1);
    v5 = (__m128i *)(*((_QWORD *)&v31 + 1) - v10);
    v8 = v31 + v10;
    if ( v7 && (unsigned __int64)v7 > *((_QWORD *)&v31 + 1) )
      v7 = (void *)*((_QWORD *)&v31 + 1);
    src[0] = (void *)v31;
    src[1] = v7;
    v34 = v8;
    v35 = (unsigned __int64)v5;
  }
  v11 = sub_16E8C20(v5, v6, v8, v9);
  v12 = a2;
  v13 = sub_16E8750(v11, a2 - a3);
  v14 = sub_1263B40(v13, " - ");
  v15 = src[1];
  v16 = *(void **)(v14 + 24);
  v17 = v14;
  if ( (void *)(*(_QWORD *)(v14 + 16) - (_QWORD)v16) < src[1] )
  {
    v17 = sub_16E7EE0(v14, (const char *)src[0], src[1]);
  }
  else if ( src[1] )
  {
    memcpy(v16, src[0], (size_t)src[1]);
    *(_QWORD *)(v17 + 24) += v15;
  }
  result = sub_1263B40(v17, "\n");
  while ( v35 )
  {
    v32 = 10;
    v19 = (char *)sub_16D20C0(&v34, &v32, 1, 0);
    if ( v19 == (char *)-1LL )
    {
      v22 = (void *)v34;
      v19 = (char *)v35;
      v23 = 0;
      v24 = 0;
    }
    else
    {
      v20 = v35;
      v21 = v19 + 1;
      v22 = (void *)v34;
      if ( (unsigned __int64)(v19 + 1) > v35 )
        v21 = (char *)v35;
      v23 = v35 - (_QWORD)v21;
      v24 = &v21[v34];
      if ( v19 && (unsigned __int64)v19 > v35 )
        v19 = (char *)v35;
    }
    v34 = (__int64)v24;
    src[0] = v22;
    v35 = v23;
    src[1] = v19;
    v25 = sub_16E8C20(v23, v22, v24, v20);
    v26 = sub_16E8750(v25, v12);
    result = *(_QWORD *)(v26 + 16);
    v27 = *(char **)(v26 + 24);
    if ( (void *)(result - (_QWORD)v27) < src[1] )
    {
      v26 = sub_16E7EE0(v26, (const char *)src[0]);
      result = *(_QWORD *)(v26 + 16);
      v27 = *(char **)(v26 + 24);
      if ( v27 == (char *)result )
        goto LABEL_25;
    }
    else
    {
      if ( src[1] )
      {
        v30 = (char *)src[1];
        memcpy(v27, src[0], (size_t)src[1]);
        v28 = &v30[*(_QWORD *)(v26 + 24)];
        *(_QWORD *)(v26 + 24) = v28;
        result = *(_QWORD *)(v26 + 16);
        v27 = v28;
      }
      if ( v27 == (char *)result )
      {
LABEL_25:
        result = sub_16E7EE0(v26, "\n", 1);
        continue;
      }
    }
    *v27 = 10;
    ++*(_QWORD *)(v26 + 24);
  }
  return result;
}
