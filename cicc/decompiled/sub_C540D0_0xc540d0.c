// Function: sub_C540D0
// Address: 0xc540d0
//
__int64 __fastcall sub_C540D0(__int128 a1, unsigned int a2, int a3)
{
  __m128i *v5; // rdi
  char *v6; // rsi
  char *v7; // rax
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  void *v13; // rdi
  void *v14; // r13
  __int64 result; // rax
  char *v16; // rax
  char *v17; // rdx
  char *v18; // rsi
  unsigned __int64 v19; // rdi
  char *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r12
  char *v23; // rdi
  char *v24; // rdx
  __m128i v25; // xmm0
  __int64 v26; // rax
  char *v27; // [rsp+8h] [rbp-78h]
  __int128 v28; // [rsp+10h] [rbp-70h] BYREF
  char v29; // [rsp+2Fh] [rbp-51h] BYREF
  void *src[2]; // [rsp+30h] [rbp-50h]
  char *v31; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v32; // [rsp+48h] [rbp-38h]

  v28 = a1;
  v5 = (__m128i *)&v28;
  v6 = &v29;
  v29 = 10;
  v7 = (char *)sub_C931B0(&v28, &v29, 1, 0);
  if ( v7 == (char *)-1LL )
  {
    v25 = _mm_loadu_si128((const __m128i *)&v28);
    v31 = 0;
    v32 = 0;
    *(__m128i *)src = v25;
  }
  else
  {
    v6 = v7 + 1;
    if ( (unsigned __int64)(v7 + 1) > *((_QWORD *)&v28 + 1) )
    {
      v6 = (char *)*((_QWORD *)&v28 + 1);
      v5 = 0;
    }
    else
    {
      v5 = (__m128i *)(*((_QWORD *)&v28 + 1) - (_QWORD)v6);
    }
    src[0] = (void *)v28;
    if ( (unsigned __int64)v7 > *((_QWORD *)&v28 + 1) )
      v7 = (char *)*((_QWORD *)&v28 + 1);
    v32 = (unsigned __int64)v5;
    v31 = &v6[v28];
    src[1] = v7;
  }
  v8 = sub_CB7210(v5, v6);
  v9 = a2;
  v10 = sub_CB69B0(v8, a2 - a3);
  v11 = *(_QWORD *)(v10 + 32);
  v12 = v10;
  if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v11) <= 2 )
  {
    v26 = sub_CB6200(v10, " - ", 3);
    v13 = *(void **)(v26 + 32);
    v12 = v26;
  }
  else
  {
    *(_BYTE *)(v11 + 2) = 32;
    *(_WORD *)v11 = 11552;
    v13 = (void *)(*(_QWORD *)(v10 + 32) + 3LL);
    *(_QWORD *)(v10 + 32) = v13;
  }
  v14 = src[1];
  if ( (void *)(*(_QWORD *)(v12 + 24) - (_QWORD)v13) < src[1] )
  {
    v12 = sub_CB6200(v12, src[0], src[1]);
  }
  else if ( src[1] )
  {
    memcpy(v13, src[0], (size_t)src[1]);
    *(_QWORD *)(v12 + 32) += v14;
  }
  result = sub_904010(v12, "\n");
  while ( v32 )
  {
    v29 = 10;
    v16 = (char *)sub_C931B0(&v31, &v29, 1, 0);
    if ( v16 == (char *)-1LL )
    {
      v18 = v31;
      v16 = (char *)v32;
      v19 = 0;
      v20 = 0;
    }
    else
    {
      v17 = v16 + 1;
      v18 = v31;
      if ( (unsigned __int64)(v16 + 1) > v32 )
      {
        v17 = (char *)v32;
        v19 = 0;
      }
      else
      {
        v19 = v32 - (_QWORD)v17;
      }
      v20 = &v17[(_QWORD)v31];
      if ( (unsigned __int64)v16 > v32 )
        v16 = (char *)v32;
    }
    v31 = v20;
    src[0] = v18;
    v32 = v19;
    src[1] = v16;
    v21 = sub_CB7210(v19, v18);
    v22 = sub_CB69B0(v21, v9);
    result = *(_QWORD *)(v22 + 24);
    v23 = *(char **)(v22 + 32);
    if ( (void *)(result - (_QWORD)v23) < src[1] )
    {
      v22 = sub_CB6200(v22, src[0], src[1]);
      result = *(_QWORD *)(v22 + 24);
      v23 = *(char **)(v22 + 32);
      if ( v23 == (char *)result )
        goto LABEL_28;
    }
    else
    {
      if ( src[1] )
      {
        v27 = (char *)src[1];
        memcpy(v23, src[0], (size_t)src[1]);
        v24 = &v27[*(_QWORD *)(v22 + 32)];
        *(_QWORD *)(v22 + 32) = v24;
        result = *(_QWORD *)(v22 + 24);
        v23 = v24;
      }
      if ( v23 == (char *)result )
      {
LABEL_28:
        result = sub_CB6200(v22, "\n", 1);
        continue;
      }
    }
    *v23 = 10;
    ++*(_QWORD *)(v22 + 32);
  }
  return result;
}
