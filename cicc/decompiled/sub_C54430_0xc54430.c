// Function: sub_C54430
// Address: 0xc54430
//
__int64 __fastcall sub_C54430(__int128 a1, int a2, int a3)
{
  __m128i *v5; // rdi
  char *v6; // rsi
  char *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  _WORD *v12; // rdx
  void *v13; // rdi
  void *v14; // r14
  unsigned int v15; // ebx
  __int64 result; // rax
  char *v17; // rax
  char *v18; // rdx
  char *v19; // rsi
  unsigned __int64 v20; // rdi
  char *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r12
  char *v24; // rdi
  char *v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rax
  __int64 v28; // rax
  char *v29; // [rsp+8h] [rbp-78h]
  __int128 v30; // [rsp+10h] [rbp-70h] BYREF
  char v31; // [rsp+2Fh] [rbp-51h] BYREF
  void *src[2]; // [rsp+30h] [rbp-50h]
  char *v33; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v34; // [rsp+48h] [rbp-38h]

  v30 = a1;
  v5 = (__m128i *)&v30;
  v6 = &v31;
  v31 = 10;
  v7 = (char *)sub_C931B0(&v30, &v31, 1, 0);
  if ( v7 == (char *)-1LL )
  {
    v26 = _mm_loadu_si128((const __m128i *)&v30);
    v33 = 0;
    v34 = 0;
    *(__m128i *)src = v26;
  }
  else
  {
    v6 = v7 + 1;
    if ( (unsigned __int64)(v7 + 1) > *((_QWORD *)&v30 + 1) )
    {
      v6 = (char *)*((_QWORD *)&v30 + 1);
      v5 = 0;
    }
    else
    {
      v5 = (__m128i *)(*((_QWORD *)&v30 + 1) - (_QWORD)v6);
    }
    src[0] = (void *)v30;
    if ( (unsigned __int64)v7 > *((_QWORD *)&v30 + 1) )
      v7 = (char *)*((_QWORD *)&v30 + 1);
    v34 = (unsigned __int64)v5;
    v33 = &v6[v30];
    src[1] = v7;
  }
  v8 = sub_CB7210(v5, v6);
  v9 = sub_CB69B0(v8, (unsigned int)(a2 - a3));
  v10 = *(_QWORD *)(v9 + 32);
  v11 = v9;
  if ( (unsigned __int64)(*(_QWORD *)(v9 + 24) - v10) <= 2 )
  {
    v28 = sub_CB6200(v9, " - ", 3);
    v12 = *(_WORD **)(v28 + 32);
    v11 = v28;
  }
  else
  {
    *(_BYTE *)(v10 + 2) = 32;
    *(_WORD *)v10 = 11552;
    v12 = (_WORD *)(*(_QWORD *)(v9 + 32) + 3LL);
    *(_QWORD *)(v9 + 32) = v12;
  }
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 1u )
  {
    v27 = sub_CB6200(v11, "  ", 2);
    v13 = *(void **)(v27 + 32);
    v11 = v27;
  }
  else
  {
    *v12 = 8224;
    v13 = (void *)(*(_QWORD *)(v11 + 32) + 2LL);
    *(_QWORD *)(v11 + 32) = v13;
  }
  v14 = src[1];
  if ( (void *)(*(_QWORD *)(v11 + 24) - (_QWORD)v13) < src[1] )
  {
    v11 = sub_CB6200(v11, src[0], src[1]);
  }
  else if ( src[1] )
  {
    memcpy(v13, src[0], (size_t)src[1]);
    *(_QWORD *)(v11 + 32) += v14;
  }
  v15 = a2 + 2;
  result = sub_904010(v11, "\n");
  while ( v34 )
  {
    while ( 1 )
    {
      v31 = 10;
      v17 = (char *)sub_C931B0(&v33, &v31, 1, 0);
      if ( v17 == (char *)-1LL )
      {
        v19 = v33;
        v17 = (char *)v34;
        v20 = 0;
        v21 = 0;
      }
      else
      {
        v18 = v17 + 1;
        v19 = v33;
        if ( (unsigned __int64)(v17 + 1) > v34 )
        {
          v18 = (char *)v34;
          v20 = 0;
        }
        else
        {
          v20 = v34 - (_QWORD)v18;
        }
        v21 = &v18[(_QWORD)v33];
        if ( (unsigned __int64)v17 > v34 )
          v17 = (char *)v34;
      }
      v33 = v21;
      src[0] = v19;
      v34 = v20;
      src[1] = v17;
      v22 = sub_CB7210(v20, v19);
      v23 = sub_CB69B0(v22, v15);
      result = *(_QWORD *)(v23 + 24);
      v24 = *(char **)(v23 + 32);
      if ( src[1] > (void *)(result - (_QWORD)v24) )
      {
        v23 = sub_CB6200(v23, src[0], src[1]);
        result = *(_QWORD *)(v23 + 24);
        v24 = *(char **)(v23 + 32);
      }
      else if ( src[1] )
      {
        v29 = (char *)src[1];
        memcpy(v24, src[0], (size_t)src[1]);
        v25 = &v29[*(_QWORD *)(v23 + 32)];
        *(_QWORD *)(v23 + 32) = v25;
        result = *(_QWORD *)(v23 + 24);
        v24 = v25;
      }
      if ( v24 == (char *)result )
        break;
      *v24 = 10;
      ++*(_QWORD *)(v23 + 32);
      if ( !v34 )
        return result;
    }
    result = sub_CB6200(v23, "\n", 1);
  }
  return result;
}
