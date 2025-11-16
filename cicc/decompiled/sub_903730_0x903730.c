// Function: sub_903730
// Address: 0x903730
//
__int64 __fastcall sub_903730(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  const void *v7; // r15
  size_t v8; // rax
  _QWORD *v9; // rdi
  size_t v10; // r13
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  const char *v14; // r15
  size_t v15; // rax
  _WORD *v16; // rdi
  size_t v17; // r13
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  const char *v21; // r15
  size_t v22; // rax
  _BYTE *v23; // rdi
  size_t v24; // r13
  _BYTE *v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  const void *v29; // r15
  size_t v30; // rax
  void *v31; // rdi
  size_t v32; // r13
  __int64 v33; // r12
  _BYTE *v34; // rax
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __m128i *v37; // rdx
  __m128i si128; // xmm0
  __int64 v39; // rax
  __int64 v40; // rdx

  v4 = sub_CB7210();
  v5 = *(_QWORD *)(v4 + 32);
  v6 = v4;
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v5) <= 2 )
  {
    v6 = sub_CB6200(v4, "[ \"", 3);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 34;
    *(_WORD *)v5 = 8283;
    *(_QWORD *)(v4 + 32) += 3LL;
  }
  v7 = *(const void **)(a3 + 16);
  if ( !v7 )
    goto LABEL_26;
  v8 = strlen(*(const char **)(a3 + 16));
  v9 = *(_QWORD **)(v6 + 32);
  v10 = v8;
  v11 = *(_QWORD *)(v6 + 24) - (_QWORD)v9;
  if ( v10 > v11 )
  {
    v6 = sub_CB6200(v6, v7, v10);
LABEL_26:
    v9 = *(_QWORD **)(v6 + 32);
    v11 = *(_QWORD *)(v6 + 24) - (_QWORD)v9;
    goto LABEL_27;
  }
  if ( v10 )
  {
    memcpy(v9, v7, v10);
    v9 = (_QWORD *)(v10 + *(_QWORD *)(v6 + 32));
    v12 = *(_QWORD *)(v6 + 24) - (_QWORD)v9;
    *(_QWORD *)(v6 + 32) = v9;
    if ( v12 <= 7 )
      goto LABEL_7;
    goto LABEL_28;
  }
LABEL_27:
  if ( v11 <= 7 )
  {
LABEL_7:
    v13 = sub_CB6200(v6, "\" -llc \"", 8);
    v14 = *(const char **)(a3 + 32);
    v6 = v13;
    if ( v14 )
      goto LABEL_8;
LABEL_29:
    v16 = *(_WORD **)(v6 + 32);
    v18 = *(_QWORD *)(v6 + 24) - (_QWORD)v16;
    goto LABEL_30;
  }
LABEL_28:
  *v9 = 0x2220636C6C2D2022LL;
  *(_QWORD *)(v6 + 32) += 8LL;
  v14 = *(const char **)(a3 + 32);
  if ( !v14 )
    goto LABEL_29;
LABEL_8:
  v15 = strlen(v14);
  v16 = *(_WORD **)(v6 + 32);
  v17 = v15;
  v18 = *(_QWORD *)(v6 + 24) - (_QWORD)v16;
  if ( v17 > v18 )
  {
    v6 = sub_CB6200(v6, v14, v17);
    goto LABEL_29;
  }
  if ( v17 )
  {
    memcpy(v16, v14, v17);
    v16 = (_WORD *)(v17 + *(_QWORD *)(v6 + 32));
    v19 = *(_QWORD *)(v6 + 24) - (_QWORD)v16;
    *(_QWORD *)(v6 + 32) = v16;
    if ( v19 <= 5 )
      goto LABEL_11;
    goto LABEL_31;
  }
LABEL_30:
  if ( v18 <= 5 )
  {
LABEL_11:
    v20 = sub_CB6200(v6, "\" -o \"", 6);
    v21 = *(const char **)(a3 + 8);
    v6 = v20;
    if ( v21 )
      goto LABEL_12;
LABEL_32:
    v25 = *(_BYTE **)(v6 + 24);
    v23 = *(_BYTE **)(v6 + 32);
    goto LABEL_33;
  }
LABEL_31:
  *(_DWORD *)v16 = 1865228322;
  v16[2] = 8736;
  *(_QWORD *)(v6 + 32) += 6LL;
  v21 = *(const char **)(a3 + 8);
  if ( !v21 )
    goto LABEL_32;
LABEL_12:
  v22 = strlen(v21);
  v23 = *(_BYTE **)(v6 + 32);
  v24 = v22;
  v25 = *(_BYTE **)(v6 + 24);
  if ( v24 > v25 - v23 )
  {
    v6 = sub_CB6200(v6, v21, v24);
    goto LABEL_32;
  }
  if ( v24 )
  {
    memcpy(v23, v21, v24);
    v26 = *(_BYTE **)(v6 + 24);
    v23 = (_BYTE *)(v24 + *(_QWORD *)(v6 + 32));
    *(_QWORD *)(v6 + 32) = v23;
    if ( v26 == v23 )
      goto LABEL_15;
    goto LABEL_34;
  }
LABEL_33:
  if ( v25 == v23 )
  {
LABEL_15:
    v27 = 0;
    sub_CB6200(v6, "\"", 1);
    if ( *(int *)(a3 + 52) <= 1 )
      goto LABEL_35;
    goto LABEL_23;
  }
LABEL_34:
  *v23 = 34;
  v27 = 0;
  ++*(_QWORD *)(v6 + 32);
  if ( *(int *)(a3 + 52) <= 1 )
    goto LABEL_35;
  do
  {
LABEL_23:
    v33 = sub_CB7210();
    v34 = *(_BYTE **)(v33 + 32);
    if ( (unsigned __int64)v34 < *(_QWORD *)(v33 + 24) )
    {
      *(_QWORD *)(v33 + 32) = v34 + 1;
      *v34 = 32;
    }
    else
    {
      v33 = sub_CB5D20(v33, 32);
    }
    v28 = *(_QWORD *)(a3 + 80);
    ++v27;
    v29 = *(const void **)(v28 + 8 * v27);
    if ( v29 )
    {
      v30 = strlen(*(const char **)(v28 + 8 * v27));
      v31 = *(void **)(v33 + 32);
      v32 = v30;
      if ( v30 > *(_QWORD *)(v33 + 24) - (_QWORD)v31 )
      {
        sub_CB6200(v33, v29, v30);
      }
      else if ( v30 )
      {
        memcpy(v31, v29, v30);
        *(_QWORD *)(v33 + 32) += v32;
      }
    }
  }
  while ( *(_DWORD *)(a3 + 52) > (int)v27 + 1 );
LABEL_35:
  v35 = sub_CB7210();
  v36 = *(_BYTE **)(v35 + 32);
  if ( (unsigned __int64)v36 >= *(_QWORD *)(v35 + 24) )
  {
    v35 = sub_CB5D20(v35, 32);
  }
  else
  {
    *(_QWORD *)(v35 + 32) = v36 + 1;
    *v36 = 32;
  }
  v37 = *(__m128i **)(v35 + 32);
  if ( *(_QWORD *)(v35 + 24) - (_QWORD)v37 <= 0x18u )
  {
    sub_CB6200(v35, "-nvvm-version=nvvm-latest", 25);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
    v37[1].m128i_i8[8] = 116;
    v37[1].m128i_i64[0] = 0x736574616C2D6D76LL;
    *v37 = si128;
    *(_QWORD *)(v35 + 32) += 25LL;
  }
  v39 = sub_CB7210();
  v40 = *(_QWORD *)(v39 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v39 + 24) - v40) <= 2 )
    return sub_CB6200(v39, " ]\n", 3);
  *(_BYTE *)(v40 + 2) = 10;
  *(_WORD *)v40 = 23840;
  *(_QWORD *)(v39 + 32) += 3LL;
  return 23840;
}
