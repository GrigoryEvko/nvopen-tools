// Function: sub_1397A10
// Address: 0x1397a10
//
__int64 __fastcall sub_1397A10(__int64 *a1, __int64 a2)
{
  __m128i *v4; // rdx
  __int64 v5; // r14
  unsigned __int64 v6; // rax
  __m128i si128; // xmm0
  __int64 v8; // r13
  __int64 v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // rdi
  const char *v12; // rsi
  _BYTE *v13; // rax
  size_t v14; // r14
  _WORD *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  void *v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  unsigned __int64 v24; // rdx
  __int64 i; // r15
  __int64 v26; // rdi
  __int64 v27; // rax
  _QWORD *v28; // rdx
  char *v29; // rdx
  __int64 v30; // r13
  unsigned __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // rax
  size_t v34; // rdx
  _WORD *v35; // rdi
  const char *v36; // rsi
  unsigned __int64 v37; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __m128i v41; // xmm0
  size_t v42; // [rsp+8h] [rbp-38h]

  v4 = *(__m128i **)(a2 + 24);
  v5 = *a1;
  v6 = *(_QWORD *)(a2 + 16) - (_QWORD)v4;
  if ( *a1 )
  {
    if ( v6 <= 0x1E )
    {
      v8 = sub_16E7EE0(a2, "Call graph node for function: '", 31);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F70810);
      v8 = a2;
      qmemcpy(&v4[1], "for function: '", 15);
      *v4 = si128;
      *(_QWORD *)(a2 + 24) += 31LL;
    }
    v9 = sub_1649960(v5);
    v11 = *(_BYTE **)(v8 + 24);
    v12 = (const char *)v9;
    v13 = *(_BYTE **)(v8 + 16);
    v14 = v10;
    if ( v10 > v13 - v11 )
    {
      v8 = sub_16E7EE0(v8, v12, v10);
      v13 = *(_BYTE **)(v8 + 16);
      v11 = *(_BYTE **)(v8 + 24);
    }
    else if ( v10 )
    {
      memcpy(v11, v12, v10);
      v13 = *(_BYTE **)(v8 + 16);
      v11 = (_BYTE *)(v14 + *(_QWORD *)(v8 + 24));
      *(_QWORD *)(v8 + 24) = v11;
    }
    if ( v11 == v13 )
    {
      sub_16E7EE0(v8, "'", 1);
    }
    else
    {
      *v11 = 39;
      ++*(_QWORD *)(v8 + 24);
    }
    goto LABEL_10;
  }
  if ( v6 <= 0x20 )
  {
    sub_16E7EE0(a2, "Call graph node <<null function>>", 33);
LABEL_10:
    v15 = *(_WORD **)(a2 + 24);
    goto LABEL_11;
  }
  v41 = _mm_load_si128((const __m128i *)&xmmword_3F70810);
  v4[2].m128i_i8[0] = 62;
  *v4 = v41;
  v4[1] = _mm_load_si128((const __m128i *)&xmmword_3F70820);
  v15 = (_WORD *)(*(_QWORD *)(a2 + 24) + 33LL);
  *(_QWORD *)(a2 + 24) = v15;
LABEL_11:
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v15 <= 1u )
  {
    v16 = sub_16E7EE0(a2, "<<", 2);
  }
  else
  {
    *v15 = 15420;
    v16 = a2;
    *(_QWORD *)(a2 + 24) += 2LL;
  }
  v17 = sub_16E7B40(v16, a1);
  v18 = *(void **)(v17 + 24);
  v19 = v17;
  if ( *(_QWORD *)(v17 + 16) - (_QWORD)v18 <= 9u )
  {
    v19 = sub_16E7EE0(v17, ">>  #uses=", 10);
  }
  else
  {
    qmemcpy(v18, ">>  #uses=", 10);
    *(_QWORD *)(v17 + 24) += 10LL;
  }
  v20 = sub_16E7A90(v19, *((unsigned int *)a1 + 8));
  v21 = *(_BYTE **)(v20 + 24);
  if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 16) )
  {
    sub_16E7DE0(v20, 10);
  }
  else
  {
    *(_QWORD *)(v20 + 24) = v21 + 1;
    *v21 = 10;
  }
  v22 = a1[1];
  v23 = a1[2];
  v24 = *(_QWORD *)(a2 + 24);
  if ( v22 != v23 )
  {
    for ( i = v22; v23 != i; i += 32 )
    {
      while ( 1 )
      {
        if ( *(_QWORD *)(a2 + 16) - v24 > 4 )
        {
          *(_DWORD *)v24 = 1396908064;
          v26 = a2;
          *(_BYTE *)(v24 + 4) = 60;
          *(_QWORD *)(a2 + 24) += 5LL;
        }
        else
        {
          v26 = sub_16E7EE0(a2, "  CS<", 5);
        }
        v27 = sub_16E7B40(v26, *(_QWORD *)(i + 16));
        v28 = *(_QWORD **)(v27 + 24);
        if ( *(_QWORD *)(v27 + 16) - (_QWORD)v28 <= 7u )
        {
          sub_16E7EE0(v27, "> calls ", 8);
        }
        else
        {
          *v28 = 0x20736C6C6163203ELL;
          *(_QWORD *)(v27 + 24) += 8LL;
        }
        v29 = *(char **)(a2 + 24);
        v30 = **(_QWORD **)(i + 24);
        v31 = *(_QWORD *)(a2 + 16) - (_QWORD)v29;
        if ( v30 )
          break;
        if ( v31 <= 0xD )
        {
          sub_16E7EE0(a2, "external node\n", 14);
          goto LABEL_30;
        }
        i += 32;
        qmemcpy(v29, "external node\n", 14);
        v24 = *(_QWORD *)(a2 + 24) + 14LL;
        *(_QWORD *)(a2 + 24) = v24;
        if ( v23 == i )
          goto LABEL_36;
      }
      if ( v31 <= 9 )
      {
        v32 = sub_16E7EE0(a2, "function '", 10);
      }
      else
      {
        v32 = a2;
        qmemcpy(v29, "function '", 10);
        *(_QWORD *)(a2 + 24) += 10LL;
      }
      v33 = sub_1649960(v30);
      v35 = *(_WORD **)(v32 + 24);
      v36 = (const char *)v33;
      v37 = *(_QWORD *)(v32 + 16) - (_QWORD)v35;
      if ( v37 < v34 )
      {
        v39 = sub_16E7EE0(v32, v36);
        v35 = *(_WORD **)(v39 + 24);
        v32 = v39;
        if ( *(_QWORD *)(v39 + 16) - (_QWORD)v35 <= 1u )
          goto LABEL_39;
      }
      else
      {
        if ( v34 )
        {
          v42 = v34;
          memcpy(v35, v36, v34);
          v40 = *(_QWORD *)(v32 + 16);
          v35 = (_WORD *)(v42 + *(_QWORD *)(v32 + 24));
          *(_QWORD *)(v32 + 24) = v35;
          v37 = v40 - (_QWORD)v35;
        }
        if ( v37 <= 1 )
        {
LABEL_39:
          sub_16E7EE0(v32, "'\n", 2);
          goto LABEL_30;
        }
      }
      *v35 = 2599;
      *(_QWORD *)(v32 + 24) += 2LL;
LABEL_30:
      v24 = *(_QWORD *)(a2 + 24);
    }
  }
LABEL_36:
  if ( v24 >= *(_QWORD *)(a2 + 16) )
    return sub_16E7DE0(a2, 10);
  *(_QWORD *)(a2 + 24) = v24 + 1;
  *(_BYTE *)v24 = 10;
  return v24 + 1;
}
