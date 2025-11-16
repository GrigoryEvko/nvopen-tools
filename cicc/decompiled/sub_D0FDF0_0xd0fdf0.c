// Function: sub_D0FDF0
// Address: 0xd0fdf0
//
__int64 __fastcall sub_D0FDF0(unsigned __int64 a1, __int64 a2)
{
  __m128i *v4; // rdx
  __int64 v5; // r14
  unsigned __int64 v6; // rax
  __m128i si128; // xmm0
  __int64 v8; // r13
  const char *v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // rdi
  unsigned __int8 *v12; // rsi
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
  _QWORD *v26; // rdx
  char *v27; // rdx
  __int64 v28; // r13
  unsigned __int64 v29; // rax
  __int64 v30; // r14
  const char *v31; // rax
  size_t v32; // rdx
  _WORD *v33; // rdi
  unsigned __int8 *v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  __m128i v40; // xmm0
  size_t v41; // [rsp+8h] [rbp-38h]

  v4 = *(__m128i **)(a2 + 32);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(a2 + 24) - (_QWORD)v4;
  if ( v5 )
  {
    if ( v6 <= 0x1E )
    {
      v8 = sub_CB6200(a2, "Call graph node for function: '", 0x1Fu);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F70810);
      v8 = a2;
      qmemcpy(&v4[1], "for function: '", 15);
      *v4 = si128;
      *(_QWORD *)(a2 + 32) += 31LL;
    }
    v9 = sub_BD5D20(v5);
    v11 = *(_BYTE **)(v8 + 32);
    v12 = (unsigned __int8 *)v9;
    v13 = *(_BYTE **)(v8 + 24);
    v14 = v10;
    if ( v10 > v13 - v11 )
    {
      v8 = sub_CB6200(v8, v12, v10);
      v13 = *(_BYTE **)(v8 + 24);
      v11 = *(_BYTE **)(v8 + 32);
    }
    else if ( v10 )
    {
      memcpy(v11, v12, v10);
      v13 = *(_BYTE **)(v8 + 24);
      v11 = (_BYTE *)(v14 + *(_QWORD *)(v8 + 32));
      *(_QWORD *)(v8 + 32) = v11;
    }
    if ( v11 == v13 )
    {
      sub_CB6200(v8, (unsigned __int8 *)"'", 1u);
    }
    else
    {
      *v11 = 39;
      ++*(_QWORD *)(v8 + 32);
    }
    goto LABEL_10;
  }
  if ( v6 <= 0x20 )
  {
    sub_CB6200(a2, "Call graph node <<null function>>", 0x21u);
LABEL_10:
    v15 = *(_WORD **)(a2 + 32);
    goto LABEL_11;
  }
  v40 = _mm_load_si128((const __m128i *)&xmmword_3F70810);
  v4[2].m128i_i8[0] = 62;
  *v4 = v40;
  v4[1] = _mm_load_si128((const __m128i *)&xmmword_3F70820);
  v15 = (_WORD *)(*(_QWORD *)(a2 + 32) + 33LL);
  *(_QWORD *)(a2 + 32) = v15;
LABEL_11:
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v15 <= 1u )
  {
    v16 = sub_CB6200(a2, (unsigned __int8 *)"<<", 2u);
  }
  else
  {
    *v15 = 15420;
    v16 = a2;
    *(_QWORD *)(a2 + 32) += 2LL;
  }
  v17 = sub_CB5A80(v16, a1);
  v18 = *(void **)(v17 + 32);
  v19 = v17;
  if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 9u )
  {
    v19 = sub_CB6200(v17, ">>  #uses=", 0xAu);
  }
  else
  {
    qmemcpy(v18, ">>  #uses=", 10);
    *(_QWORD *)(v17 + 32) += 10LL;
  }
  v20 = sub_CB59D0(v19, *(unsigned int *)(a1 + 40));
  v21 = *(_BYTE **)(v20 + 32);
  if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 24) )
  {
    sub_CB5D20(v20, 10);
  }
  else
  {
    *(_QWORD *)(v20 + 32) = v21 + 1;
    *v21 = 10;
  }
  v22 = *(_QWORD *)(a1 + 16);
  v23 = *(_QWORD *)(a1 + 24);
  v24 = *(_QWORD *)(a2 + 32);
  if ( v22 != v23 )
  {
    for ( i = v22; v23 != i; i += 40 )
    {
      while ( 1 )
      {
        if ( *(_QWORD *)(a2 + 24) - v24 <= 4 )
        {
          v36 = sub_CB6200(a2, "  CS<", 5u);
        }
        else
        {
          *(_DWORD *)v24 = 1396908064;
          v36 = a2;
          *(_BYTE *)(v24 + 4) = 60;
          *(_QWORD *)(a2 + 32) += 5LL;
        }
        if ( *(_BYTE *)(i + 24) )
          sub_CB5A80(v36, *(_QWORD *)(i + 16));
        else
          sub_F03F40(v36);
        v26 = *(_QWORD **)(v36 + 32);
        if ( *(_QWORD *)(v36 + 24) - (_QWORD)v26 <= 7u )
        {
          sub_CB6200(v36, "> calls ", 8u);
        }
        else
        {
          *v26 = 0x20736C6C6163203ELL;
          *(_QWORD *)(v36 + 32) += 8LL;
        }
        v27 = *(char **)(a2 + 32);
        v28 = *(_QWORD *)(*(_QWORD *)(i + 32) + 8LL);
        v29 = *(_QWORD *)(a2 + 24) - (_QWORD)v27;
        if ( v28 )
          break;
        if ( v29 <= 0xD )
        {
          sub_CB6200(a2, "external node\n", 0xEu);
          goto LABEL_30;
        }
        i += 40;
        qmemcpy(v27, "external node\n", 14);
        v24 = *(_QWORD *)(a2 + 32) + 14LL;
        *(_QWORD *)(a2 + 32) = v24;
        if ( v23 == i )
          goto LABEL_39;
      }
      if ( v29 <= 9 )
      {
        v30 = sub_CB6200(a2, (unsigned __int8 *)"function '", 0xAu);
      }
      else
      {
        v30 = a2;
        qmemcpy(v27, "function '", 10);
        *(_QWORD *)(a2 + 32) += 10LL;
      }
      v31 = sub_BD5D20(v28);
      v33 = *(_WORD **)(v30 + 32);
      v34 = (unsigned __int8 *)v31;
      v35 = *(_QWORD *)(v30 + 24) - (_QWORD)v33;
      if ( v35 < v32 )
      {
        v38 = sub_CB6200(v30, v34, v32);
        v33 = *(_WORD **)(v38 + 32);
        v30 = v38;
        if ( *(_QWORD *)(v38 + 24) - (_QWORD)v33 <= 1u )
          goto LABEL_42;
      }
      else
      {
        if ( v32 )
        {
          v41 = v32;
          memcpy(v33, v34, v32);
          v39 = *(_QWORD *)(v30 + 24);
          v33 = (_WORD *)(v41 + *(_QWORD *)(v30 + 32));
          *(_QWORD *)(v30 + 32) = v33;
          v35 = v39 - (_QWORD)v33;
        }
        if ( v35 <= 1 )
        {
LABEL_42:
          sub_CB6200(v30, (unsigned __int8 *)"'\n", 2u);
          goto LABEL_30;
        }
      }
      *v33 = 2599;
      *(_QWORD *)(v30 + 32) += 2LL;
LABEL_30:
      v24 = *(_QWORD *)(a2 + 32);
    }
  }
LABEL_39:
  if ( v24 >= *(_QWORD *)(a2 + 24) )
    return sub_CB5D20(a2, 10);
  *(_QWORD *)(a2 + 32) = v24 + 1;
  *(_BYTE *)v24 = 10;
  return v24 + 1;
}
