// Function: sub_B85E70
// Address: 0xb85e70
//
void __fastcall sub_B85E70(__int64 a1)
{
  __int64 v2; // rax
  __m128i *v3; // rdx
  void *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // r13
  __int64 i; // r14
  const void *v8; // rsi
  size_t v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // rax
  _WORD *v12; // rdx
  _QWORD *v13; // r15
  __int64 v14; // rax
  void **v15; // rbx
  void **v16; // r12
  __int64 v17; // rdi
  _BYTE *v18; // rax

  if ( (int)qword_4F81B88 <= 0 )
    return;
  v2 = sub_C5F790(a1);
  v3 = *(__m128i **)(v2 + 32);
  v4 = (void *)v2;
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v3 <= 0xFu )
  {
    sub_CB6200(v2, "Pass Arguments: ", 16);
  }
  else
  {
    *v3 = _mm_load_si128((const __m128i *)&xmmword_3F552E0);
    *(_QWORD *)(v2 + 32) += 16LL;
  }
  v5 = *(_QWORD *)(a1 + 256);
  v6 = v5 + 8LL * *(unsigned int *)(a1 + 264);
  for ( i = v5; v6 != i; i += 8 )
  {
    while ( 1 )
    {
      v4 = (void *)a1;
      v10 = sub_B85AD0(a1, *(_QWORD *)(*(_QWORD *)i + 16LL));
      if ( v10 )
        break;
LABEL_10:
      i += 8;
      if ( v6 == i )
        goto LABEL_17;
    }
    v11 = sub_C5F790(a1);
    v12 = *(_WORD **)(v11 + 32);
    v13 = (_QWORD *)v11;
    if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 > 1u )
    {
      *v12 = 11552;
      v4 = (void *)(*(_QWORD *)(v11 + 32) + 2LL);
      *(_QWORD *)(v11 + 32) = v4;
    }
    else
    {
      v14 = sub_CB6200(v11, " -", 2);
      v4 = *(void **)(v14 + 32);
      v13 = (_QWORD *)v14;
    }
    v8 = *(const void **)(v10 + 16);
    v9 = *(_QWORD *)(v10 + 24);
    if ( v9 <= v13[3] - (_QWORD)v4 )
    {
      if ( v9 )
      {
        memcpy(v4, v8, v9);
        v13[4] += v9;
      }
      goto LABEL_10;
    }
    v4 = v13;
    sub_CB6200(v13, v8, v9);
  }
LABEL_17:
  v15 = *(void ***)(a1 + 32);
  v16 = &v15[*(unsigned int *)(a1 + 40)];
  while ( v16 != v15 )
  {
    v4 = *v15++;
    sub_B85D60((__int64)v4);
  }
  v17 = sub_C5F790(v4);
  v18 = *(_BYTE **)(v17 + 32);
  if ( *(_BYTE **)(v17 + 24) == v18 )
  {
    sub_CB6200(v17, "\n", 1);
  }
  else
  {
    *v18 = 10;
    ++*(_QWORD *)(v17 + 32);
  }
}
