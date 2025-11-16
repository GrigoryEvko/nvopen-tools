// Function: sub_22A94B0
// Address: 0x22a94b0
//
char __fastcall sub_22A94B0(unsigned __int8 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // rdi
  void *v10; // rdx
  __int64 v11; // rdi
  _BYTE *v12; // rax
  void *v13; // rdx
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdx
  __m128i si128; // xmm0
  __int64 v18; // rdi
  _BYTE *v19; // rax
  void *v20; // rdx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax

  v7 = *(char **)(a2 + 32);
  if ( a1[3] )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 9u )
    {
      sub_CB6200(a2, "  Symbol: ", 0xAu);
    }
    else
    {
      qmemcpy(v7, "  Symbol: ", 10);
      *(_QWORD *)(a2 + 32) += 10LL;
    }
    sub_A5BF40(a1[3], a2, 1, 0);
    v8 = *(_BYTE **)(a2 + 32);
    if ( *(_BYTE **)(a2 + 24) == v8 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
      v7 = *(char **)(a2 + 32);
    }
    else
    {
      *v8 = 10;
      v7 = (char *)(*(_QWORD *)(a2 + 32) + 1LL);
      *(_QWORD *)(a2 + 32) = v7;
    }
  }
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 0xAu )
  {
    v27 = sub_CB6200(a2, "  Binding:\n", 0xBu);
    v10 = *(void **)(v27 + 32);
    v9 = v27;
  }
  else
  {
    v9 = a2;
    qmemcpy(v7, "  Binding:\n", 11);
    v10 = (void *)(*(_QWORD *)(a2 + 32) + 11LL);
    *(_QWORD *)(a2 + 32) = v10;
  }
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 0xEu )
  {
    v9 = sub_CB6200(v9, "    Record ID: ", 0xFu);
  }
  else
  {
    qmemcpy(v10, "    Record ID: ", 15);
    *(_QWORD *)(v9 + 32) += 15LL;
  }
  v11 = sub_CB59D0(v9, *(unsigned int *)a1);
  v12 = *(_BYTE **)(v11 + 32);
  if ( *(_BYTE **)(v11 + 24) == v12 )
  {
    v26 = sub_CB6200(v11, (unsigned __int8 *)"\n", 1u);
    v13 = *(void **)(v26 + 32);
    v11 = v26;
  }
  else
  {
    *v12 = 10;
    v13 = (void *)(*(_QWORD *)(v11 + 32) + 1LL);
    *(_QWORD *)(v11 + 32) = v13;
  }
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v13 <= 0xAu )
  {
    v11 = sub_CB6200(v11, "    Space: ", 0xBu);
  }
  else
  {
    qmemcpy(v13, "    Space: ", 11);
    *(_QWORD *)(v11 + 32) += 11LL;
  }
  v14 = sub_CB59D0(v11, *((unsigned int *)a1 + 1));
  v15 = *(_BYTE **)(v14 + 32);
  if ( *(_BYTE **)(v14 + 24) == v15 )
  {
    v25 = sub_CB6200(v14, (unsigned __int8 *)"\n", 1u);
    v16 = *(_QWORD *)(v25 + 32);
    v14 = v25;
  }
  else
  {
    *v15 = 10;
    v16 = *(_QWORD *)(v14 + 32) + 1LL;
    *(_QWORD *)(v14 + 32) = v16;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v14 + 24) - v16) <= 0x10 )
  {
    v14 = sub_CB6200(v14, "    Lower Bound: ", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_43665E0);
    *(_BYTE *)(v16 + 16) = 32;
    *(__m128i *)v16 = si128;
    *(_QWORD *)(v14 + 32) += 17LL;
  }
  v18 = sub_CB59D0(v14, *((unsigned int *)a1 + 2));
  v19 = *(_BYTE **)(v18 + 32);
  if ( *(_BYTE **)(v18 + 24) == v19 )
  {
    v24 = sub_CB6200(v18, (unsigned __int8 *)"\n", 1u);
    v20 = *(void **)(v24 + 32);
    v18 = v24;
  }
  else
  {
    *v19 = 10;
    v20 = (void *)(*(_QWORD *)(v18 + 32) + 1LL);
    *(_QWORD *)(v18 + 32) = v20;
  }
  if ( *(_QWORD *)(v18 + 24) - (_QWORD)v20 <= 9u )
  {
    v18 = sub_CB6200(v18, "    Size: ", 0xAu);
  }
  else
  {
    qmemcpy(v20, "    Size: ", 10);
    *(_QWORD *)(v18 + 32) += 10LL;
  }
  v21 = sub_CB59D0(v18, *((unsigned int *)a1 + 3));
  v22 = *(_BYTE **)(v21 + 32);
  if ( *(_BYTE **)(v21 + 24) == v22 )
  {
    sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v22 = 10;
    ++*(_QWORD *)(v21 + 32);
  }
  return sub_22A87A0(a3, a2, a4);
}
