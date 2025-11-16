// Function: sub_104C7F0
// Address: 0x104c7f0
//
__int64 __fastcall sub_104C7F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r12
  __m128i *v9; // rdx
  const char *v10; // rax
  size_t v11; // rdx
  _BYTE *v12; // rdi
  unsigned __int8 *v13; // rsi
  _BYTE *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r14
  __m128i *v18; // rdx
  __m128i si128; // xmm0
  __m128i *v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rdx
  unsigned __int8 **v25; // rbx
  unsigned __int8 **v26; // r14
  _BYTE *v27; // rax
  _BYTE *v28; // rax
  _BYTE *v30; // rax
  __m128i v31; // xmm0
  __int64 v32; // rdi
  __int64 v33; // rax
  void *v34; // rdx
  size_t v35; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  v9 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0x1Fu )
  {
    v8 = sub_CB6200(*a2, "PostDominatorTree for function: ", 0x20u);
  }
  else
  {
    *v9 = _mm_load_si128((const __m128i *)&xmmword_3F8E620);
    v9[1] = _mm_load_si128((const __m128i *)&xmmword_3F8E630);
    *(_QWORD *)(v8 + 32) += 32LL;
  }
  v10 = sub_BD5D20(a3);
  v12 = *(_BYTE **)(v8 + 32);
  v13 = (unsigned __int8 *)v10;
  v14 = *(_BYTE **)(v8 + 24);
  if ( v14 - v12 < v11 )
  {
    v8 = sub_CB6200(v8, v13, v11);
    v14 = *(_BYTE **)(v8 + 24);
    v12 = *(_BYTE **)(v8 + 32);
  }
  else if ( v11 )
  {
    v35 = v11;
    memcpy(v12, v13, v11);
    v30 = *(_BYTE **)(v8 + 24);
    v12 = (_BYTE *)(v35 + *(_QWORD *)(v8 + 32));
    *(_QWORD *)(v8 + 32) = v12;
    if ( v12 != v30 )
      goto LABEL_6;
    goto LABEL_27;
  }
  if ( v12 != v14 )
  {
LABEL_6:
    *v12 = 10;
    ++*(_QWORD *)(v8 + 32);
    goto LABEL_7;
  }
LABEL_27:
  sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
LABEL_7:
  v15 = sub_BC1CD0(a4, &unk_4F8FBC8, a3);
  v16 = *a2;
  v17 = v15;
  v18 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v18 <= 0x3Du )
  {
    sub_CB6200(*a2, "=============================--------------------------------\n", 0x3Eu);
    v20 = *(__m128i **)(v16 + 32);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E640);
    qmemcpy(&v18[3], "-------------\n", 14);
    *v18 = si128;
    v18[1] = _mm_load_si128((const __m128i *)&xmmword_3F8E650);
    v18[2] = _mm_load_si128((const __m128i *)&xmmword_3F8E660);
    v20 = (__m128i *)(*(_QWORD *)(v16 + 32) + 62LL);
    *(_QWORD *)(v16 + 32) = v20;
  }
  if ( *(_QWORD *)(v16 + 24) - (_QWORD)v20 <= 0x1Bu )
  {
    sub_CB6200(v16, "Inorder PostDominator Tree: ", 0x1Cu);
    v22 = *(_QWORD *)(v16 + 32);
  }
  else
  {
    v21 = _mm_load_si128((const __m128i *)&xmmword_3F8E670);
    qmemcpy(&v20[1], "nator Tree: ", 12);
    *v20 = v21;
    v22 = *(_QWORD *)(v16 + 32) + 28LL;
    *(_QWORD *)(v16 + 32) = v22;
  }
  if ( !*(_BYTE *)(v17 + 144) )
  {
    if ( (unsigned __int64)(*(_QWORD *)(v16 + 24) - v22) <= 0x13 )
    {
      v32 = sub_CB6200(v16, "DFSNumbers invalid: ", 0x14u);
    }
    else
    {
      v31 = _mm_load_si128((const __m128i *)&xmmword_3F8E680);
      *(_DWORD *)(v22 + 16) = 540697705;
      v32 = v16;
      *(__m128i *)v22 = v31;
      *(_QWORD *)(v16 + 32) += 20LL;
    }
    v33 = sub_CB59D0(v32, *(unsigned int *)(v17 + 148));
    v34 = *(void **)(v33 + 32);
    if ( *(_QWORD *)(v33 + 24) - (_QWORD)v34 <= 0xDu )
    {
      sub_CB6200(v33, " slow queries.", 0xEu);
    }
    else
    {
      qmemcpy(v34, " slow queries.", 14);
      *(_QWORD *)(v33 + 32) += 14LL;
    }
    v22 = *(_QWORD *)(v16 + 32);
  }
  if ( *(_QWORD *)(v16 + 24) == v22 )
  {
    sub_CB6200(v16, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *(_BYTE *)v22 = 10;
    ++*(_QWORD *)(v16 + 32);
  }
  v23 = *(_QWORD *)(v17 + 128);
  if ( v23 )
    sub_B1AF60(v23, v16, 1u);
  v24 = *(_QWORD *)(v16 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v16 + 24) - v24) <= 6 )
  {
    sub_CB6200(v16, "Roots: ", 7u);
  }
  else
  {
    *(_DWORD *)v24 = 1953460050;
    *(_WORD *)(v24 + 4) = 14963;
    *(_BYTE *)(v24 + 6) = 32;
    *(_QWORD *)(v16 + 32) += 7LL;
  }
  v25 = *(unsigned __int8 ***)(v17 + 8);
  v26 = &v25[*(unsigned int *)(v17 + 16)];
  while ( v26 != v25 )
  {
    while ( 1 )
    {
      sub_A5BF40(*v25, v16, 0, 0);
      v27 = *(_BYTE **)(v16 + 32);
      if ( *(_BYTE **)(v16 + 24) == v27 )
        break;
      ++v25;
      *v27 = 32;
      ++*(_QWORD *)(v16 + 32);
      if ( v26 == v25 )
        goto LABEL_23;
    }
    ++v25;
    sub_CB6200(v16, (unsigned __int8 *)" ", 1u);
  }
LABEL_23:
  v28 = *(_BYTE **)(v16 + 32);
  if ( *(_BYTE **)(v16 + 24) == v28 )
  {
    sub_CB6200(v16, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v28 = 10;
    ++*(_QWORD *)(v16 + 32);
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
