// Function: sub_104CC90
// Address: 0x104cc90
//
_BYTE *__fastcall sub_104CC90(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i v4; // xmm0
  __m128i *v5; // rdx
  __int64 v6; // rax
  __m128i si128; // xmm0
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned __int8 **v11; // rbx
  unsigned __int8 **v12; // r13
  _BYTE *v13; // rax
  _BYTE *result; // rax
  __m128i v15; // xmm0
  __int64 v16; // rdi
  __int64 v17; // rax
  void *v18; // rdx

  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x3Du )
  {
    sub_CB6200(a2, "=============================--------------------------------\n", 0x3Eu);
    v5 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 > 0x1Bu )
    {
LABEL_3:
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E670);
      qmemcpy(&v5[1], "nator Tree: ", 12);
      *v5 = si128;
      v8 = *(_QWORD *)(a2 + 32) + 28LL;
      *(_QWORD *)(a2 + 32) = v8;
      if ( *(_BYTE *)(a1 + 312) )
        goto LABEL_4;
LABEL_19:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 0x13 )
      {
        v16 = sub_CB6200(a2, "DFSNumbers invalid: ", 0x14u);
      }
      else
      {
        v15 = _mm_load_si128((const __m128i *)&xmmword_3F8E680);
        *(_DWORD *)(v8 + 16) = 540697705;
        v16 = a2;
        *(__m128i *)v8 = v15;
        *(_QWORD *)(a2 + 32) += 20LL;
      }
      v17 = sub_CB59D0(v16, *(unsigned int *)(a1 + 316));
      v18 = *(void **)(v17 + 32);
      if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 0xDu )
      {
        sub_CB6200(v17, " slow queries.", 0xEu);
      }
      else
      {
        qmemcpy(v18, " slow queries.", 14);
        *(_QWORD *)(v17 + 32) += 14LL;
      }
      v8 = *(_QWORD *)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) != v8 )
        goto LABEL_5;
      goto LABEL_24;
    }
  }
  else
  {
    v4 = _mm_load_si128((const __m128i *)&xmmword_3F8E640);
    qmemcpy(&v3[3], "-------------\n", 14);
    *v3 = v4;
    v3[1] = _mm_load_si128((const __m128i *)&xmmword_3F8E650);
    v3[2] = _mm_load_si128((const __m128i *)&xmmword_3F8E660);
    v5 = (__m128i *)(*(_QWORD *)(a2 + 32) + 62LL);
    v6 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 32) = v5;
    if ( (unsigned __int64)(v6 - (_QWORD)v5) > 0x1B )
      goto LABEL_3;
  }
  sub_CB6200(a2, "Inorder PostDominator Tree: ", 0x1Cu);
  v8 = *(_QWORD *)(a2 + 32);
  if ( !*(_BYTE *)(a1 + 312) )
    goto LABEL_19;
LABEL_4:
  if ( *(_QWORD *)(a2 + 24) != v8 )
  {
LABEL_5:
    *(_BYTE *)v8 = 10;
    ++*(_QWORD *)(a2 + 32);
    goto LABEL_6;
  }
LABEL_24:
  sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
LABEL_6:
  v9 = *(_QWORD *)(a1 + 296);
  if ( v9 )
    sub_B1AF60(v9, a2, 1u);
  v10 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v10) <= 6 )
  {
    sub_CB6200(a2, "Roots: ", 7u);
  }
  else
  {
    *(_DWORD *)v10 = 1953460050;
    *(_WORD *)(v10 + 4) = 14963;
    *(_BYTE *)(v10 + 6) = 32;
    *(_QWORD *)(a2 + 32) += 7LL;
  }
  v11 = *(unsigned __int8 ***)(a1 + 176);
  v12 = &v11[*(unsigned int *)(a1 + 184)];
  while ( v12 != v11 )
  {
    while ( 1 )
    {
      sub_A5BF40(*v11, a2, 0, 0);
      v13 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v13 )
        break;
      ++v11;
      *v13 = 32;
      ++*(_QWORD *)(a2 + 32);
      if ( v12 == v11 )
        goto LABEL_15;
    }
    ++v11;
    sub_CB6200(a2, (unsigned __int8 *)" ", 1u);
  }
LABEL_15:
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
