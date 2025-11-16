// Function: sub_CFE240
// Address: 0xcfe240
//
__int64 __fastcall sub_CFE240(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r15
  __m128i *v10; // rdx
  __m128i si128; // xmm0
  const char *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // r9
  _BYTE *v16; // rdi
  unsigned __int8 *v17; // rsi
  _BYTE *v18; // rax
  size_t v19; // rdx
  size_t v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // r13
  _BYTE *v24; // rax
  __int64 v25; // r15
  _WORD *v26; // rdx
  _BYTE *v28; // rax

  v7 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v8 = *a2;
  v9 = v7;
  v10 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v10 <= 0x20u )
  {
    v8 = sub_CB6200(*a2, "Cached assumptions for function: ", 0x21u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F705D0);
    v10[2].m128i_i8[0] = 32;
    *v10 = si128;
    v10[1] = _mm_load_si128((const __m128i *)&xmmword_3F705E0);
    *(_QWORD *)(v8 + 32) += 33LL;
  }
  v12 = sub_BD5D20(a3);
  v16 = *(_BYTE **)(v8 + 32);
  v17 = (unsigned __int8 *)v12;
  v18 = *(_BYTE **)(v8 + 24);
  v20 = v19;
  v21 = v18 - v16;
  if ( v20 > v18 - v16 )
  {
    v8 = sub_CB6200(v8, v17, v20);
    v18 = *(_BYTE **)(v8 + 24);
    v16 = *(_BYTE **)(v8 + 32);
LABEL_5:
    if ( v18 != v16 )
      goto LABEL_6;
LABEL_19:
    v17 = (unsigned __int8 *)"\n";
    sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
    if ( *(_BYTE *)(v9 + 200) )
      goto LABEL_7;
    goto LABEL_20;
  }
  if ( !v20 )
    goto LABEL_5;
  memcpy(v16, v17, v20);
  v28 = *(_BYTE **)(v8 + 24);
  v16 = (_BYTE *)(v20 + *(_QWORD *)(v8 + 32));
  *(_QWORD *)(v8 + 32) = v16;
  if ( v28 == v16 )
    goto LABEL_19;
LABEL_6:
  *v16 = 10;
  ++*(_QWORD *)(v8 + 32);
  if ( *(_BYTE *)(v9 + 200) )
    goto LABEL_7;
LABEL_20:
  sub_CFDFC0(v9 + 8, (__int64)v17, v21, v13, v14, v15);
LABEL_7:
  v22 = *(_QWORD *)(v9 + 24);
  v23 = v22 + 32LL * *(unsigned int *)(v9 + 32);
  if ( v23 != v22 )
  {
    while ( 1 )
    {
      if ( !*(_QWORD *)(v22 + 16) )
        goto LABEL_12;
      v25 = *a2;
      v26 = *(_WORD **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v26 > 1u )
      {
        *v26 = 8224;
        *(_QWORD *)(v25 + 32) += 2LL;
      }
      else
      {
        v25 = sub_CB6200(*a2, (unsigned __int8 *)"  ", 2u);
      }
      sub_A69870(
        *(_QWORD *)(*(_QWORD *)(v22 + 16) - 32LL * (*(_DWORD *)(*(_QWORD *)(v22 + 16) + 4LL) & 0x7FFFFFF)),
        (_BYTE *)v25,
        0);
      v24 = *(_BYTE **)(v25 + 32);
      if ( *(_BYTE **)(v25 + 24) == v24 )
      {
        v22 += 32;
        sub_CB6200(v25, (unsigned __int8 *)"\n", 1u);
        if ( v23 == v22 )
          break;
      }
      else
      {
        *v24 = 10;
        ++*(_QWORD *)(v25 + 32);
LABEL_12:
        v22 += 32;
        if ( v23 == v22 )
          break;
      }
    }
  }
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
