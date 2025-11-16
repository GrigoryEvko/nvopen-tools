// Function: sub_1C31E60
// Address: 0x1c31e60
//
__int64 __fastcall sub_1C31E60(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // r13
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  const char *v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  char *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  void *v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v18; // rdi
  _WORD *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  _BYTE *v22; // rdx
  size_t v23; // [rsp+8h] [rbp-28h]

  sub_1C31A90(a3, *(_QWORD *)(a1 + 24));
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v5 = *(_QWORD *)(a1 + 24);
    v6 = *(__m128i **)(v5 + 24);
    if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x12u )
    {
      v5 = sub_16E7EE0(*(_QWORD *)(a1 + 24), ": Global Variable `", 0x13u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42D04F0);
      v6[1].m128i_i8[2] = 96;
      v6[1].m128i_i16[0] = 8293;
      *v6 = si128;
      *(_QWORD *)(v5 + 24) += 19LL;
    }
    v8 = sub_1649960(a2);
    v10 = *(_BYTE **)(v5 + 24);
    v11 = (char *)v8;
    v12 = *(_QWORD *)(v5 + 16) - (_QWORD)v10;
    if ( v12 < v9 )
    {
      v20 = sub_16E7EE0(v5, v11, v9);
      v10 = *(_BYTE **)(v20 + 24);
      v5 = v20;
      v12 = *(_QWORD *)(v20 + 16) - (_QWORD)v10;
    }
    else if ( v9 )
    {
      v23 = v9;
      memcpy(v10, v11, v9);
      v21 = *(_QWORD *)(v5 + 16);
      v22 = (_BYTE *)(*(_QWORD *)(v5 + 24) + v23);
      *(_QWORD *)(v5 + 24) = v22;
      v10 = v22;
      v12 = v21 - (_QWORD)v22;
    }
    if ( v12 <= 2 )
    {
      sub_16E7EE0(v5, "': ", 3u);
    }
    else
    {
      v10[2] = 32;
      *(_WORD *)v10 = 14887;
      *(_QWORD *)(v5 + 24) += 3LL;
    }
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 24);
    v19 = *(_WORD **)(v18 + 24);
    if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 1u )
    {
      sub_16E7EE0(v18, ": ", 2u);
    }
    else
    {
      *v19 = 8250;
      *(_QWORD *)(v18 + 24) += 2LL;
    }
  }
  v13 = *(_QWORD *)(a1 + 24);
  v14 = *(void **)(v13 + 24);
  if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 0xBu )
  {
    sub_16E7EE0(v13, "\n  context: ", 0xCu);
  }
  else
  {
    qmemcpy(v14, "\n  context: ", 12);
    *(_QWORD *)(v13 + 24) += 12LL;
  }
  sub_155C2B0(a2, *(_QWORD *)(a1 + 24), 0);
  v15 = *(_QWORD *)(a1 + 24);
  v16 = *(_QWORD *)(v15 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v15 + 16) - v16) <= 2 )
  {
    sub_16E7EE0(v15, "\n  ", 3u);
  }
  else
  {
    *(_BYTE *)(v16 + 2) = 32;
    *(_WORD *)v16 = 8202;
    *(_QWORD *)(v15 + 24) += 3LL;
  }
  return *(_QWORD *)(a1 + 24);
}
