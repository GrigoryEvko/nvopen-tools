// Function: sub_1C36220
// Address: 0x1c36220
//
__int64 __fastcall sub_1C36220(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  int v4; // eax
  __int64 v5; // rax
  unsigned int v6; // eax
  __int64 v8; // rax
  __m128i *v9; // rdx
  __int64 v10; // rdi
  __m128i v11; // xmm0
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __int64 v14; // rax
  __m128i *v15; // rdx
  __int64 v16; // rdi
  void *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __m128i *v20; // rdx
  __int64 v21; // rdi
  __m128i si128; // xmm0
  _BYTE *v23; // rax
  __int64 v24; // rax

  v3 = *(__int64 **)(a2 - 24);
  if ( *(_BYTE *)(*v3 + 8) != 11 || (v4 = sub_1643030(*v3), ((v4 - 32) & 0xFFFFFFDF) != 0) && v4 != 128 )
  {
    v14 = sub_1C321C0(a1, a2, 0);
    v15 = *(__m128i **)(v14 + 24);
    v16 = v14;
    if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 0x2Fu )
    {
      v16 = sub_16E7EE0(v14, "Atomic operations on non-i32/i64/i128 types are ", 0x30u);
      v17 = *(void **)(v16 + 24);
      if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 0xDu )
        goto LABEL_15;
    }
    else
    {
      *v15 = _mm_load_si128((const __m128i *)&xmmword_42D0AF0);
      v15[1] = _mm_load_si128((const __m128i *)&xmmword_42D0B00);
      v15[2] = _mm_load_si128((const __m128i *)&xmmword_42D0B10);
      v17 = (void *)(*(_QWORD *)(v14 + 24) + 48LL);
      v18 = *(_QWORD *)(v14 + 16);
      *(_QWORD *)(v16 + 24) = v17;
      if ( (unsigned __int64)(v18 - (_QWORD)v17) <= 0xD )
      {
LABEL_15:
        sub_16E7EE0(v16, "not supported\n", 0xEu);
        sub_1C31880(a1);
        goto LABEL_16;
      }
    }
    qmemcpy(v17, "not supported\n", 14);
    *(_QWORD *)(v16 + 24) += 14LL;
    sub_1C31880(a1);
LABEL_16:
    v5 = **(_QWORD **)(a2 - 72);
    if ( *(_BYTE *)(v5 + 8) == 15 )
      goto LABEL_5;
LABEL_17:
    v19 = sub_1C321C0(a1, a2, 0);
    v20 = *(__m128i **)(v19 + 24);
    v21 = v19;
    if ( *(_QWORD *)(v19 + 16) - (_QWORD)v20 <= 0x25u )
    {
      v21 = sub_16E7EE0(v19, "cmpxchg pointer operand not a pointer?", 0x26u);
      v23 = *(_BYTE **)(v21 + 24);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42D0B20);
      v20[2].m128i_i32[0] = 1702129257;
      v20[2].m128i_i16[2] = 16242;
      *v20 = si128;
      v20[1] = _mm_load_si128((const __m128i *)&xmmword_42D0B30);
      v23 = (_BYTE *)(*(_QWORD *)(v19 + 24) + 38LL);
      *(_QWORD *)(v21 + 24) = v23;
    }
    if ( *(_BYTE **)(v21 + 16) == v23 )
    {
      sub_16E7EE0(v21, "\n", 1u);
    }
    else
    {
      *v23 = 10;
      ++*(_QWORD *)(v21 + 24);
    }
    goto LABEL_12;
  }
  v5 = **(_QWORD **)(a2 - 72);
  if ( *(_BYTE *)(v5 + 8) != 15 )
    goto LABEL_17;
LABEL_5:
  v6 = *(_DWORD *)(v5 + 8);
  if ( v6 > 0x1FF && v6 >> 8 != 3 )
  {
    v8 = sub_1C321C0(a1, a2, 0);
    v9 = *(__m128i **)(v8 + 24);
    v10 = v8;
    if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 0x2Du )
    {
      v24 = sub_16E7EE0(v8, "cmpxchg pointer operand must point to generic,", 0x2Eu);
      v12 = *(_QWORD *)(v24 + 24);
      v10 = v24;
    }
    else
    {
      v11 = _mm_load_si128((const __m128i *)&xmmword_42D0B20);
      qmemcpy(&v9[2], "nt to generic,", 14);
      *v9 = v11;
      v9[1] = _mm_load_si128((const __m128i *)&xmmword_42D0B40);
      v12 = *(_QWORD *)(v8 + 24) + 46LL;
      *(_QWORD *)(v8 + 24) = v12;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v10 + 16) - v12) <= 0x20 )
    {
      sub_16E7EE0(v10, " global, or shared address space\n", 0x21u);
    }
    else
    {
      v13 = _mm_load_si128((const __m128i *)&xmmword_42D0B50);
      *(_BYTE *)(v12 + 32) = 10;
      *(__m128i *)v12 = v13;
      *(__m128i *)(v12 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0B60);
      *(_QWORD *)(v10 + 24) += 33LL;
    }
LABEL_12:
    sub_1C31880(a1);
  }
  return sub_1C34B00(a1, a2);
}
