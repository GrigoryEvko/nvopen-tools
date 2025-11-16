// Function: sub_393EC10
// Address: 0x393ec10
//
__int64 __fastcall sub_393EC10(_QWORD *a1, unsigned __int64 *a2)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // rsi
  __int64 v6; // rdi
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // r9
  unsigned __int64 v10; // r10
  unsigned __int64 v11; // rdx
  unsigned int v12; // esi
  _QWORD *v14; // rax
  __m128i *v15; // rdx
  __int64 v16; // rdi
  __m128i si128; // xmm0
  __int64 v18; // rax
  _WORD *v19; // rdx
  __int64 v20; // rdi

  v4 = *a1;
  v5 = a1[1];
  v6 = *(_QWORD *)(*a1 + 8LL);
  v7 = v5 + 4;
  v8 = *(_QWORD *)(v4 + 16) - v6;
  if ( v8 < v5 + 4
    || (a1[1] = v7, v9 = *(_QWORD *)(v4 + 8), v10 = v5 + 8, v11 = *(_QWORD *)(v4 + 16) - v9, v11 < v5 + 8) )
  {
    v14 = sub_16E8CB0();
    v15 = (__m128i *)v14[3];
    v16 = (__int64)v14;
    if ( v14[2] - (_QWORD)v15 <= 0x20u )
    {
      v16 = sub_16E7EE0((__int64)v14, "Unexpected end of memory buffer: ", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4530950);
      v15[2].m128i_i8[0] = 32;
      *v15 = si128;
      v15[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
      v14[3] += 33LL;
    }
    v18 = sub_16E7A90(v16, a1[1] + 4LL);
    v19 = *(_WORD **)(v18 + 24);
    v20 = v18;
    if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 1u )
    {
      sub_16E7EE0(v18, ".\n", 2u);
      return 0;
    }
    else
    {
      *v19 = 2606;
      *(_QWORD *)(v20 + 24) += 2LL;
      return 0;
    }
  }
  else
  {
    if ( v8 > v5 )
      v8 = v5;
    if ( v7 > v11 )
      v7 = v11;
    v12 = *(_DWORD *)(v6 + v8);
    a1[1] = v10;
    *a2 = v12 | ((unsigned __int64)*(unsigned int *)(v9 + v7) << 32);
    return 1;
  }
}
