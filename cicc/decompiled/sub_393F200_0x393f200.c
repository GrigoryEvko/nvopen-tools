// Function: sub_393F200
// Address: 0x393f200
//
__int64 __fastcall sub_393F200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD *v11; // rax
  __m128i *v12; // rdx
  __int64 v13; // rdi
  __m128i si128; // xmm0
  char *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _WORD *v20; // rdx
  __int64 v21; // rdi

  v7 = *(_QWORD *)(a1 + 72);
  v8 = *(_QWORD *)(a1 + 80) + 4LL;
  v9 = *(_QWORD *)(v7 + 16) - *(_QWORD *)(v7 + 8);
  if ( v9 < v8 )
  {
    v11 = sub_16E8CB0();
    v12 = (__m128i *)v11[3];
    v13 = (__int64)v11;
    if ( v11[2] - (_QWORD)v12 <= 0x20u )
    {
      v13 = sub_16E7EE0((__int64)v11, "Unexpected end of memory buffer: ", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4530950);
      v12[2].m128i_i8[0] = 32;
      *v12 = si128;
      v12[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
      v11[3] += 33LL;
    }
    v15 = (char *)(*(_QWORD *)(a1 + 80) + 4LL);
    v16 = sub_16E7A90(v13, (__int64)v15);
    v20 = *(_WORD **)(v16 + 24);
    v21 = v16;
    if ( *(_QWORD *)(v16 + 16) - (_QWORD)v20 <= 1u )
    {
      v15 = ".\n";
      sub_16E7EE0(v16, ".\n", 2u);
    }
    else
    {
      *v20 = 2606;
      *(_QWORD *)(v16 + 24) += 2LL;
    }
    sub_393D180(v21, (__int64)v15, (__int64)v20, v17, v18, v19);
    return 4;
  }
  else
  {
    *(_QWORD *)(a1 + 80) = v8;
    sub_393D180(a1, a2, v9, v7, a5, a6);
    return 0;
  }
}
