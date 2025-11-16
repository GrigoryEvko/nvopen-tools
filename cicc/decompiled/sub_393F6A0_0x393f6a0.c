// Function: sub_393F6A0
// Address: 0x393f6a0
//
__int64 __fastcall sub_393F6A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // r12d
  _QWORD *v19; // rax
  __m128i *v20; // rdx
  __int64 v21; // rdi
  __m128i si128; // xmm0
  char *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  _WORD *v28; // rdx
  __int64 v29; // rdi

  v7 = *(_QWORD *)(a1 + 72);
  v8 = *(_QWORD *)(a1 + 80);
  v9 = *(_QWORD *)(v7 + 8);
  v10 = v8 + 4;
  v11 = *(_QWORD *)(v7 + 16) - v9;
  if ( v11 < v8 + 4 )
  {
    v19 = sub_16E8CB0();
    v20 = (__m128i *)v19[3];
    v21 = (__int64)v19;
    if ( v19[2] - (_QWORD)v20 <= 0x20u )
    {
      v21 = sub_16E7EE0((__int64)v19, "Unexpected end of memory buffer: ", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4530950);
      v20[2].m128i_i8[0] = 32;
      *v20 = si128;
      v20[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
      v19[3] += 33LL;
    }
    v23 = (char *)(*(_QWORD *)(a1 + 80) + 4LL);
    v24 = sub_16E7A90(v21, (__int64)v23);
    v28 = *(_WORD **)(v24 + 24);
    v29 = v24;
    if ( *(_QWORD *)(v24 + 16) - (_QWORD)v28 <= 1u )
    {
      v23 = ".\n";
      sub_16E7EE0(v24, ".\n", 2u);
    }
    else
    {
      *v28 = 2606;
      *(_QWORD *)(v24 + 24) += 2LL;
    }
    sub_393D180(v29, (__int64)v23, (__int64)v28, v25, v26, v27);
    return 4;
  }
  else
  {
    *(_QWORD *)(a1 + 80) = v10;
    if ( v11 > v8 )
      v11 = v8;
    if ( (_DWORD)a2 == *(_DWORD *)(v9 + v11) )
    {
      v13 = a1;
      v18 = sub_393F200(a1, a2, v8, v9, a5, a6);
      if ( !v18 )
        sub_393D180(v13, a2, v14, v15, v16, v17);
      return v18;
    }
    else
    {
      sub_393D180(v10, a2, v8, v9, a5, a6);
      return 5;
    }
  }
}
