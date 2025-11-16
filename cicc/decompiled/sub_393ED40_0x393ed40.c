// Function: sub_393ED40
// Address: 0x393ed40
//
__int64 __fastcall sub_393ED40(__int64 *a1, __int64 *a2)
{
  __int64 v4; // rdi
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  __m128i *v11; // rdx
  __int64 v12; // rdi
  __m128i si128; // xmm0
  __int64 v14; // rsi
  __int64 v15; // rax
  _WORD *v16; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 v20; // r13
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  _QWORD *v26; // rax
  __m128i *v27; // rdx
  __m128i v28; // xmm0
  char v29; // [rsp+Fh] [rbp-31h] BYREF
  __int64 v30; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v31; // [rsp+18h] [rbp-28h]

  v4 = *a1;
  v5 = a1[1];
  do
  {
    v8 = *(_QWORD *)(v4 + 8);
    v9 = v5;
    v5 += 4LL;
    v6 = *(_QWORD *)(v4 + 16) - v8;
    if ( v6 < v5 )
    {
      v10 = sub_16E8CB0();
      v11 = (__m128i *)v10[3];
      v12 = (__int64)v10;
      if ( v10[2] - (_QWORD)v11 <= 0x20u )
      {
        v12 = sub_16E7EE0((__int64)v10, "Unexpected end of memory buffer: ", 0x21u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4530950);
        v11[2].m128i_i8[0] = 32;
        *v11 = si128;
        v11[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
        v10[3] += 33LL;
      }
      v14 = a1[1] + 4;
LABEL_9:
      v15 = sub_16E7A90(v12, v14);
      v16 = *(_WORD **)(v15 + 24);
      if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 1u )
      {
        sub_16E7EE0(v15, ".\n", 2u);
      }
      else
      {
        *v16 = 2606;
        *(_QWORD *)(v15 + 24) += 2LL;
      }
      return 0;
    }
    a1[1] = v5;
    if ( v6 > v9 )
      v6 = v9;
    v7 = *(_DWORD *)(v8 + v6);
  }
  while ( !v7 );
  v18 = *(_QWORD *)(v4 + 16);
  v19 = *(_QWORD *)(v4 + 8);
  v20 = (unsigned int)(4 * v7);
  v21 = v18 - v19;
  if ( v20 + v5 > v18 - v19 )
  {
    v26 = sub_16E8CB0();
    v27 = (__m128i *)v26[3];
    v12 = (__int64)v26;
    if ( v26[2] - (_QWORD)v27 <= 0x20u )
    {
      v12 = sub_16E7EE0((__int64)v26, "Unexpected end of memory buffer: ", 0x21u);
    }
    else
    {
      v28 = _mm_load_si128((const __m128i *)&xmmword_4530950);
      v27[2].m128i_i8[0] = 32;
      *v27 = v28;
      v27[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
      v26[3] += 33LL;
    }
    v14 = v20 + a1[1];
    goto LABEL_9;
  }
  v22 = 0;
  if ( v21 >= v5 )
  {
    v18 = v19 + v5;
    v22 = v20;
    if ( v20 + v5 <= v5 )
    {
      v25 = v21;
      if ( v21 > v5 )
        v25 = v5;
      v22 = v25 - v5;
    }
  }
  v30 = v18;
  v31 = v22;
  v29 = 0;
  v23 = sub_16D20C0(&v30, &v29, 1u, 0);
  if ( v23 == -1 )
  {
    v24 = v30;
    v23 = v31;
  }
  else
  {
    v24 = v30;
    if ( v23 && v23 > v31 )
      v23 = v31;
  }
  a2[1] = v23;
  *a2 = v24;
  a1[1] += v20;
  return 1;
}
