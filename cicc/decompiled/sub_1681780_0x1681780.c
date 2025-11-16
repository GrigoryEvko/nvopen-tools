// Function: sub_1681780
// Address: 0x1681780
//
void __fastcall sub_1681780(_QWORD *a1, char *a2, size_t a3, const char **a4, __int64 a5)
{
  const char *v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rsi
  _QWORD *v11; // rdi
  __int64 v12; // rdx
  const char **v13; // r13
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rax
  __int64 v16; // r12
  _BYTE *v17; // rax
  __m128i *v18; // rdi
  unsigned __int64 v19; // rax
  __m128i si128; // xmm0
  __int64 v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v28; // [rsp+20h] [rbp-50h] BYREF
  size_t v29; // [rsp+28h] [rbp-48h]
  _QWORD v30[8]; // [rsp+30h] [rbp-40h] BYREF

  v8 = a2;
  if ( ((*a2 - 43) & 0xFD) != 0 )
  {
    v9 = (__int64)&a2[a3];
  }
  else if ( a3 )
  {
    ++a2;
    v9 = (__int64)&v8[a3];
  }
  else
  {
    v9 = (__int64)a2;
  }
  v28 = v30;
  sub_1680AA0((__int64 *)&v28, a2, v9);
  v10 = v29;
  v11 = v28;
  v13 = sub_1680B50(v28, v29, a4, a5);
  if ( v11 != v30 )
  {
    v10 = v30[0] + 1LL;
    j_j___libc_free_0(v11, v30[0] + 1LL);
  }
  if ( v13 )
  {
    if ( *v8 == 43 )
    {
      *a1 |= (unsigned __int64)v13[2];
      a1[1] |= (unsigned __int64)v13[3];
      a1[2] |= (unsigned __int64)v13[4];
      sub_16809A0(a1, (__int64)v13, a4, a5);
    }
    else
    {
      v14 = (unsigned __int64)v13[2];
      v15 = ~(unsigned __int64)v13[4];
      a1[1] &= ~(unsigned __int64)v13[3];
      a1[2] &= v15;
      *a1 &= ~v14;
      sub_1680CA0(a1, v13, (__int64)a4, a5);
    }
  }
  else
  {
    v16 = sub_16E8CB0(v11, v10, v12);
    v17 = *(_BYTE **)(v16 + 24);
    if ( *(_BYTE **)(v16 + 16) == v17 )
    {
      v25 = sub_16E7EE0(v16, "'", 1);
      v18 = *(__m128i **)(v25 + 24);
      v16 = v25;
    }
    else
    {
      *v17 = 39;
      v18 = (__m128i *)(*(_QWORD *)(v16 + 24) + 1LL);
      *(_QWORD *)(v16 + 24) = v18;
    }
    v19 = *(_QWORD *)(v16 + 16) - (_QWORD)v18;
    if ( v19 < a3 )
    {
      v24 = sub_16E7EE0(v16, v8, a3);
      v18 = *(__m128i **)(v24 + 24);
      v16 = v24;
      v19 = *(_QWORD *)(v24 + 16) - (_QWORD)v18;
    }
    else if ( a3 )
    {
      memcpy(v18, v8, a3);
      v26 = *(_QWORD *)(v16 + 16);
      v18 = (__m128i *)(a3 + *(_QWORD *)(v16 + 24));
      *(_QWORD *)(v16 + 24) = v18;
      v19 = v26 - (_QWORD)v18;
    }
    if ( v19 <= 0x2C )
    {
      v23 = sub_16E7EE0(v16, "' is not a recognized feature for this target", 45);
      v21 = *(_QWORD *)(v23 + 24);
      v16 = v23;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F82940);
      qmemcpy(&v18[2], "r this target", 13);
      *v18 = si128;
      v18[1] = _mm_load_si128((const __m128i *)&xmmword_3F82950);
      v21 = *(_QWORD *)(v16 + 24) + 45LL;
      *(_QWORD *)(v16 + 24) = v21;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v16 + 16) - v21) <= 0x13 )
    {
      sub_16E7EE0(v16, " (ignoring feature)\n", 20);
    }
    else
    {
      v22 = _mm_load_si128((const __m128i *)&xmmword_3F82960);
      *(_DWORD *)(v21 + 16) = 170485106;
      *(__m128i *)v21 = v22;
      *(_QWORD *)(v16 + 24) += 20LL;
    }
  }
}
