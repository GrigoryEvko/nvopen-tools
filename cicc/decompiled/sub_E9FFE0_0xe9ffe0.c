// Function: sub_E9FFE0
// Address: 0xe9ffe0
//
unsigned __int64 __fastcall sub_E9FFE0(__int64 a1, char *a2, size_t a3, const char **a4, __int64 a5)
{
  char v10; // r8
  size_t v11; // rsi
  char *v12; // rdi
  const char **v13; // rax
  unsigned __int64 result; // rax
  _QWORD *v15; // r12
  _BYTE *v16; // rax
  __m128i *v17; // rdi
  unsigned __int64 v18; // rax
  __m128i si128; // xmm0
  __int64 v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __m128i *v26; // rbx
  char v27; // [rsp+Fh] [rbp-31h]

  v10 = *a2;
  if ( ((*a2 - 43) & 0xFD) != 0 )
  {
    v13 = sub_E9F950(a2, a3, a4, a5);
    if ( v13 )
    {
LABEL_8:
      *(_QWORD *)(a1 + 8LL * (*((_DWORD *)v13 + 4) >> 6)) &= ~(1LL << *((_DWORD *)v13 + 4));
      return sub_E9F5A0(a1, *((_DWORD *)v13 + 4), (__int64)a4, a5);
    }
  }
  else
  {
    if ( a3 )
    {
      v11 = a3 - 1;
      v12 = a2 + 1;
    }
    else
    {
      v12 = a2;
      v11 = 0;
    }
    v27 = v10;
    v13 = sub_E9F950(v12, v11, a4, a5);
    if ( v13 )
    {
      if ( v27 == 43 )
      {
        *(_QWORD *)(a1 + 8LL * (*((_DWORD *)v13 + 4) >> 6)) |= 1LL << *((_DWORD *)v13 + 4);
        return sub_E9F500(a1, (__int64)(v13 + 3), (__int64)a4, a5);
      }
      goto LABEL_8;
    }
  }
  v15 = sub_CB72A0();
  v16 = (_BYTE *)v15[4];
  if ( (_BYTE *)v15[3] == v16 )
  {
    v24 = sub_CB6200((__int64)v15, (unsigned __int8 *)"'", 1u);
    v17 = *(__m128i **)(v24 + 32);
    v15 = (_QWORD *)v24;
  }
  else
  {
    *v16 = 39;
    v17 = (__m128i *)(v15[4] + 1LL);
    v15[4] = v17;
  }
  v18 = v15[3] - (_QWORD)v17;
  if ( v18 < a3 )
  {
    v23 = sub_CB6200((__int64)v15, (unsigned __int8 *)a2, a3);
    v17 = *(__m128i **)(v23 + 32);
    v15 = (_QWORD *)v23;
    v18 = *(_QWORD *)(v23 + 24) - (_QWORD)v17;
  }
  else if ( a3 )
  {
    memcpy(v17, a2, a3);
    v25 = v15[3];
    v26 = (__m128i *)(v15[4] + a3);
    v15[4] = v26;
    v17 = v26;
    v18 = v25 - (_QWORD)v26;
  }
  if ( v18 <= 0x2C )
  {
    v22 = sub_CB6200((__int64)v15, "' is not a recognized feature for this target", 0x2Du);
    v20 = *(_QWORD *)(v22 + 32);
    v15 = (_QWORD *)v22;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F82940);
    qmemcpy(&v17[2], "r this target", 13);
    *v17 = si128;
    v17[1] = _mm_load_si128((const __m128i *)&xmmword_3F82950);
    v20 = v15[4] + 45LL;
    v15[4] = v20;
  }
  result = v15[3] - v20;
  if ( result <= 0x13 )
    return sub_CB6200((__int64)v15, " (ignoring feature)\n", 0x14u);
  v21 = _mm_load_si128((const __m128i *)&xmmword_3F82960);
  *(_DWORD *)(v20 + 16) = 170485106;
  *(__m128i *)v20 = v21;
  v15[4] += 20LL;
  return result;
}
