// Function: sub_EA0AA0
// Address: 0xea0aa0
//
const char *__fastcall sub_EA0AA0(__int64 a1, _DWORD *a2, size_t a3)
{
  const char **v4; // rax
  _QWORD *v6; // r14
  _BYTE *v7; // rax
  __m128i *v8; // rdi
  unsigned __int64 v9; // rax
  __m128i si128; // xmm0
  __int64 v11; // rdx
  __m128i v12; // xmm0
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax

  v4 = sub_E9F8B0(a2, a3, *(const char ***)(a1 + 160), *(_QWORD *)(a1 + 168));
  if ( v4 )
    return v4[11];
  if ( a3 != 4 || *a2 != 1886152040 )
  {
    v6 = sub_CB72A0();
    v7 = (_BYTE *)v6[4];
    if ( (_BYTE *)v6[3] == v7 )
    {
      v13 = sub_CB6200((__int64)v6, (unsigned __int8 *)"'", 1u);
      v8 = *(__m128i **)(v13 + 32);
      v6 = (_QWORD *)v13;
    }
    else
    {
      *v7 = 39;
      v8 = (__m128i *)(v6[4] + 1LL);
      v6[4] = v8;
    }
    v9 = v6[3] - (_QWORD)v8;
    if ( a3 > v9 )
    {
      v15 = sub_CB6200((__int64)v6, (unsigned __int8 *)a2, a3);
      v8 = *(__m128i **)(v15 + 32);
      v6 = (_QWORD *)v15;
      v9 = *(_QWORD *)(v15 + 24) - (_QWORD)v8;
    }
    else if ( a3 )
    {
      memcpy(v8, a2, a3);
      v16 = v6[3];
      v8 = (__m128i *)(a3 + v6[4]);
      v6[4] = v8;
      v9 = v16 - (_QWORD)v8;
    }
    if ( v9 <= 0x2E )
    {
      v14 = sub_CB6200((__int64)v6, "' is not a recognized processor for this target", 0x2Fu);
      v11 = *(_QWORD *)(v14 + 32);
      v6 = (_QWORD *)v14;
    }
    else
    {
      *v8 = _mm_load_si128((const __m128i *)&xmmword_3F82940);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F82970);
      qmemcpy(&v8[2], "for this target", 15);
      v8[1] = si128;
      v11 = v6[4] + 47LL;
      v6[4] = v11;
    }
    if ( (unsigned __int64)(v6[3] - v11) <= 0x15 )
    {
      sub_CB6200((__int64)v6, (unsigned __int8 *)" (ignoring processor)\n", 0x16u);
    }
    else
    {
      v12 = _mm_load_si128((const __m128i *)&xmmword_3F82980);
      *(_DWORD *)(v11 + 16) = 1919906675;
      *(_WORD *)(v11 + 20) = 2601;
      *(__m128i *)v11 = v12;
      v6[4] += 22LL;
    }
  }
  return (const char *)&unk_3F8F0C0;
}
