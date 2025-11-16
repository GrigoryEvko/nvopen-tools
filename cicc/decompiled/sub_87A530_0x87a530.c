// Function: sub_87A530
// Address: 0x87a530
//
__int64 __fastcall sub_87A530(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  char *v7; // r13
  size_t v8; // r15
  __int16 v9; // dx
  int v10; // ecx
  __m128i v11; // xmm3
  __int64 v13; // rdx
  __int64 v14; // [rsp+8h] [rbp-38h]

  v6 = *a1;
  v7 = (char *)qword_4F60028;
  v8 = *(_QWORD *)(*a1 + 16);
  if ( v8 + 1 > qword_4F60020 )
  {
    v13 = qword_4F60020 + 300;
    if ( v8 + 1 >= qword_4F60020 + 300 )
      v13 = v8 + 1;
    v14 = v13;
    qword_4F60028 = (void *)sub_822C60(qword_4F60028, qword_4F60020, v13, a4, a5, a6);
    v7 = (char *)qword_4F60028;
    v6 = *a1;
    qword_4F60020 = v14;
  }
  memcpy(v7 + 1, *(const void **)(v6 + 8), v8);
  *v7 = a2 == 0 ? 126 : 33;
  v9 = *((_WORD *)a1 + 6);
  v10 = *((_DWORD *)a1 + 2);
  *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
  *((__m128i *)a1 + 1) = _mm_loadu_si128(&xmmword_4F06660[1]);
  *((__m128i *)a1 + 2) = _mm_loadu_si128(&xmmword_4F06660[2]);
  v11 = _mm_loadu_si128(&xmmword_4F06660[3]);
  *((_BYTE *)a1 + 16) |= 0x20u;
  *((_WORD *)a1 + 6) = v9;
  *((_DWORD *)a1 + 2) = v10;
  *((__m128i *)a1 + 3) = v11;
  return sub_878540(v7, v8 + 1, a1);
}
