// Function: sub_87F5F0
// Address: 0x87f5f0
//
_QWORD *__fastcall sub_87F5F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rdx
  __int64 v5; // rsi
  _QWORD *result; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __m128i si128; // xmm0

  v4 = a1;
  v5 = qword_4F600E8;
  if ( !qword_4F600E8 )
  {
    v7 = sub_877070(a1, 0, a1, a4);
    qword_4F600E8 = v7;
    v8 = sub_7279A0(21);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3C1F3B0);
    v4 = a1;
    strcpy((char *)(v8 + 16), "ect>");
    *(__m128i *)v8 = si128;
    v5 = qword_4F600E8;
    *(_QWORD *)(v7 + 8) = v8;
    *(_BYTE *)(v7 + 73) |= 1u;
    *(_QWORD *)(v7 + 16) = 20;
  }
  result = sub_87EBB0(7u, v5, v4);
  *((_DWORD *)result + 10) = *(_DWORD *)qword_4F04C68[0];
  return result;
}
