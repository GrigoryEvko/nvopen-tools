// Function: sub_87AA70
// Address: 0x87aa70
//
__int64 __fastcall sub_87AA70(__m128i *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r14
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  __int64 v10; // r12
  char *v11; // r15
  __int64 result; // rax

  v5 = sub_877070(a1, a2, a3, a4);
  v6 = 0;
  v7 = v5;
  v8 = ++qword_4F5FE58;
  do
  {
    v9 = v8;
    v10 = v6++;
    v8 /= 0xAu;
  }
  while ( v9 > 9 );
  v11 = (char *)sub_7279A0(v10 + 19);
  sprintf(v11, "<struct binding %lu>", qword_4F5FE58);
  *(_QWORD *)(v7 + 8) = v11;
  *(_QWORD *)(v7 + 16) = v10 + 18;
  *a1 = _mm_loadu_si128(xmmword_4F06660);
  a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
  result = *a2;
  a1->m128i_i64[0] = v7;
  a1->m128i_i64[1] = result;
  return result;
}
