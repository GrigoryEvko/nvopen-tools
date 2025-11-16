// Function: sub_87AB50
// Address: 0x87ab50
//
__int64 __fastcall sub_87AB50(__int64 a1, __m128i *a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __m128i v6; // xmm3
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // r13
  __int64 v10; // rax

  v4 = *(unsigned __int8 *)(a1 + 140);
  if ( (_BYTE)v4 == 12 )
  {
    v5 = a1;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v4 = *(unsigned __int8 *)(v5 + 140);
    }
    while ( (_BYTE)v4 == 12 );
  }
  if ( (_BYTE)v4 )
  {
    *a2 = _mm_loadu_si128(xmmword_4F06660);
    a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    a2[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    a2->m128i_i64[1] = *a3;
    result = sub_877120(a1);
  }
  else
  {
    *a2 = _mm_loadu_si128(xmmword_4F06660);
    a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v6 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v7 = *(_QWORD *)dword_4F07508;
    a2[1].m128i_i8[1] |= 0x20u;
    a2[3] = v6;
    a2->m128i_i64[1] = v7;
    result = qword_4F600F0;
    if ( !qword_4F600F0 )
    {
      v9 = sub_877070(a1, a2, v4, a3);
      qword_4F600F0 = v9;
      v10 = sub_7279A0(8);
      *(_DWORD *)v10 = 1920099644;
      *(_WORD *)(v10 + 4) = 29295;
      *(_BYTE *)(v10 + 6) = 62;
      *(_BYTE *)(v10 + 7) = 0;
      *(_QWORD *)(v9 + 8) = v10;
      result = qword_4F600F0;
      *(_BYTE *)(v9 + 73) &= ~1u;
      *(_QWORD *)(v9 + 16) = 7;
    }
  }
  a2[3].m128i_i64[1] = a1;
  a2[1].m128i_i8[0] |= 0x10u;
  a2->m128i_i64[0] = result;
  return result;
}
