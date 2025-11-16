// Function: sub_7C9F50
// Address: 0x7c9f50
//
__int64 __fastcall sub_7C9F50(unsigned __int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 **v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 result; // rax
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __m128i v12; // xmm0
  __m128i v13; // xmm4
  __m128i v14; // xmm6
  __m128i v15; // xmm0

  v2 = (_QWORD *)a1;
  v3 = (__int64 **)a2;
  *(_QWORD *)a2 = 0;
  v4 = qword_4F061C8;
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  ++*(_BYTE *)(v4 + 63);
  if ( word_4F06418[0] != 55 )
  {
    a2 = 40;
    a1 = 1;
    if ( !sub_7BE5B0(1u, 0x28u, 0, 0) )
    {
      v10 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v11 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v12 = _mm_loadu_si128(&xmmword_4F06660[3]);
      qword_4D04A00 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      *(__m128i *)&word_4D04A10 = v10;
      xmmword_4D04A20 = v11;
      HIBYTE(word_4D04A10) = v10.m128i_i8[1] | 0x20;
      qword_4D04A08 = *(_QWORD *)dword_4F07508;
      unk_4D04A30 = v12;
    }
  }
  *v2 = sub_7BE640();
  result = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 63LL);
  if ( word_4F06418[0] == 55 )
  {
    sub_7B8B50(a1, (unsigned int *)a2, v5, v6, v7, v8);
    if ( !sub_7BE5B0(1u, 0x28u, 0, 0) )
    {
      v13 = _mm_loadu_si128(xmmword_4F06660);
      v14 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v15 = _mm_loadu_si128(&xmmword_4F06660[3]);
      *(__m128i *)&word_4D04A10 = _mm_loadu_si128(&xmmword_4F06660[1]);
      *(__m128i *)&qword_4D04A00 = v13;
      HIBYTE(word_4D04A10) |= 0x20u;
      xmmword_4D04A20 = v14;
      qword_4D04A08 = *(_QWORD *)dword_4F07508;
      unk_4D04A30 = v15;
    }
    *v3 = sub_7BE640();
    result = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 83LL);
  }
  else
  {
    --*(_BYTE *)(result + 83);
  }
  return result;
}
