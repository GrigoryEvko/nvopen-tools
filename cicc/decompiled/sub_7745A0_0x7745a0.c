// Function: sub_7745A0
// Address: 0x7745a0
//
unsigned __int64 __fastcall sub_7745A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r8
  unsigned int v3; // ecx
  __int64 v4; // rsi
  __int64 v5; // r9
  unsigned int v6; // edx
  __m128i *v7; // rax
  __m128i v8; // xmm0
  __m128i *v9; // rax
  int v10; // eax
  unsigned __int64 result; // rax
  int v12; // eax

  v2 = a2 - 1;
  v3 = *(_DWORD *)(a1 + 8);
  v4 = *(_QWORD *)a1;
  v5 = a1 + 144;
  v6 = v3 & (v2 >> 3);
  v7 = (__m128i *)(*(_QWORD *)a1 + 16LL * v6);
  if ( !v7->m128i_i64[0] )
  {
    v7->m128i_i64[0] = v2;
    v7->m128i_i64[1] = v5;
    v12 = *(_DWORD *)(a1 + 12) + 1;
    *(_DWORD *)(a1 + 12) = v12;
    result = (unsigned int)(2 * v12);
    if ( (unsigned int)result <= v3 )
      return result;
    return (unsigned __int64)sub_7704A0(a1);
  }
  v8 = _mm_loadu_si128(v7);
  v7->m128i_i64[0] = v2;
  v7->m128i_i64[1] = v5;
  do
  {
    v6 = v3 & (v6 + 1);
    v9 = (__m128i *)(v4 + 16LL * v6);
  }
  while ( v9->m128i_i64[0] );
  *v9 = v8;
  v10 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v10;
  result = (unsigned int)(2 * v10);
  if ( (unsigned int)result > v3 )
    return (unsigned __int64)sub_7704A0(a1);
  return result;
}
