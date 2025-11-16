// Function: sub_1EB6BB0
// Address: 0x1eb6bb0
//
__int64 __fastcall sub_1EB6BB0(__int64 a1, __m128i *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  unsigned int v4; // eax
  const __m128i *v5; // rdx

  result = a2->m128i_u16[6];
  if ( a2->m128i_i64[0] )
  {
    v3 = *(_QWORD *)(a2->m128i_i64[0] + 32) + 40LL * a2->m128i_u16[7];
    if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0
      && (*(_BYTE *)v3 || (*(_WORD *)(v3 + 2) & 0xFF0) == 0)
      && *(_DWORD *)(v3 + 8) == (unsigned __int16)result )
    {
      *(_BYTE *)(v3 + 3) |= 0x40u;
      result = a2->m128i_u16[6];
    }
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 648) + 4 * result) = 1;
  if ( !*(_BYTE *)(a1 + 1088) )
  {
    v4 = *(_DWORD *)(a1 + 400);
    v5 = (const __m128i *)(*(_QWORD *)(a1 + 392) + 24LL * v4 - 24);
    if ( a2 != v5 )
    {
      *a2 = _mm_loadu_si128(v5);
      a2[1].m128i_i8[0] = v5[1].m128i_i8[0];
      *(_BYTE *)(*(_QWORD *)(a1 + 600)
               + (*(_DWORD *)(*(_QWORD *)(a1 + 392) + 24LL * *(unsigned int *)(a1 + 400) - 16) & 0x7FFFFFFF)) = -85 * (((__int64)a2->m128i_i64 - *(_QWORD *)(a1 + 392)) >> 3);
      v4 = *(_DWORD *)(a1 + 400);
    }
    result = v4 - 1;
    *(_DWORD *)(a1 + 400) = result;
  }
  return result;
}
