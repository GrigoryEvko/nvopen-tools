// Function: sub_33C8C10
// Address: 0x33c8c10
//
__int64 __fastcall sub_33C8C10(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  __m128i v7; // xmm0
  __int64 v9; // rax
  __int64 *v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // r8

  v6 = *(_DWORD *)(a5 + 24);
  if ( v6 == 35 || v6 == 11 )
  {
    v9 = *(_QWORD *)(a5 + 96);
    v10 = *(__int64 **)(v9 + 24);
    v11 = *(_DWORD *)(v9 + 32);
    if ( v11 <= 0x40 )
    {
      v12 = 0;
      if ( v11 )
        v12 = (__int64)((_QWORD)v10 << (64 - (unsigned __int8)v11)) >> (64 - (unsigned __int8)v11);
    }
    else
    {
      v12 = *v10;
    }
  }
  else
  {
    if ( v6 != 51 )
    {
      v7 = _mm_loadu_si128(a2);
      *(_QWORD *)(a1 + 16) = a2[1].m128i_i64[0];
      *(__m128i *)a1 = v7;
      return a1;
    }
    v12 = 0;
  }
  sub_33C8B50(a1, a2, a3, a4, v12);
  return a1;
}
