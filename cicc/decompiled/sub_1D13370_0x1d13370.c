// Function: sub_1D13370
// Address: 0x1d13370
//
__m128i *__fastcall sub_1D13370(__m128i *a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  __m128i v7; // xmm0
  __int64 *v9; // rcx
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 *v16; // rax
  __int64 v17; // rax

  v6 = *(unsigned __int16 *)(a4 + 24);
  if ( v6 == 14 || v6 == 36 )
  {
    sub_1E341E0(a1, *(_QWORD *)(a3 + 32), *(unsigned int *)(a4 + 84), a5);
  }
  else
  {
    if ( v6 != 52
      || (v9 = *(__int64 **)(a4 + 32), v10 = v9[5], v11 = *(unsigned __int16 *)(v10 + 24), v11 != 10) && v11 != 32
      || (v12 = *v9, v13 = *(unsigned __int16 *)(v12 + 24), v13 != 14) && v13 != 36 )
    {
      v7 = _mm_loadu_si128(a2);
      a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
      *a1 = v7;
      return a1;
    }
    v14 = *(_QWORD *)(v10 + 88);
    v15 = *(_DWORD *)(v14 + 32);
    v16 = *(__int64 **)(v14 + 24);
    if ( v15 > 0x40 )
      v17 = *v16;
    else
      v17 = (__int64)((_QWORD)v16 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
    sub_1E341E0(a1, *(_QWORD *)(a3 + 32), *(unsigned int *)(v12 + 84), a5 + v17);
  }
  return a1;
}
