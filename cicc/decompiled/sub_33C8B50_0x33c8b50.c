// Function: sub_33C8B50
// Address: 0x33c8b50
//
__int64 __fastcall sub_33C8B50(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  __m128i v7; // xmm0
  unsigned int v9; // r9d
  __int64 *v10; // rcx
  __int64 v11; // rdi
  int v12; // eax
  __int64 v13; // rcx
  int v14; // eax
  __int64 v15; // rcx
  unsigned int v16; // esi
  _QWORD *v17; // rax

  v6 = *(_DWORD *)(a4 + 24);
  if ( v6 == 15 || v6 == 39 )
  {
    v9 = *(_DWORD *)(a4 + 96);
  }
  else
  {
    if ( v6 != 56
      || (v10 = *(__int64 **)(a4 + 40), v11 = v10[5], v12 = *(_DWORD *)(v11 + 24), v12 != 35) && v12 != 11
      || (v13 = *v10, v14 = *(_DWORD *)(v13 + 24), v14 != 15) && v14 != 39 )
    {
      v7 = _mm_loadu_si128(a2);
      *(_QWORD *)(a1 + 16) = a2[1].m128i_i64[0];
      *(__m128i *)a1 = v7;
      return a1;
    }
    v9 = *(_DWORD *)(v13 + 96);
    v15 = *(_QWORD *)(v11 + 96);
    v16 = *(_DWORD *)(v15 + 32);
    v17 = *(_QWORD **)(v15 + 24);
    if ( v16 > 0x40 )
    {
      a5 += *v17;
    }
    else if ( v16 )
    {
      a5 += (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
    }
  }
  sub_2EAC300(a1, *(_QWORD *)(a3 + 40), v9, a5);
  return a1;
}
