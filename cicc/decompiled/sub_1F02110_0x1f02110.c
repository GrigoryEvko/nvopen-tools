// Function: sub_1F02110
// Address: 0x1f02110
//
void __fastcall sub_1F02110(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __m128i *v6; // r13
  __m128i *v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  __m128i *v10; // r14
  unsigned __int64 v11; // r12
  __m128i *v12; // rax
  __m128i v13; // xmm0
  unsigned int v14; // [rsp-3Ch] [rbp-3Ch]

  if ( *(_DWORD *)(a1 + 200) > 1u )
  {
    v6 = *(__m128i **)(a1 + 32);
    v7 = v6 + 1;
    v8 = v6->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_BYTE *)(v8 + 236) & 1) != 0 )
    {
      v9 = *(unsigned int *)(v8 + 240);
      v10 = &v6[*(unsigned int *)(a1 + 40)];
      if ( v7 == v10 )
        return;
    }
    else
    {
      sub_1F01DD0(v6->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, a2, a3, a4, a5, a6);
      v12 = *(__m128i **)(a1 + 32);
      v9 = *(unsigned int *)(v8 + 240);
      v10 = &v12[*(unsigned int *)(a1 + 40)];
      if ( v10 == v7 )
      {
LABEL_12:
        if ( v12 != v6 )
        {
          v13 = _mm_loadu_si128(v12);
          *v12 = _mm_loadu_si128(v6);
          *v6 = v13;
        }
        return;
      }
    }
    do
    {
      if ( (v7->m128i_i64[0] & 6) == 0 )
      {
        v11 = v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v11 + 236) & 1) == 0 )
        {
          v14 = v9;
          sub_1F01DD0(v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, a2, v9, a4, a5, a6);
          v9 = v14;
        }
        if ( *(_DWORD *)(v11 + 240) > (unsigned int)v9 )
          v6 = v7;
      }
      ++v7;
    }
    while ( v7 != v10 );
    v12 = *(__m128i **)(a1 + 32);
    goto LABEL_12;
  }
}
