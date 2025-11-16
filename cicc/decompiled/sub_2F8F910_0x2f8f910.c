// Function: sub_2F8F910
// Address: 0x2f8f910
//
void __fastcall sub_2F8F910(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r15
  __m128i *v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  __m128i *v10; // r13
  unsigned int v11; // eax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r12
  __m128i *v15; // rax
  __m128i v16; // xmm0
  unsigned int v17; // [rsp-3Ch] [rbp-3Ch]

  if ( *(_DWORD *)(a1 + 208) <= 1u )
    return;
  v6 = *(__m128i **)(a1 + 40);
  v7 = v6 + 1;
  v8 = v6->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_BYTE *)(v8 + 254) & 1) == 0 )
  {
    sub_2F8F5D0(v6->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, a2, a3, a4, a5, a6);
    v15 = *(__m128i **)(a1 + 40);
    v9 = *(unsigned int *)(v8 + 240);
    v10 = &v15[*(unsigned int *)(a1 + 48)];
    if ( v10 == v7 )
    {
LABEL_15:
      if ( v6 != v15 )
      {
        v16 = _mm_loadu_si128(v15);
        *v15 = _mm_loadu_si128(v6);
        *v6 = v16;
      }
      return;
    }
    while ( 1 )
    {
LABEL_8:
      if ( (v7->m128i_i64[0] & 6) != 0 )
        goto LABEL_7;
      v12 = v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
      v13 = v12;
      if ( (*(_BYTE *)(v12 + 254) & 1) == 0 )
        break;
      v11 = *(_DWORD *)(v12 + 240);
      if ( v11 > (unsigned int)v9 )
        goto LABEL_6;
LABEL_7:
      if ( ++v7 == v10 )
      {
        v15 = *(__m128i **)(a1 + 40);
        goto LABEL_15;
      }
    }
    v17 = v9;
    sub_2F8F5D0(v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, a2, v9, a4, a5, a6);
    v9 = v17;
    if ( *(_DWORD *)(v13 + 240) <= v17 )
      goto LABEL_7;
    v14 = v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_BYTE *)(v14 + 254) & 1) == 0 )
      sub_2F8F5D0(v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, a2, v17, a4, a5, a6);
    v11 = *(_DWORD *)(v14 + 240);
LABEL_6:
    v9 = v11;
    v6 = v7;
    goto LABEL_7;
  }
  v9 = *(unsigned int *)(v8 + 240);
  v10 = &v6[*(unsigned int *)(a1 + 48)];
  if ( v10 != v7 )
    goto LABEL_8;
}
