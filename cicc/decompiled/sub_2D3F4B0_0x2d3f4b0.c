// Function: sub_2D3F4B0
// Address: 0x2d3f4b0
//
__int64 __fastcall sub_2D3F4B0(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  __m128i *v4; // r14
  char v5; // r12
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r12
  const __m128i *v12; // r15
  __m128i *v13; // r12
  __m128i *v14; // rdi
  __m128i *v15; // rax
  __int64 v16; // rax
  const __m128i *v17; // [rsp+8h] [rbp-1B8h]
  __m128i v18[27]; // [rsp+10h] [rbp-1B0h] BYREF

  v3 = a2;
  v4 = *(__m128i **)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v12 = (const __m128i *)(a1 + 16);
    v17 = (const __m128i *)(a1 + 400);
  }
  else
  {
    v6 = sub_AF1560(a2 - 1);
    v3 = v6;
    if ( v6 > 0x40 )
    {
      v12 = (const __m128i *)(a1 + 16);
      v17 = (const __m128i *)(a1 + 400);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 96LL * v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 6144;
        v3 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v3;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 6LL * v7;
        sub_2D3F240(a1, v4, &v4[v10]);
        return sub_C7D6A0((__int64)v4, v10 * 16, 8);
      }
      v12 = (const __m128i *)(a1 + 16);
      v3 = 64;
      v17 = (const __m128i *)(a1 + 400);
    }
  }
  v13 = v18;
  do
  {
    if ( v12->m128i_i64[0] == -4096 )
    {
      if ( v12->m128i_i64[1] == -4096 )
        goto LABEL_21;
    }
    else if ( v12->m128i_i64[0] == -8192 && v12->m128i_i64[1] == -8192 )
    {
      goto LABEL_21;
    }
    if ( v13 )
      *v13 = _mm_loadu_si128(v12);
    v13[1].m128i_i64[0] = 0;
    v14 = v13 + 1;
    v15 = v13 + 2;
    v13 += 6;
    v13[-5].m128i_i64[1] = 1;
    do
    {
      if ( v15 )
      {
        v15->m128i_i64[0] = -1;
        v15->m128i_i64[1] = -1;
      }
      ++v15;
    }
    while ( v15 != v13 );
    sub_2D3F0A0(v14, (__m128i *)&v12[1]);
    if ( (v12[1].m128i_i8[8] & 1) == 0 )
      sub_C7D6A0(v12[2].m128i_i64[0], 16LL * v12[2].m128i_u32[2], 8);
LABEL_21:
    v12 += 6;
  }
  while ( v12 != v17 );
  if ( v3 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v16 = sub_C7D670(96LL * v3, 8);
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = v16;
  }
  return sub_2D3F240(a1, v18, v13);
}
