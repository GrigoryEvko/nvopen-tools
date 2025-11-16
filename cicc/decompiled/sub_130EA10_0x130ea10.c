// Function: sub_130EA10
// Address: 0x130ea10
//
__int64 __fastcall sub_130EA10(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, const __m128i *a5)
{
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  char v10; // cl
  unsigned int v11; // eax
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r13
  unsigned __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v25; // [rsp+14h] [rbp-3Ch]

  v7 = a5->m128i_i64[1] & 0xFFFFFFFFFFFFF000LL;
  if ( v7 > 0x7000000000000000LL )
  {
    v25 = 200;
    v13 = 4944;
  }
  else
  {
    _BitScanReverse64(&v8, v7);
    v9 = v8 - ((((v7 - 1) & v7) == 0) - 1);
    if ( v9 < 0xE )
      v9 = 14;
    v10 = v9 - 3;
    v11 = v9 - 14;
    if ( !v11 )
      v10 = 12;
    v12 = (((v7 - 1) >> v10) & 3) + 4 * v11 + 1;
    v25 = v12;
    v13 = 24 * v12 + 144;
  }
  v14 = sub_131C440(a1, a3, a5->m128i_i64[0] * v13, 64);
  if ( v14 )
  {
    a2[6].m128i_i64[1] = v14;
    v15 = v14 + 144 * a5->m128i_i64[0];
    if ( !a5->m128i_i64[0] )
    {
LABEL_15:
      a2[3].m128i_i64[1] = a4;
      a2[4] = _mm_loadu_si128(a5);
      a2[5] = _mm_loadu_si128(a5 + 1);
      v22 = a5[2].m128i_i64[0];
      a2[7].m128i_i32[0] = v25;
      a2[6].m128i_i64[0] = v22;
      a2->m128i_i64[0] = (__int64)sub_130E2B0;
      a2->m128i_i64[1] = (__int64)sub_130D0F0;
      a2[1].m128i_i64[0] = (__int64)sub_130E060;
      a2[1].m128i_i64[1] = (__int64)sub_130E070;
      a2[2].m128i_i64[0] = (__int64)sub_130E7B0;
      a2[2].m128i_i64[1] = (__int64)sub_130D1D0;
      return 0;
    }
    v16 = v14;
    v17 = 0;
    v18 = 24LL * v25;
    while ( !(unsigned __int8)sub_130AF40(v16) )
    {
      *(_BYTE *)(v16 + 112) = 1;
      v19 = v15;
      v20 = 0;
      for ( *(_QWORD *)(v16 + 120) = v15; ; v19 = *(_QWORD *)(v16 + 120) )
      {
        v21 = v20 + v19;
        v20 += 24;
        *(_BYTE *)v21 = 0;
        *(_QWORD *)(v21 + 8) = 0;
        *(_QWORD *)(v21 + 16) = 0;
        if ( v20 == v18 )
          break;
      }
      v15 += v18;
      ++v17;
      v16 += 144;
      *(_QWORD *)(v16 - 16) = 0;
      *(_DWORD *)(v16 - 8) = 0;
      if ( a5->m128i_i64[0] <= v17 )
        goto LABEL_15;
    }
  }
  return 1;
}
