// Function: sub_1D62240
// Address: 0x1d62240
//
__int64 __fastcall sub_1D62240(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v6; // r14d
  const __m128i *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // r8
  __m128i v14; // xmm2
  __int64 v15; // rax
  __m128i *v16; // rax
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdi
  __m128i *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-68h] BYREF
  __m128i v26; // [rsp+10h] [rbp-60h] BYREF
  __m128i v27; // [rsp+20h] [rbp-50h] BYREF
  __m128i v28; // [rsp+30h] [rbp-40h] BYREF
  __int64 v29; // [rsp+40h] [rbp-30h]

  if ( a3 == 1 )
    return sub_1D61F00(a1, a2, a4);
  v6 = 1;
  if ( a3 )
  {
    v7 = *(const __m128i **)(a1 + 56);
    v8 = v7[1].m128i_i64[1];
    if ( v8 && a2 != v7[2].m128i_i64[1] )
      return 0;
    v9 = *(_QWORD *)(a1 + 8);
    v10 = v8 + a3;
    v11 = *(_QWORD *)(a1 + 32);
    v12 = *(_QWORD *)(a1 + 24);
    v26 = _mm_loadu_si128(v7);
    v13 = *(unsigned int *)(a1 + 40);
    v27 = _mm_loadu_si128(v7 + 1);
    v14 = _mm_loadu_si128(v7 + 2);
    v27.m128i_i64[1] = v10;
    v28 = v14;
    v15 = v7[3].m128i_i64[0];
    v28.m128i_i64[1] = a2;
    v29 = v15;
    v6 = (*(__int64 (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64, _QWORD))(*(_QWORD *)v9 + 736LL))(
           v9,
           v12,
           &v26,
           v11,
           v13,
           0);
    if ( (_BYTE)v6 )
    {
      v16 = *(__m128i **)(a1 + 56);
      *v16 = _mm_loadu_si128(&v26);
      v16[1] = _mm_loadu_si128(&v27);
      v16[2] = _mm_loadu_si128(&v28);
      v16[3].m128i_i64[0] = v29;
      if ( *(_BYTE *)(a2 + 16) == 35 )
      {
        if ( *(_QWORD *)(a2 - 48) )
        {
          v18 = *(_QWORD *)(a2 - 24);
          if ( *(_BYTE *)(v18 + 16) == 13 )
          {
            v28.m128i_i64[1] = *(_QWORD *)(a2 - 48);
            v19 = *(_DWORD *)(v18 + 32);
            if ( v19 > 0x40 )
              v20 = **(_QWORD **)(v18 + 24);
            else
              v20 = (__int64)(*(_QWORD *)(v18 + 24) << (64 - (unsigned __int8)v19)) >> (64 - (unsigned __int8)v19);
            v21 = *(_QWORD *)(a1 + 8);
            v26.m128i_i64[1] += v27.m128i_i64[1] * v20;
            v22 = (*(__int64 (__fastcall **)(__int64, _QWORD, __m128i *, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v21 + 736LL))(
                    v21,
                    *(_QWORD *)(a1 + 24),
                    &v26,
                    *(_QWORD *)(a1 + 32),
                    *(unsigned int *)(a1 + 40),
                    0);
            if ( (_BYTE)v22 )
            {
              v23 = *(_QWORD *)a1;
              v25 = a2;
              v6 = v22;
              sub_14EF3D0(v23, &v25);
              v24 = *(__m128i **)(a1 + 56);
              *v24 = _mm_loadu_si128(&v26);
              v24[1] = _mm_loadu_si128(&v27);
              v24[2] = _mm_loadu_si128(&v28);
              v24[3].m128i_i64[0] = v29;
            }
          }
        }
      }
    }
    else
    {
      return 0;
    }
  }
  return v6;
}
