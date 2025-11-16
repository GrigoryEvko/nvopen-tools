// Function: sub_21C9560
// Address: 0x21c9560
//
__int64 __fastcall sub_21C9560(__int64 a1, __int64 *a2, int a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  unsigned int v7; // r12d
  unsigned int v11; // eax
  __m128i *v12; // rsi
  __m128i *v13; // rax
  __m128i *v14; // rsi
  __m128i *v15; // rsi
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rax
  __m128i *v20; // rsi
  __int32 v21; // edx
  __m128i v22; // [rsp+10h] [rbp-70h] BYREF
  __m128i v23; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  int v25; // [rsp+38h] [rbp-48h]
  __m128i v26; // [rsp+40h] [rbp-40h] BYREF

  v22.m128i_i64[0] = 0;
  v22.m128i_i32[2] = 0;
  v23.m128i_i64[0] = 0;
  v23.m128i_i32[2] = 0;
  if ( a3 != 3 )
    return 1;
  LOBYTE(v11) = sub_21C2A00(a1, *a2, a2[1], (__int64)&v22);
  v7 = v11;
  if ( !(_BYTE)v11 )
  {
    if ( !(unsigned __int8)sub_21C2F60(a1, *a2, *a2, a2[1], (__int64)&v22, (__int64)&v23, a5, a6, a7) )
      return 1;
    v12 = *(__m128i **)(a4 + 8);
    v13 = *(__m128i **)(a4 + 16);
    if ( v12 == v13 )
    {
      sub_1D4B0A0((const __m128i **)a4, v12, &v22);
      v14 = *(__m128i **)(a4 + 8);
      if ( v14 != *(__m128i **)(a4 + 16) )
      {
        if ( !v14 )
          goto LABEL_11;
        goto LABEL_10;
      }
    }
    else
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(&v22);
        v12 = *(__m128i **)(a4 + 8);
        v13 = *(__m128i **)(a4 + 16);
      }
      v14 = v12 + 1;
      *(_QWORD *)(a4 + 8) = v14;
      if ( v13 != v14 )
      {
LABEL_10:
        *v14 = _mm_loadu_si128(&v23);
        v14 = *(__m128i **)(a4 + 8);
LABEL_11:
        *(_QWORD *)(a4 + 8) = v14 + 1;
        return v7;
      }
    }
    sub_1D4B0A0((const __m128i **)a4, v14, &v23);
    return v7;
  }
  v15 = *(__m128i **)(a4 + 8);
  if ( v15 == *(__m128i **)(a4 + 16) )
  {
    sub_1D4B0A0((const __m128i **)a4, v15, &v22);
  }
  else
  {
    if ( v15 )
    {
      a5 = _mm_loadu_si128(&v22);
      *v15 = a5;
      v15 = *(__m128i **)(a4 + 8);
    }
    *(_QWORD *)(a4 + 8) = v15 + 1;
  }
  v16 = *a2;
  v17 = *(_QWORD *)(a1 + 272);
  v18 = *(_QWORD *)(v16 + 72);
  v24 = v18;
  if ( v18 )
    sub_1623A60((__int64)&v24, v18, 2);
  v25 = *(_DWORD *)(v16 + 64);
  v19 = sub_1D38BB0(v17, 0, (__int64)&v24, 5, 0, 1, a5, a6, a7, 0);
  v20 = *(__m128i **)(a4 + 8);
  v26.m128i_i64[0] = v19;
  v26.m128i_i32[2] = v21;
  if ( v20 == *(__m128i **)(a4 + 16) )
  {
    sub_1D4B3A0((const __m128i **)a4, v20, &v26);
  }
  else
  {
    if ( v20 )
    {
      *v20 = _mm_loadu_si128(&v26);
      v20 = *(__m128i **)(a4 + 8);
    }
    *(_QWORD *)(a4 + 8) = v20 + 1;
  }
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return 0;
}
