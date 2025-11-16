// Function: sub_730C30
// Address: 0x730c30
//
__int64 __fastcall sub_730C30(__int64 *a1, int a2, int a3, __int64 a4, const __m128i *a5, int a6)
{
  __int64 v6; // r15
  __int64 v12; // rax
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __int64 v18; // rsi
  __m128i v19; // xmm5
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // [rsp+18h] [rbp-A8h]
  int v26; // [rsp+2Ch] [rbp-94h] BYREF
  _OWORD v27[4]; // [rsp+30h] [rbp-90h] BYREF
  __m128i v28; // [rsp+70h] [rbp-50h]
  __m128i v29; // [rsp+80h] [rbp-40h]

  v26 = 0;
  if ( (*((_BYTE *)a1 + 89) & 4) != 0 )
  {
    v6 = *a1;
    if ( *((_BYTE *)a1 + 173) == 12 && *((_BYTE *)a1 + 176) == 3 && a1[24] )
      v6 = a1[24];
    v12 = a1[5];
    if ( v12 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v12 + 32) + 90LL) & 0x10) != 0 )
        return v6;
      v13 = _mm_loadu_si128(a5);
      v14 = _mm_loadu_si128(a5 + 1);
      v16 = _mm_loadu_si128(a5 + 2);
      v17 = _mm_loadu_si128(a5 + 3);
      v18 = *(_QWORD *)(v12 + 32);
      v19 = _mm_loadu_si128(a5 + 5);
      v28 = _mm_loadu_si128(a5 + 4);
      v28.m128i_i32[3] = 0;
      v27[0] = v13;
      v27[1] = v14;
      v27[2] = v16;
      v27[3] = v17;
      v29 = v19;
      v20 = sub_8A1CE0(v6, v18, a2, a3, a4, 0, 0, a6, (__int64)&v26, (__int64)v27);
      v21 = v20;
      if ( v6 == v20 )
      {
        v25 = v20;
        v22 = sub_894B00(v20, v18, v20);
        v21 = v25;
        if ( v22 )
        {
          v23 = sub_8A2270(v22, a2, a3, a4, a6, (unsigned int)&v26, (__int64)a5);
          v21 = sub_7D3640(v18, v23, a4);
        }
      }
      if ( !v26 )
        return v21;
    }
    return 0;
  }
  return a1[24];
}
