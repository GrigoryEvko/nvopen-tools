// Function: sub_39888D0
// Address: 0x39888d0
//
__m128i *__fastcall sub_39888D0(__int64 a1)
{
  __int64 v1; // rdx
  __m128i *v2; // r12
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r15
  const __m128i *v7; // r14
  unsigned __int64 v8; // rbx
  const __m128i *v9; // [rsp+10h] [rbp-90h]
  __int32 v10; // [rsp+1Ch] [rbp-84h]
  __m128i *v11; // [rsp+20h] [rbp-80h]
  __m128i *v12; // [rsp+28h] [rbp-78h]
  char v13[8]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v14; // [rsp+38h] [rbp-68h]
  char v15[8]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v16; // [rsp+58h] [rbp-48h]

  v1 = *(unsigned int *)(a1 + 48);
  v2 = *(__m128i **)(a1 + 40);
  if ( v1 != 1 )
  {
    v4 = 16 * v1;
    v9 = &v2[v1];
    if ( v9 != v2 )
    {
      _BitScanReverse64(&v5, v4 >> 4);
      sub_3987CE0(v2, &v2[v1], 2LL * (int)(63 - (v5 ^ 0x3F)));
      if ( (unsigned __int64)v4 <= 0x100 )
      {
        sub_39859D0(v2, v9);
      }
      else
      {
        v11 = v2 + 16;
        sub_39859D0(v2, v2 + 16);
        if ( v9 != &v2[16] )
        {
          do
          {
            v6 = v11->m128i_i64[1];
            v7 = v11;
            v10 = v11->m128i_i32[0];
            while ( 1 )
            {
              v12 = (__m128i *)v7--;
              sub_15B1350((__int64)v15, *(unsigned __int64 **)(v6 + 24), *(unsigned __int64 **)(v6 + 32));
              v8 = v16;
              sub_15B1350(
                (__int64)v13,
                *(unsigned __int64 **)(v7->m128i_i64[1] + 24),
                *(unsigned __int64 **)(v7->m128i_i64[1] + 32));
              if ( v8 >= v14 )
                break;
              v7[1] = _mm_loadu_si128(v7);
            }
            ++v11;
            v12->m128i_i32[0] = v10;
            v12->m128i_i64[1] = v6;
          }
          while ( v9 != v11 );
        }
      }
      return *(__m128i **)(a1 + 40);
    }
  }
  return v2;
}
