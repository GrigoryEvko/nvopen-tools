// Function: sub_A3D360
// Address: 0xa3d360
//
void __fastcall sub_A3D360(__m128i *a1, __m128i *a2, __int64 a3)
{
  __m128i *v3; // rbx
  bool v4; // r8
  __m128i *v5; // rax
  __int64 v6; // rsi
  __int32 v7; // edi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __m128i v10; // xmm0
  __m128i *v11; // r15
  __int64 v12; // rax
  __int64 v15; // [rsp+18h] [rbp-58h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h] BYREF
  __m128i v17; // [rsp+30h] [rbp-40h] BYREF

  v15 = a3;
  if ( a1 != a2 )
  {
    v3 = a1 + 1;
    while ( a2 != v3 )
    {
      v4 = sub_A3D0E0((__int64)&v15, v3, a1);
      v5 = v3++;
      if ( v4 )
      {
        v6 = v3[-1].m128i_i64[0];
        v7 = v3[-1].m128i_i32[2];
        v8 = v5 - a1;
        if ( (char *)v5 - (char *)a1 > 0 )
        {
          do
          {
            v9 = v5[-1].m128i_i64[0];
            --v5;
            v5[1].m128i_i64[0] = v9;
            v5[1].m128i_i32[2] = v5->m128i_i32[2];
            --v8;
          }
          while ( v8 );
        }
        a1->m128i_i64[0] = v6;
        a1->m128i_i32[2] = v7;
      }
      else
      {
        v10 = _mm_loadu_si128(v3 - 1);
        v11 = v3 - 2;
        v16 = v15;
        v17 = v10;
        while ( sub_A3D0E0((__int64)&v16, &v17, v11) )
        {
          v12 = v11->m128i_i64[0];
          --v11;
          v11[2].m128i_i64[0] = v12;
          v11[2].m128i_i32[2] = v11[1].m128i_i32[2];
        }
        v11[1].m128i_i64[0] = v17.m128i_i64[0];
        v11[1].m128i_i32[2] = v17.m128i_i32[2];
      }
    }
  }
}
