// Function: sub_1EF80B0
// Address: 0x1ef80b0
//
void __fastcall sub_1EF80B0(__int64 a1, __int64 a2)
{
  __m128i *v3; // rbx
  unsigned __int32 v4; // r13d
  __m128i *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rsi
  bool v8; // cc
  __m128i *v9; // r15
  unsigned __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int64 v12; // rdi
  __m128i v13; // xmm0
  int v14; // edx
  unsigned __int64 v15; // rdi
  __m128i *v16; // r12
  unsigned __int64 v17; // rdi
  __m128i v18; // xmm1
  __int32 v19; // eax
  unsigned __int64 v20; // [rsp-60h] [rbp-60h]
  __int32 v21; // [rsp-50h] [rbp-50h]
  __int32 v22; // [rsp-4Ch] [rbp-4Ch]
  __int64 v23; // [rsp-48h] [rbp-48h]
  __int64 v24; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v3 = (__m128i *)(a1 + 40);
    if ( a2 != a1 + 40 )
    {
      v20 = a1 + ((a2 - 80 - a1) & 0xFFFFFFFFFFFFFFF8LL) + 80;
      do
      {
        v4 = v3->m128i_u32[2];
        v24 = v3[1].m128i_i64[1];
        v5 = v3;
        v6 = v3[1].m128i_i64[0];
        v3 = (__m128i *)((char *)v3 + 40);
        v7 = v3[-3].m128i_i64[1];
        v8 = v4 <= *(_DWORD *)(a1 + 8);
        v3[-2].m128i_i64[1] = 0;
        v23 = v6;
        v21 = v3[-2].m128i_i32[1];
        v22 = v3[-1].m128i_i32[2];
        v3[-1].m128i_i64[0] = 0;
        v3[-1].m128i_i32[2] = 0;
        if ( v8 )
        {
          v16 = v3 - 5;
          if ( v4 > v3[-5].m128i_i32[2] )
          {
            while ( 1 )
            {
              v17 = v16[3].m128i_u64[1];
              v16[2].m128i_i64[1] = v16->m128i_i64[0];
              v16[3].m128i_i32[0] = v16->m128i_i32[2];
              v16[3].m128i_i32[1] = v16->m128i_i32[3];
              _libc_free(v17);
              v18 = _mm_loadu_si128(v16 + 1);
              v19 = v16[2].m128i_i32[0];
              v16[1].m128i_i64[0] = 0;
              v16[1].m128i_i64[1] = 0;
              v16[4].m128i_i32[2] = v19;
              v16[2].m128i_i32[0] = 0;
              *(__m128i *)((char *)v16 + 56) = v18;
              if ( v4 <= v16[-2].m128i_i32[0] )
                break;
              v16 = (__m128i *)((char *)v16 - 40);
            }
          }
          else
          {
            v16 = v5;
          }
          v16->m128i_i32[2] = v4;
          v16->m128i_i64[0] = v7;
          v16->m128i_i32[3] = v21;
          _libc_free(0);
          v16[1].m128i_i64[0] = v23;
          v16[1].m128i_i64[1] = v24;
          v16[2].m128i_i32[0] = v22;
        }
        else
        {
          v9 = v3 - 4;
          v10 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v5->m128i_i64 - a1) >> 3);
          if ( (__int64)v5->m128i_i64 - a1 > 0 )
          {
            do
            {
              v11 = v9[-1].m128i_i64[0];
              v12 = v9[2].m128i_u64[1];
              v9 = (__m128i *)((char *)v9 - 40);
              v9[4].m128i_i64[0] = v11;
              v9[4].m128i_i32[2] = v9[2].m128i_i32[0];
              v9[4].m128i_i32[3] = v9[2].m128i_i32[1];
              _libc_free(v12);
              v13 = _mm_loadu_si128((__m128i *)((char *)v9 + 40));
              v14 = v9[3].m128i_i32[2];
              v9[2].m128i_i64[1] = 0;
              v9[3].m128i_i64[0] = 0;
              v9[5] = v13;
              v9[6].m128i_i32[0] = v14;
              v9[3].m128i_i32[2] = 0;
              --v10;
            }
            while ( v10 );
          }
          v15 = *(_QWORD *)(a1 + 16);
          *(_DWORD *)(a1 + 8) = v4;
          *(_QWORD *)a1 = v7;
          *(_DWORD *)(a1 + 12) = v21;
          _libc_free(v15);
          *(_QWORD *)(a1 + 16) = v23;
          *(_QWORD *)(a1 + 24) = v24;
          *(_DWORD *)(a1 + 32) = v22;
        }
      }
      while ( v3 != (__m128i *)v20 );
    }
  }
}
