// Function: sub_23ECF80
// Address: 0x23ecf80
//
char __fastcall sub_23ECF80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // r13d
  __m128i *v9; // rax
  __int64 v10; // r15
  unsigned int v11; // r14d
  unsigned __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rdx
  const __m128i *v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rdx
  const __m128i *v24; // rcx
  unsigned __int64 v25; // r13
  __int64 v26; // rdi
  const void *v27; // rsi
  unsigned __int64 v28; // r13
  __int64 v29; // rdi
  const void *v30; // rsi
  _QWORD v32[3]; // [rsp+10h] [rbp-50h] BYREF
  bool v33; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD *)(a2 - 32);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v7 = *(_DWORD *)(v6 + 36);
  if ( v7 == 342 )
  {
    v14 = *(unsigned int *)(a1 + 5968);
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 5972) )
    {
      sub_C8D5F0(a1 + 5960, (const void *)(a1 + 5976), v14 + 1, 8u, a5, a6);
      v14 = *(unsigned int *)(a1 + 5968);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 5960) + 8 * v14) = a2;
    ++*(_DWORD *)(a1 + 5968);
  }
  else if ( v7 == 216 )
  {
    *(_QWORD *)(a1 + 5992) = a2;
  }
  v9 = *(__m128i **)(a1 + 8);
  if ( v9[5].m128i_i8[6] )
  {
    LOBYTE(v9) = sub_B46A10(a2);
    if ( (_BYTE)v9 )
    {
      LODWORD(v9) = -32 * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v10 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v11 = *(_DWORD *)(v10 + 32);
      if ( v11 )
      {
        if ( v11 <= 0x40 )
        {
          v9 = (__m128i *)(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11));
          if ( *(__m128i **)(v10 + 24) == v9 )
            return (char)v9;
          v12 = *(_QWORD *)(v10 + 24);
        }
        else
        {
          LODWORD(v9) = sub_C445E0(v10 + 24);
          if ( v11 == (_DWORD)v9 )
            return (char)v9;
          LODWORD(v9) = sub_C444A0(v10 + 24);
          if ( v11 - (unsigned int)v9 > 0x40 )
            return (char)v9;
          v9 = *(__m128i **)(v10 + 24);
          v12 = v9->m128i_i64[0];
        }
        if ( v12 != -1 )
        {
          LOBYTE(v9) = sub_AC3610(*(_QWORD *)(a1 + 464), v12);
          if ( (_BYTE)v9 )
          {
            v9 = (__m128i *)sub_98C100(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), 1);
            v13 = (__int64)v9;
            if ( v9 )
            {
              LOBYTE(v9) = sub_23ECB50(*(_QWORD *)(a1 + 8), (__int64)v9);
              if ( (_BYTE)v9 )
              {
                v32[0] = a2;
                v33 = v7 == 210;
                v32[1] = v13;
                v32[2] = v12;
                LOBYTE(v9) = sub_B4D040(v13);
                if ( (_BYTE)v9 )
                {
                  v21 = *(unsigned int *)(a1 + 5664);
                  v22 = v21 + 1;
                  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 5668) )
                  {
                    v25 = *(_QWORD *)(a1 + 5656);
                    v26 = a1 + 5656;
                    v27 = (const void *)(a1 + 5672);
                    if ( v25 > (unsigned __int64)v32 || (unsigned __int64)v32 >= v25 + 32 * v21 )
                    {
                      sub_C8D5F0(v26, v27, v22, 0x20u, v15, v16);
                      v21 = *(unsigned int *)(a1 + 5664);
                      v23 = *(_QWORD *)(a1 + 5656);
                      v24 = (const __m128i *)v32;
                    }
                    else
                    {
                      sub_C8D5F0(v26, v27, v22, 0x20u, v15, v16);
                      v23 = *(_QWORD *)(a1 + 5656);
                      v21 = *(unsigned int *)(a1 + 5664);
                      v24 = (const __m128i *)((char *)v32 + v23 - v25);
                    }
                  }
                  else
                  {
                    v23 = *(_QWORD *)(a1 + 5656);
                    v24 = (const __m128i *)v32;
                  }
                  v9 = (__m128i *)(v23 + 32 * v21);
                  *v9 = _mm_loadu_si128(v24);
                  v9[1] = _mm_loadu_si128(v24 + 1);
                  ++*(_DWORD *)(a1 + 5664);
                }
                else if ( byte_4FE0D28 )
                {
                  v17 = *(unsigned int *)(a1 + 5392);
                  v18 = v17 + 1;
                  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 5396) )
                  {
                    v28 = *(_QWORD *)(a1 + 5384);
                    v29 = a1 + 5384;
                    v30 = (const void *)(a1 + 5400);
                    if ( v28 > (unsigned __int64)v32 || (unsigned __int64)v32 >= v28 + 32 * v17 )
                    {
                      sub_C8D5F0(v29, v30, v18, 0x20u, v15, v16);
                      v17 = *(unsigned int *)(a1 + 5392);
                      v19 = *(_QWORD *)(a1 + 5384);
                      v20 = (const __m128i *)v32;
                    }
                    else
                    {
                      sub_C8D5F0(v29, v30, v18, 0x20u, v15, v16);
                      v19 = *(_QWORD *)(a1 + 5384);
                      v17 = *(unsigned int *)(a1 + 5392);
                      v20 = (const __m128i *)((char *)v32 + v19 - v28);
                    }
                  }
                  else
                  {
                    v19 = *(_QWORD *)(a1 + 5384);
                    v20 = (const __m128i *)v32;
                  }
                  v9 = (__m128i *)(v19 + 32 * v17);
                  *v9 = _mm_loadu_si128(v20);
                  v9[1] = _mm_loadu_si128(v20 + 1);
                  ++*(_DWORD *)(a1 + 5392);
                }
              }
            }
            else
            {
              *(_BYTE *)(a1 + 5928) = 1;
            }
          }
        }
      }
    }
  }
  return (char)v9;
}
