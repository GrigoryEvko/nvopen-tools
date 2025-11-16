// Function: sub_180E4A0
// Address: 0x180e4a0
//
char __fastcall sub_180E4A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  int v7; // r12d
  __int64 v8; // r14
  unsigned int v9; // r15d
  unsigned __int64 v10; // r14
  __int64 v11; // r15
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  __m128i v15; // xmm3
  __int64 v16; // rax
  __m128i v17; // xmm1
  __m128i v19; // [rsp-58h] [rbp-58h] BYREF
  __m128i v20[4]; // [rsp-48h] [rbp-48h] BYREF

  v6 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v6 + 16) )
    BUG();
  v7 = *(_DWORD *)(v6 + 36);
  if ( v7 == 201 )
  {
    v6 = *(unsigned int *)(a1 + 3736);
    if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 3740) )
    {
      sub_16CD150(a1 + 3728, (const void *)(a1 + 3744), 0, 8, a5, a6);
      v6 = *(unsigned int *)(a1 + 3736);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 3728) + 8 * v6) = a2;
    ++*(_DWORD *)(a1 + 3736);
  }
  else
  {
    if ( v7 == 120 )
    {
      *(_QWORD *)(a1 + 3760) = a2;
      return v6;
    }
    v6 = *(_QWORD *)(a1 + 8);
    if ( *(_BYTE *)(v6 + 230) )
    {
      LOBYTE(v6) = v7 - 116;
      if ( (unsigned int)(v7 - 116) <= 1 )
      {
        v8 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v8 + 16) != 13 )
          BUG();
        v9 = *(_DWORD *)(v8 + 32);
        if ( v9 <= 0x40 )
        {
          v10 = *(_QWORD *)(v8 + 24);
          v6 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9);
          if ( v10 == v6 )
            return v6;
        }
        else
        {
          LODWORD(v6) = sub_16A58F0(v8 + 24);
          if ( v9 == (_DWORD)v6 )
            return v6;
          LODWORD(v6) = sub_16A57B0(v8 + 24);
          if ( v9 - (unsigned int)v6 > 0x40 )
            return v6;
          v6 = *(_QWORD *)(v8 + 24);
          v10 = *(_QWORD *)v6;
        }
        if ( v10 != -1 )
        {
          LOBYTE(v6) = sub_1594730(*(_QWORD *)(a1 + 488), v10);
          if ( (_BYTE)v6 )
          {
            v6 = sub_180E050(a1, *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
            v11 = v6;
            if ( v6 )
            {
              LOBYTE(v6) = sub_180D640(*(_QWORD *)(a1 + 8), v6);
              if ( (_BYTE)v6 )
              {
                v19.m128i_i64[0] = a2;
                v20[0].m128i_i8[8] = v7 == 116;
                v19.m128i_i64[1] = v11;
                v20[0].m128i_i64[0] = v10;
                LOBYTE(v6) = sub_15F8F00(v11);
                if ( (_BYTE)v6 )
                {
                  v16 = *(unsigned int *)(a1 + 3440);
                  if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 3444) )
                  {
                    sub_16CD150(a1 + 3432, (const void *)(a1 + 3448), 0, 32, v12, v13);
                    v16 = *(unsigned int *)(a1 + 3440);
                  }
                  v17 = _mm_loadu_si128(v20);
                  v6 = *(_QWORD *)(a1 + 3432) + 32 * v16;
                  *(__m128i *)v6 = _mm_loadu_si128(&v19);
                  *(__m128i *)(v6 + 16) = v17;
                  ++*(_DWORD *)(a1 + 3440);
                }
                else if ( byte_4FA7BA0 )
                {
                  v14 = *(unsigned int *)(a1 + 3168);
                  if ( (unsigned int)v14 >= *(_DWORD *)(a1 + 3172) )
                  {
                    sub_16CD150(a1 + 3160, (const void *)(a1 + 3176), 0, 32, v12, v13);
                    v14 = *(unsigned int *)(a1 + 3168);
                  }
                  v15 = _mm_loadu_si128(v20);
                  v6 = *(_QWORD *)(a1 + 3160) + 32 * v14;
                  *(__m128i *)v6 = _mm_loadu_si128(&v19);
                  *(__m128i *)(v6 + 16) = v15;
                  ++*(_DWORD *)(a1 + 3168);
                }
              }
            }
          }
        }
      }
    }
  }
  return v6;
}
