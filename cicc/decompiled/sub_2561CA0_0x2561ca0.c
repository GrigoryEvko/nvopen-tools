// Function: sub_2561CA0
// Address: 0x2561ca0
//
__int64 __fastcall sub_2561CA0(__m128i *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // r12
  __m128i v5; // xmm1
  __m128i *v6; // rax
  __int64 v7; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax

  v2 = sub_2509800(a1);
  if ( v2 != 5 )
  {
    if ( v2 > 5 )
    {
      if ( (unsigned __int8)(v2 - 6) <= 1u )
        goto LABEL_12;
    }
    else
    {
      if ( v2 == 4 )
      {
        v3 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
        v4 = v3;
        if ( v3 )
        {
          v5 = _mm_loadu_si128(a1);
          v6 = (__m128i *)(v3 + 56);
          v6[-3].m128i_i64[0] = 0;
          v6[-3].m128i_i64[1] = 0;
          v6[1] = v5;
          v6[-2].m128i_i64[0] = 0;
          v6[-2].m128i_i32[2] = 0;
          *(_QWORD *)(v4 + 40) = v6;
          *(_QWORD *)(v4 + 48) = 0x200000000LL;
          *(_QWORD *)(v4 + 96) = 0x1FF00000000LL;
          v7 = *(_QWORD *)(a2 + 128);
          *(_OWORD *)(v4 + 104) = 0;
          *(_QWORD *)(v4 + 168) = v7;
          *(_QWORD *)v4 = off_4A1BD88;
          *(_QWORD *)(v4 + 88) = &unk_4A1BE20;
          *(_OWORD *)(v4 + 120) = 0;
          *(_OWORD *)(v4 + 136) = 0;
          *(_OWORD *)(v4 + 152) = 0;
          return v4;
        }
        return 0;
      }
      if ( v2 >= 0 )
LABEL_12:
        BUG();
    }
    return 0;
  }
  v9 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
  v10 = v9;
  if ( !v9 )
    return 0;
  *(__m128i *)(v9 + 72) = _mm_loadu_si128(a1);
  sub_2553350(v9);
  *(_QWORD *)(v10 + 96) = 0x1FF00000000LL;
  v11 = *(_QWORD *)(a2 + 128);
  *(_OWORD *)(v10 + 104) = 0;
  *(_QWORD *)(v10 + 168) = v11;
  *(_QWORD *)v10 = off_4A1BE80;
  *(_QWORD *)(v10 + 88) = &unk_4A1BF18;
  *(_OWORD *)(v10 + 120) = 0;
  *(_OWORD *)(v10 + 136) = 0;
  *(_OWORD *)(v10 + 152) = 0;
  return v10;
}
