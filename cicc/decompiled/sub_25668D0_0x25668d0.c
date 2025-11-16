// Function: sub_25668D0
// Address: 0x25668d0
//
__int64 __fastcall sub_25668D0(__m128i *a1, __int64 a2)
{
  char v3; // al
  __int64 *v4; // rsi
  __int64 v5; // rax
  __int64 v6; // r12
  __m128i v7; // xmm0
  __m128i *v8; // rax
  __int64 v9; // rbx
  unsigned __int8 *v10; // rax

  v3 = sub_2509800(a1);
  if ( v3 != 4 )
  {
    if ( v3 > 4 )
    {
      if ( (unsigned __int8)(v3 - 5) > 2u )
        return 0;
    }
    else if ( (unsigned __int8)v3 > 3u )
    {
      return 0;
    }
    BUG();
  }
  v4 = *(__int64 **)(a2 + 128);
  v5 = sub_A777F0(0x110u, v4);
  v6 = v5;
  if ( v5 )
  {
    v7 = _mm_loadu_si128(a1);
    v8 = (__m128i *)(v5 + 56);
    v8[-3].m128i_i64[0] = 0;
    v8[-3].m128i_i64[1] = 0;
    v8[1] = v7;
    v8[-2].m128i_i64[0] = 0;
    v8[-2].m128i_i32[2] = 0;
    *(_QWORD *)(v6 + 40) = v8;
    *(_QWORD *)(v6 + 48) = 0x200000000LL;
    *(_WORD *)(v6 + 96) = 256;
    *(_QWORD *)(v6 + 104) = v6 + 120;
    *(_QWORD *)(v6 + 112) = 0x600000000LL;
    *(_QWORD *)v6 = off_4A190D8;
    *(_QWORD *)(v6 + 168) = 0;
    *(_QWORD *)(v6 + 176) = 0;
    *(_QWORD *)(v6 + 184) = 0;
    *(_DWORD *)(v6 + 192) = 0;
    *(_QWORD *)(v6 + 200) = 0;
    *(_QWORD *)(v6 + 208) = 0;
    *(_QWORD *)(v6 + 216) = 0;
    *(_DWORD *)(v6 + 224) = 0;
    *(_QWORD *)(v6 + 232) = 0;
    *(_QWORD *)(v6 + 240) = 0;
    *(_QWORD *)(v6 + 248) = 0;
    *(_DWORD *)(v6 + 256) = 0;
    v9 = *(_QWORD *)(a2 + 208);
    *(_QWORD *)(v6 + 88) = &unk_4A19168;
    *(_QWORD *)(v6 + 264) = 0;
    v10 = sub_250CBE0(a1->m128i_i64, (__int64)v4);
    *(_QWORD *)(v6 + 264) = sub_2554D30(*(_QWORD *)(v9 + 240), (__int64)v10, 0);
  }
  return v6;
}
