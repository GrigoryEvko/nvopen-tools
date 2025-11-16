// Function: sub_68BC10
// Address: 0x68bc10
//
__int64 __fastcall sub_68BC10(__int64 a1, const __m128i *a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int8 v5; // al
  int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 v10; // [rsp+18h] [rbp-378h] BYREF
  _OWORD v11[9]; // [rsp+20h] [rbp-370h] BYREF
  __m128i v12; // [rsp+B0h] [rbp-2E0h]
  __m128i v13; // [rsp+C0h] [rbp-2D0h]
  __m128i v14; // [rsp+D0h] [rbp-2C0h]
  __m128i v15; // [rsp+E0h] [rbp-2B0h]
  __m128i v16; // [rsp+F0h] [rbp-2A0h]
  __m128i v17; // [rsp+100h] [rbp-290h]
  __m128i v18; // [rsp+110h] [rbp-280h]
  __m128i v19; // [rsp+120h] [rbp-270h]
  __m128i v20; // [rsp+130h] [rbp-260h]
  __m128i v21; // [rsp+140h] [rbp-250h]
  __m128i v22; // [rsp+150h] [rbp-240h]
  __m128i v23; // [rsp+160h] [rbp-230h]
  __m128i v24; // [rsp+170h] [rbp-220h]
  _QWORD v25[66]; // [rsp+180h] [rbp-210h] BYREF

  memset(v25, 0, 0x1D8u);
  v25[19] = v25;
  v25[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v25[22]) |= 1u;
  v25[0] = *(_QWORD *)a1;
  v25[36] = *(_QWORD *)(a1 + 120);
  v25[6] = *(_QWORD *)(a1 + 64);
  v3 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v4 = *(_QWORD *)(v3 + 624);
  *(_QWORD *)(v3 + 624) = v25;
  v11[0] = _mm_loadu_si128(a2);
  v5 = a2[1].m128i_i8[0];
  v11[1] = _mm_loadu_si128(a2 + 1);
  v11[2] = _mm_loadu_si128(a2 + 2);
  v11[3] = _mm_loadu_si128(a2 + 3);
  v11[4] = _mm_loadu_si128(a2 + 4);
  v11[5] = _mm_loadu_si128(a2 + 5);
  v11[6] = _mm_loadu_si128(a2 + 6);
  v11[7] = _mm_loadu_si128(a2 + 7);
  v11[8] = _mm_loadu_si128(a2 + 8);
  if ( v5 == 2 )
  {
    v12 = _mm_loadu_si128(a2 + 9);
    v13 = _mm_loadu_si128(a2 + 10);
    v14 = _mm_loadu_si128(a2 + 11);
    v15 = _mm_loadu_si128(a2 + 12);
    v16 = _mm_loadu_si128(a2 + 13);
    v17 = _mm_loadu_si128(a2 + 14);
    v18 = _mm_loadu_si128(a2 + 15);
    v19 = _mm_loadu_si128(a2 + 16);
    v20 = _mm_loadu_si128(a2 + 17);
    v21 = _mm_loadu_si128(a2 + 18);
    v22 = _mm_loadu_si128(a2 + 19);
    v23 = _mm_loadu_si128(a2 + 20);
    v24 = _mm_loadu_si128(a2 + 21);
  }
  else if ( v5 == 5 || v5 == 1 )
  {
    v12.m128i_i64[0] = a2[9].m128i_i64[0];
  }
  v6 = sub_8D3A70(*(_QWORD *)(a1 + 120));
  v7 = *(_QWORD *)(a1 + 120);
  if ( v6 )
  {
    sub_8470D0((unsigned int)v11, v7, 1, 1, 144, 0, (__int64)&v10);
  }
  else
  {
    sub_843C40((unsigned int)v11, v7, 0, 0, 1, 1, 144);
    v10 = sub_725A70(3);
    *(_QWORD *)(v10 + 56) = sub_6F6F40(v11, 0);
  }
  sub_6E2920(v10);
  v8 = v10;
  if ( v10 )
  {
    *(_BYTE *)(a1 + 177) = 2;
    *(_QWORD *)(a1 + 184) = v8;
    *(_QWORD *)(v8 + 8) = a1;
    sub_7340D0(v8, *(_BYTE *)(a1 + 136) <= 2u, 1);
    if ( (*(_BYTE *)(a1 + 176) & 0x40) == 0 )
    {
      if ( *(_QWORD *)a1 )
        sub_86F660(a1);
    }
  }
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_QWORD *)(result + 624) = v4;
  return result;
}
