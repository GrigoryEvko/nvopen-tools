// Function: sub_3140B20
// Address: 0x3140b20
//
unsigned __int64 __fastcall sub_3140B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 result; // rax
  _QWORD *v6; // rax
  __m128i *v7; // rdx
  __int64 v8; // rdi
  __m128i si128; // xmm0

  *(_BYTE *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)a1 = &unk_4A085B0;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  result = 0x100000000LL;
  *(_QWORD *)(a1 + 24) = 0x100000000LL;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x100000000LL;
  *(_BYTE *)(a1 + 144) = 0;
  if ( (_BYTE)a2 )
  {
    v6 = sub_CB7210(a1, a2, a1 + 112, a4, a5);
    v7 = (__m128i *)v6[4];
    v8 = (__int64)v6;
    result = v6[3] - (_QWORD)v7;
    if ( result <= 0x44 )
    {
      return sub_CB6200(v8, "Pass Level, Pass Name, Num of Dropped Variables, Func or Module Name\n", 0x45u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_44D03A0);
      v7[4].m128i_i32[0] = 1701667150;
      v7[4].m128i_i8[4] = 10;
      *v7 = si128;
      v7[1] = _mm_load_si128((const __m128i *)&xmmword_44D03B0);
      v7[2] = _mm_load_si128((const __m128i *)&xmmword_44D03C0);
      v7[3] = _mm_load_si128((const __m128i *)&xmmword_44D03D0);
      *(_QWORD *)(v8 + 32) += 69LL;
    }
  }
  return result;
}
