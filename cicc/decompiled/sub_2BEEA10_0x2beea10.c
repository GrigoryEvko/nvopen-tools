// Function: sub_2BEEA10
// Address: 0x2beea10
//
__int64 __fastcall sub_2BEEA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // rax
  __m128i *v6; // rdx
  __int64 v7; // rdi
  __m128i si128; // xmm0
  __int64 v9; // rdi
  _BYTE *v10; // rax

  v5 = sub_CB7210(a1, a2, a3, a4, a5);
  v6 = (__m128i *)v5[4];
  v7 = (__int64)v5;
  if ( v5[3] - (_QWORD)v6 <= 0x11u )
  {
    v7 = sub_CB6200((__int64)v5, (unsigned __int8 *)"InstructionCount: ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_43A0140);
    v6[1].m128i_i16[0] = 8250;
    *v6 = si128;
    v5[4] += 18LL;
  }
  v9 = sub_CB59F0(v7, *(unsigned int *)(a2 + 40));
  v10 = *(_BYTE **)(v9 + 32);
  if ( *(_BYTE **)(v9 + 24) == v10 )
  {
    sub_CB6200(v9, (unsigned __int8 *)"\n", 1u);
    return 0;
  }
  else
  {
    *v10 = 10;
    ++*(_QWORD *)(v9 + 32);
    return 0;
  }
}
