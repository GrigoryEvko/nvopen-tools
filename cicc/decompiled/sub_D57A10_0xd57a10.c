// Function: sub_D57A10
// Address: 0xd57a10
//
void __fastcall sub_D57A10(__int64 a1, int a2)
{
  void *v2; // rax
  __int64 v3; // rax
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  unsigned int v6; // r13d
  unsigned int v7; // ebx
  __int64 v8; // rdx
  __int64 v9; // r12

  v2 = sub_CB72A0();
  v3 = sub_CB69B0((__int64)v2, 2 * a2);
  v4 = *(__m128i **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x11u )
  {
    sub_CB6200(v3, "Loop Pass Manager\n", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F71B00);
    v4[1].m128i_i16[0] = 2674;
    *v4 = si128;
    *(_QWORD *)(v3 + 32) += 18LL;
  }
  v6 = a2 + 1;
  v7 = 0;
  while ( *(_DWORD *)(a1 + 200) > v7 )
  {
    v8 = v7++;
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8 * v8);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 136LL))(v9, v6);
    sub_B81320(a1 + 176, v9, v6);
  }
}
