// Function: sub_2E25370
// Address: 0x2e25370
//
void __fastcall sub_2E25370(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int64 v4; // rbx
  __m128i si128; // xmm0
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  __m128i *v9; // rdx
  int v10; // r12d

  v2 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 8) )
  {
    v4 = 0;
    do
    {
      v9 = *(__m128i **)(a2 + 32);
      v10 = v4;
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v9 > 0x12u )
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42EF1B0);
        v9[1].m128i_i8[2] = 37;
        v6 = a2;
        v9[1].m128i_i16[0] = 10016;
        *v9 = si128;
        *(_QWORD *)(a2 + 32) += 19LL;
      }
      else
      {
        v6 = sub_CB6200(a2, "Virtual register '%", 0x13u);
      }
      v7 = sub_CB59D0(v6, v4);
      v8 = *(_QWORD *)(v7 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v7 + 24) - v8) <= 2 )
      {
        sub_CB6200(v7, "':\n", 3u);
      }
      else
      {
        *(_BYTE *)(v8 + 2) = 10;
        *(_WORD *)v8 = 14887;
        *(_QWORD *)(v7 + 32) += 3LL;
      }
      ++v4;
      sub_2E24F60((_QWORD *)(*(_QWORD *)a1 + 56LL * (v10 & 0x7FFFFFFF)), a2);
    }
    while ( v2 != v4 );
  }
}
