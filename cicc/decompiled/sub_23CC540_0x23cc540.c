// Function: sub_23CC540
// Address: 0x23cc540
//
void __fastcall sub_23CC540(__int64 a1, const __m128i *a2, char a3)
{
  __m128i v4; // xmm0
  unsigned __int64 v5; // rax
  int v6; // ebx
  unsigned int v7; // r14d
  int v8; // eax
  unsigned int v9; // esi

  v4 = _mm_loadu_si128(a2);
  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_BYTE *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 48) = 0x400000000LL;
  *(_QWORD *)(a1 + 144) = 0x400000000LL;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 184) = a1 + 200;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = -1;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(__m128i *)a1 = v4;
  v5 = sub_23CBB70((_QWORD *)a1);
  *(_QWORD *)(a1 + 216) = v5;
  *(_QWORD *)(a1 + 232) = v5;
  if ( (unsigned int)a2->m128i_i64[1] )
  {
    v6 = a2->m128i_i64[1];
    v7 = 0;
    v8 = 0;
    do
    {
      *(_DWORD *)(a1 + 224) = v7;
      v9 = v7++;
      v8 = sub_23CBE70(a1, v9, v8 + 1);
    }
    while ( v6 != v7 );
  }
  sub_23CA790(a1);
  if ( a3 )
    sub_23CB250((_QWORD *)a1);
}
