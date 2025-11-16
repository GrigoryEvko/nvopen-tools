// Function: sub_37175A0
// Address: 0x37175a0
//
void __fastcall sub_37175A0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)a1 = &unk_4A3CB08;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 1;
  *(_QWORD *)(a1 + 104) = a1 + 8;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = a2;
  *(_QWORD *)(a1 + 144) = a3;
  if ( a3 )
    sub_2631330((const __m128i **)(a1 + 112), a3);
}
