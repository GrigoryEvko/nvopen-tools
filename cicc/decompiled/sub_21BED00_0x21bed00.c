// Function: sub_21BED00
// Address: 0x21bed00
//
void __fastcall sub_21BED00(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v2);
  sub_1D49010(v2);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v3, v4, v5, v6);
}
