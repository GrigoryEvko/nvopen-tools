// Function: sub_2539AE0
// Address: 0x2539ae0
//
void __fastcall sub_2539AE0(__int64 a1, __int64 a2)
{
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96) | *(_BYTE *)(a1 + 97) & 3;
  sub_2535850(a2, (__m128i *)(a1 + 72), a1 + 88, 0);
}
