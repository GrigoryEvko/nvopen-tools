// Function: sub_2538CB0
// Address: 0x2538cb0
//
void __fastcall sub_2538CB0(__int64 a1, __int64 a2)
{
  char v2; // al
  char v3; // al
  int v4[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = *(_BYTE *)(a1 + 97);
  v4[0] = 81;
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96) | v2 & 3;
  v3 = sub_2516400(a2, (__m128i *)(a1 + 72), (__int64)v4, 1, 1, 0);
  sub_2535850(a2, (__m128i *)(a1 + 72), a1 + 88, v3);
}
