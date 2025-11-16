// Function: sub_70C550
// Address: 0x70c550
//
_BOOL8 __fastcall sub_70C550(unsigned __int8 a1, const __m128i *a2, const __m128i *a3)
{
  int v4; // r13d
  int v5; // eax
  int v7; // [rsp+8h] [rbp-28h] BYREF
  _DWORD v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = sub_70BE30(a1, a2, a3, &v7);
  v5 = sub_70BE30(a1, a2 + 1, a3 + 1, v8);
  return (v5 | v4 | v8[0] | v7) == 0;
}
