// Function: sub_806BE0
// Address: 0x806be0
//
void __fastcall sub_806BE0(__m128i *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r13
  _DWORD v4[5]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = *a3;
  sub_806A20(a1, a2, v4);
  if ( v3 )
    sub_7E6810(v3, (__int64)a1, 1);
}
