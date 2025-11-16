// Function: sub_35032B0
// Address: 0x35032b0
//
void __fastcall sub_35032B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r12
  __int64 v4; // rsi

  v2 = *(__int64 **)(a2 + 120);
  v3 = &v2[2 * *(unsigned int *)(a2 + 128)];
  while ( v3 != v2 )
  {
    v4 = *v2;
    v2 += 2;
    sub_3503250(a1, v4 & 0xFFFFFFFFFFFFFFF8LL);
  }
}
