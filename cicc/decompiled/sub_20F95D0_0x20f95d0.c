// Function: sub_20F95D0
// Address: 0x20f95d0
//
void __fastcall sub_20F95D0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r12
  __int64 v4; // rsi

  v2 = *(__int64 **)(a2 + 112);
  v3 = &v2[2 * *(unsigned int *)(a2 + 120)];
  while ( v2 != v3 )
  {
    v4 = *v2;
    v2 += 2;
    sub_20F9570(a1, v4 & 0xFFFFFFFFFFFFFFF8LL);
  }
}
