// Function: sub_B3C4B0
// Address: 0xb3c4b0
//
void __fastcall sub_B3C4B0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 v3; // rdi
  __int64 v4; // rsi

  if ( a2 < *(_DWORD *)(a1 + 72) )
  {
    *(_DWORD *)(a1 + 12) = a2;
    v2 = *(_QWORD *)(a1 + 64);
    v3 = a1 + 16;
    v4 = v2 + 56LL * a2;
    *(_DWORD *)(v3 - 12) = *(_DWORD *)v4;
    sub_B3C2C0(v3, (__int64 *)(v4 + 8));
  }
}
