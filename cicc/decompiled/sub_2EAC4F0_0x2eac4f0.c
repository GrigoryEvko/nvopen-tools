// Function: sub_2EAC4F0
// Address: 0x2eac4f0
//
__int64 __fastcall sub_2EAC4F0(__int64 a1)
{
  char v1; // cl
  unsigned int v2; // r8d
  __int64 v3; // rdx
  unsigned __int64 v4; // rax

  v1 = *(_BYTE *)(a1 + 34);
  v2 = -1;
  v3 = -(*(_QWORD *)(a1 + 8) | (1LL << v1));
  if ( (v3 & (*(_QWORD *)(a1 + 8) | (1LL << v1))) != 0 )
  {
    _BitScanReverse64(&v4, v3 & (*(_QWORD *)(a1 + 8) | (1LL << v1)));
    return 63 - ((unsigned int)v4 ^ 0x3F);
  }
  return v2;
}
