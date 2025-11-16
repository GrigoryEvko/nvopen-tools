// Function: sub_2CDF320
// Address: 0x2cdf320
//
bool __fastcall sub_2CDF320(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2
        && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80)
        && (*(_BYTE *)(v2 + 33) & 0x20) != 0
        && *(_DWORD *)(v2 + 36) == 9151 )
      {
        return **(_BYTE **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) == 20;
      }
    }
  }
  return result;
}
