// Function: sub_D52340
// Address: 0xd52340
//
__int64 __fastcall sub_D52340(__int64 a1)
{
  int v1; // ecx
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) == 0 )
    return a1 + 32LL * (v1 == 3) - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  result = *(_QWORD *)(a1 - 8);
  if ( v1 == 3 )
    result += 32;
  return result;
}
