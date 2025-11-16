// Function: sub_986550
// Address: 0x986550
//
__int64 __fastcall sub_986550(__int64 a1)
{
  __int64 v1; // r8

  v1 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    return *(_QWORD *)(a1 - 8);
  return v1;
}
