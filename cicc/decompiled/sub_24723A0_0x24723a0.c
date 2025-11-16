// Function: sub_24723A0
// Address: 0x24723a0
//
__int64 __fastcall sub_24723A0(__int64 a1, __int64 a2, unsigned int a3)
{
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    return sub_246F3F0(a1, *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * a3));
  else
    return sub_246F3F0(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) + 32LL * a3));
}
