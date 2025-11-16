// Function: sub_254C9B0
// Address: 0x254c9b0
//
__int64 __fastcall sub_254C9B0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdi

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v2 = *(_QWORD *)(a1 - 8);
  else
    v2 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  nullsub_1518();
  return (32LL * a2 + v2) | 3;
}
