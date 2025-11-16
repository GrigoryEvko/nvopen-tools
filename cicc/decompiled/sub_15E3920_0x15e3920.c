// Function: sub_15E3920
// Address: 0x15e3920
//
__int64 __fastcall sub_15E3920(__int64 a1)
{
  __int64 v1; // rdi

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v1 = *(_QWORD *)(a1 - 8);
  else
    v1 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  return *(_QWORD *)(v1 + 24);
}
