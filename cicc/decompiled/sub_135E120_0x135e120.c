// Function: sub_135E120
// Address: 0x135e120
//
__int64 __fastcall sub_135E120(__int64 a1)
{
  __int64 **v1; // rdi
  __int64 v2; // rax

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v1 = *(__int64 ***)(a1 - 8);
  else
    v1 = (__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v2 = **v1;
  if ( *(_BYTE *)(v2 + 8) == 16 )
    v2 = **(_QWORD **)(v2 + 16);
  return *(_DWORD *)(v2 + 8) >> 8;
}
