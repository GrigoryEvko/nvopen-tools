// Function: sub_1AD3470
// Address: 0x1ad3470
//
__int64 __fastcall sub_1AD3470(__int64 a1)
{
  __int64 v2; // rdi

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 16) - 73) <= 1u )
    return *(_QWORD *)(a1 - 24);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v2 = *(_QWORD *)(a1 - 8);
  else
    v2 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  return *(_QWORD *)v2;
}
