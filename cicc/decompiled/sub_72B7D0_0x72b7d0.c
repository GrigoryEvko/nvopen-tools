// Function: sub_72B7D0
// Address: 0x72b7d0
//
__int64 __fastcall sub_72B7D0(__int64 a1)
{
  while ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    a1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  return *(_QWORD *)(a1 + 48);
}
