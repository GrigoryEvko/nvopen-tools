// Function: sub_15A95F0
// Address: 0x15a95f0
//
__int64 __fastcall sub_15A95F0(__int64 a1, __int64 a2)
{
  if ( *(_BYTE *)(a2 + 8) == 16 )
    a2 = **(_QWORD **)(a2 + 16);
  return 8 * (unsigned int)sub_15A95A0(a1, *(_DWORD *)(a2 + 8) >> 8);
}
