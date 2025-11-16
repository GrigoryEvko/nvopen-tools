// Function: sub_11E3A90
// Address: 0x11e3a90
//
__int64 __fastcall sub_11E3A90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  if ( **(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) != 20 )
    return 0;
  result = sub_11CCCD0(
             *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             a3,
             *(_QWORD *)(a1 + 16),
             *(__int64 **)(a1 + 24));
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
