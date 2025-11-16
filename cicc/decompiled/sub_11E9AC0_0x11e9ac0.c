// Function: sub_11E9AC0
// Address: 0x11e9ac0
//
__int64 __fastcall sub_11E9AC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_B343C0(
             a3,
             0xF1u,
             *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             0x100u,
             *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
             0x100u,
             *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             0,
             0,
             0,
             0,
             0);
  if ( result )
  {
    if ( *(_BYTE *)result == 85 )
      *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  }
  return result;
}
