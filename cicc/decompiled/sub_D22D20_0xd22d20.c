// Function: sub_D22D20
// Address: 0xd22d20
//
__int64 __fastcall sub_D22D20(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9

  if ( sub_D222C0(a1) )
    return sub_D22340(*(_BYTE **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)), a2, v2, v3, v4, v5);
  else
    return sub_D22340(*(_BYTE **)(a1 - 96), a2, v2, v3, v4, v5);
}
