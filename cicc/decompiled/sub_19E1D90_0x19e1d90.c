// Function: sub_19E1D90
// Address: 0x19e1d90
//
__int64 __fastcall sub_19E1D90(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 78 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v2 + 16) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 && *(_DWORD *)(v2 + 36) == 197 )
      return *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  }
  return result;
}
