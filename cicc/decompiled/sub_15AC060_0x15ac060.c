// Function: sub_15AC060
// Address: 0x15ac060
//
__int64 __fastcall sub_15AC060(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax

  v1 = sub_16328F0(a1, "Debug Info Version", 18);
  if ( !v1 )
    return 0;
  if ( *(_BYTE *)v1 != 1 )
    return 0;
  v2 = *(_QWORD *)(v1 + 136);
  if ( *(_BYTE *)(v2 + 16) != 13 )
    return 0;
  result = *(_QWORD *)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
