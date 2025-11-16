// Function: sub_18A58D0
// Address: 0x18a58d0
//
__int64 __fastcall sub_18A58D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  unsigned int v4; // ecx

  result = *(_QWORD *)(a1 + 120);
  if ( *(_QWORD *)(a1 + 72) )
  {
    v2 = *(_QWORD *)(a1 + 56);
    if ( !result )
      return *(_QWORD *)(v2 + 40);
    v3 = *(_QWORD *)(a1 + 104);
    v4 = *(_DWORD *)(v2 + 32);
    if ( *(_DWORD *)(v3 + 32) > v4 || *(_DWORD *)(v3 + 32) == v4 && *(_DWORD *)(v3 + 36) > *(_DWORD *)(v2 + 36) )
      return *(_QWORD *)(v2 + 40);
    return sub_18A5060(a1);
  }
  if ( result )
    return sub_18A5060(a1);
  return result;
}
