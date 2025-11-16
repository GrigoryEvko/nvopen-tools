// Function: sub_E92900
// Address: 0xe92900
//
__int64 __fastcall sub_E92900(__int64 a1, int a2)
{
  int v2; // eax
  __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 44);
  if ( a2 )
  {
    if ( *(_DWORD *)(a1 + 40) != 2 )
      *(_DWORD *)(a1 + 40) = a2;
    result = (unsigned int)(v2 + 1);
    *(_DWORD *)(a1 + 44) = result;
  }
  else
  {
    if ( !v2 )
      sub_C64ED0("Mismatched bundle_lock/unlock directives", 1u);
    result = (unsigned int)(v2 - 1);
    *(_DWORD *)(a1 + 44) = result;
    if ( !(_DWORD)result )
      *(_DWORD *)(a1 + 40) = 0;
  }
  return result;
}
