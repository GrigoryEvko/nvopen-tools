// Function: sub_38D7880
// Address: 0x38d7880
//
__int64 __fastcall sub_38D7880(__int64 a1, int a2)
{
  int v2; // eax
  __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 40);
  if ( a2 )
  {
    if ( *(_DWORD *)(a1 + 36) != 2 )
      *(_DWORD *)(a1 + 36) = a2;
    result = (unsigned int)(v2 + 1);
    *(_DWORD *)(a1 + 40) = result;
  }
  else
  {
    if ( !v2 )
      sub_16BD130("Mismatched bundle_lock/unlock directives", 1u);
    result = (unsigned int)(v2 - 1);
    *(_DWORD *)(a1 + 40) = result;
    if ( !(_DWORD)result )
      *(_DWORD *)(a1 + 36) = 0;
  }
  return result;
}
