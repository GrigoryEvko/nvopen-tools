// Function: sub_2E2F9E0
// Address: 0x2e2f9e0
//
__int64 __fastcall sub_2E2F9E0(__int64 a1)
{
  __int64 result; // rax
  int v2; // edx

  result = *(unsigned int *)(a1 + 44);
  v2 = *(_DWORD *)(a1 + 44) & 4;
  if ( (result & 8) != 0 )
  {
    if ( !v2 )
    {
      sub_2E89060();
      result = *(unsigned int *)(a1 + 44);
      if ( (result & 4) != 0 && (result & 8) == 0 )
        return sub_2E89050();
    }
  }
  else if ( v2 )
  {
    return sub_2E89050();
  }
  return result;
}
