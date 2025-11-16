// Function: sub_3011DA0
// Address: 0x3011da0
//
__int64 __fastcall sub_3011DA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    while ( 1 )
    {
      v2 = *(_QWORD *)(result + 24);
      if ( *(_BYTE *)v2 == 37 )
        break;
      result = *(_QWORD *)(result + 8);
      if ( !result )
        return result;
    }
    result = 0;
    if ( (*(_BYTE *)(v2 + 2) & 1) != 0 )
      return *(_QWORD *)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)));
  }
  return result;
}
