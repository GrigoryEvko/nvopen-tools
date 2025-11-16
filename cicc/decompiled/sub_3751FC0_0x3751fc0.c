// Function: sub_3751FC0
// Address: 0x3751fc0
//
__int64 __fastcall sub_3751FC0(__int64 a1)
{
  unsigned int i; // r8d
  __int16 v2; // ax

  for ( i = *(_DWORD *)(a1 + 68); i; --i )
  {
    v2 = *(_WORD *)(*(_QWORD *)(a1 + 48) + 16LL * (i - 1));
    if ( v2 != 262 )
    {
      if ( v2 == 1 )
        --i;
      return i;
    }
  }
  return i;
}
