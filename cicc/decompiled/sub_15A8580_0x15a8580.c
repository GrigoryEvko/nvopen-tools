// Function: sub_15A8580
// Address: 0x15a8580
//
__int64 __fastcall sub_15A8580(__int64 a1, unsigned int a2)
{
  __int64 v2; // r8
  __int64 i; // rax
  __int64 v4; // rcx

  v2 = *(_QWORD *)(a1 + 224);
  for ( i = *(unsigned int *)(a1 + 232); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v4 = v2 + 20 * (i >> 1);
      if ( *(_DWORD *)(v4 + 12) >= a2 )
        break;
      v2 = v4 + 20;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v2;
    }
  }
  return v2;
}
