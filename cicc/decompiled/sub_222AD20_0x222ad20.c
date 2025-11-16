// Function: sub_222AD20
// Address: 0x222ad20
//
__int64 __fastcall sub_222AD20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // r12

  if ( !a3 )
    return 0;
  for ( i = 0; i != a3; ++i )
  {
    if ( putwc(*(_DWORD *)(a2 + 4 * i), *(__FILE **)(a1 + 64)) == -1 )
      break;
  }
  return i;
}
