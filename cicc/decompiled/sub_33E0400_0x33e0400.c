// Function: sub_33E0400
// Address: 0x33e0400
//
char __fastcall sub_33E0400(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al

  result = 0;
  if ( *(_DWORD *)(a2 + 64) == 2 )
  {
    result = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 24LL) == 11
          || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 24LL) == 35;
    if ( result )
    {
      if ( *(_DWORD *)(a2 + 24) != 56 )
        return sub_33E03A0(a1, a2, a3, 0);
    }
  }
  return result;
}
