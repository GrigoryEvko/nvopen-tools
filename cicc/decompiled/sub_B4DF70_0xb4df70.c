// Function: sub_B4DF70
// Address: 0xb4df70
//
__int64 __fastcall sub_B4DF70(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d

  v2 = 0;
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL) - 17 <= 1 )
    LOBYTE(v2) = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 12;
  return v2;
}
