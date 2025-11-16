// Function: sub_B4E100
// Address: 0xb4e100
//
__int64 __fastcall sub_B4E100(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned int v4; // r8d

  v3 = *(_QWORD *)(a1 + 8);
  v4 = 0;
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 > 1 || *(_QWORD *)(a2 + 8) != *(_QWORD *)(v3 + 24) )
    return 0;
  LOBYTE(v4) = *(_BYTE *)(*(_QWORD *)(a3 + 8) + 8LL) == 12;
  return v4;
}
