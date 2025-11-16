// Function: sub_85B9B0
// Address: 0x85b9b0
//
__int64 __fastcall sub_85B9B0(__int64 a1, _DWORD *a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v4; // rdx

  v2 = *(_BYTE *)(a1 + 140);
  v3 = 0;
  if ( (unsigned __int8)(v2 - 9) > 2u )
    return v3;
  v4 = *(_QWORD *)(a1 + 168);
  if ( (*(_BYTE *)(v4 + 112) & 2) == 0 )
  {
    if ( v2 == 9
      && (*(_BYTE *)(v4 + 109) & 0x20) != 0
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 181LL) & 2) != 0
      && !*(_QWORD *)(v4 + 240) )
    {
      *a2 = 1;
      return 1;
    }
    return v3;
  }
  *a2 = 1;
  return 0;
}
