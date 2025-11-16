// Function: sub_15A0F20
// Address: 0x15a0f20
//
__int64 __fastcall sub_15A0F20(__int64 a1)
{
  int v2; // r13d
  unsigned int v3; // ebx

  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    return 0;
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( !v2 )
    return 0;
  v3 = 0;
  while ( *(_BYTE *)(sub_15A0A60(a1, v3) + 16) != 9 )
  {
    if ( v2 == ++v3 )
      return 0;
  }
  return 1;
}
