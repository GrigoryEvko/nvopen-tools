// Function: sub_19936F0
// Address: 0x19936f0
//
char __fastcall sub_19936F0(__int64 a1, __int64 a2)
{
  unsigned __int16 v3; // ax
  __int64 v4; // rax
  char result; // al
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r13

  while ( 1 )
  {
    if ( (unsigned int)sub_15A9520(a1, 0) != 8 || (unsigned int)sub_15A9520(a1, 3u) != 4 )
      return 0;
    v3 = *(_WORD *)(a2 + 24);
    if ( v3 == 10 )
    {
      v4 = **(_QWORD **)(a2 - 8);
      if ( *(_BYTE *)(v4 + 8) == 15 )
        return *(_DWORD *)(v4 + 8) >> 8 == 3;
      return 0;
    }
    if ( (unsigned int)v3 - 4 <= 1 || (unsigned __int16)(v3 - 7) <= 2u )
      break;
    if ( (unsigned __int16)(v3 - 1) > 2u )
      return 0;
    a2 = *(_QWORD *)(a2 + 32);
  }
  v6 = *(_QWORD *)(a2 + 40);
  if ( !(_DWORD)v6 )
    return 0;
  v7 = 0;
  v8 = 8LL * (unsigned int)v6;
  while ( 1 )
  {
    result = sub_19936F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + v7));
    if ( result )
      break;
    v7 += 8;
    if ( v7 == v8 )
      return 0;
  }
  return result;
}
