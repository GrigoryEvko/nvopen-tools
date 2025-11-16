// Function: sub_736A50
// Address: 0x736a50
//
__int64 __fastcall sub_736A50(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax

  while ( (*(_BYTE *)(a1 + 89) & 5) == 5 )
  {
    v2 = sub_72B7D0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL));
    if ( !v2 )
      break;
    a1 = v2;
  }
  if ( dword_4F077C4 == 2 && *(char *)(a1 + 192) < 0 && (!*(_BYTE *)(a1 + 172) || sub_736A10(a1)) )
    return 1;
  v3 = 0;
  if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x10 )
    return v3;
  if ( (*(_BYTE *)(a1 + 195) & 3) == 1 || (*(_BYTE *)(a1 + 192) & 0x40) != 0 || *(_QWORD *)(a1 + 272) )
    return 1;
  v5 = *(_QWORD *)(a1 + 112);
  if ( v5 )
    return *(_QWORD *)(v5 + 272) == a1;
  return v3;
}
