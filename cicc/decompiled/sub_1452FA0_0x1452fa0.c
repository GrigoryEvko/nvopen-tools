// Function: sub_1452FA0
// Address: 0x1452fa0
//
__int64 __fastcall sub_1452FA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // edx
  unsigned int v4; // eax

  if ( a1 == a2 )
    return 1;
  if ( *(_WORD *)(a1 + 24) != 10 || *(_WORD *)(a2 + 24) != 10 )
    return 0;
  v2 = *(_QWORD *)(a1 - 8);
  if ( *(_BYTE *)(v2 + 16) <= 0x17u
    || *(_BYTE *)(*(_QWORD *)(a2 - 8) + 16LL) <= 0x17u
    || !(unsigned __int8)sub_15F41F0(*(_QWORD *)(a1 - 8)) )
  {
    return 0;
  }
  v3 = *(unsigned __int8 *)(v2 + 16);
  v4 = v3 - 35;
  LOBYTE(v4) = (unsigned int)(v3 - 35) <= 0x11;
  return ((_BYTE)v3 == 56) | v4;
}
