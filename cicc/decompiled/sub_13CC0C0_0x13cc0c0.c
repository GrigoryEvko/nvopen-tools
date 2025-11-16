// Function: sub_13CC0C0
// Address: 0x13cc0c0
//
bool __fastcall sub_13CC0C0(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned int v2; // r12d
  unsigned __int64 v3; // r12
  int v5; // r13d
  unsigned int v6; // r12d
  __int64 v7; // rax

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 > 0x10u )
    return 0;
  if ( v1 == 9 )
    return 1;
  if ( v1 == 13 )
  {
    v2 = *(_DWORD *)(a1 + 32);
    if ( v2 <= 0x40 )
    {
      v3 = *(_QWORD *)(a1 + 24);
      return (unsigned int)sub_16431D0(*(_QWORD *)a1) <= v3;
    }
    if ( v2 - (unsigned int)sub_16A57B0(a1 + 24) <= 0x40 )
    {
      v3 = **(_QWORD **)(a1 + 24);
      return (unsigned int)sub_16431D0(*(_QWORD *)a1) <= v3;
    }
    return 1;
  }
  if ( (v1 & 0xFB) == 8 )
  {
    v5 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v5 )
      return 1;
    v6 = 0;
    while ( 1 )
    {
      v7 = sub_15A0A60(a1, v6);
      if ( !(unsigned __int8)sub_13CC0C0(v7) )
        break;
      if ( ++v6 == v5 )
        return 1;
    }
  }
  return 0;
}
