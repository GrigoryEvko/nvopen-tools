// Function: sub_B449A0
// Address: 0xb449a0
//
__int64 __fastcall sub_B449A0(__int64 a1)
{
  __int64 v1; // rdx
  unsigned __int8 v2; // al
  __int64 v3; // rcx
  _BYTE **v4; // rax
  _BYTE **v5; // rcx

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return 0;
  if ( sub_B91C10(a1, 18) )
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
      BUG();
    v1 = sub_B91C10(a1, 18);
    v2 = *(_BYTE *)(v1 - 16);
    if ( (v2 & 2) != 0 )
    {
      v4 = *(_BYTE ***)(v1 - 32);
      v3 = *(unsigned int *)(v1 - 24);
    }
    else
    {
      v3 = (*(_WORD *)(v1 - 16) >> 6) & 0xF;
      v4 = (_BYTE **)(v1 - 8LL * ((v2 >> 2) & 0xF) - 16);
    }
    v5 = &v4[v3];
    while ( v5 != ++v4 )
    {
      if ( **v4 != 6 )
        return 1;
    }
  }
  return 0;
}
