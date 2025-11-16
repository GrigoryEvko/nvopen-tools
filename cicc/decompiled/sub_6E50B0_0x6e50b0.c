// Function: sub_6E50B0
// Address: 0x6e50b0
//
__int64 __fastcall sub_6E50B0(__int64 a1, __int64 *a2)
{
  char v2; // al
  __int64 v3; // r8
  unsigned __int8 v4; // al

  v2 = *(_BYTE *)(a1 + 80);
  v3 = a1;
  if ( v2 == 16 )
  {
    v3 = **(_QWORD **)(a1 + 88);
    v2 = *(_BYTE *)(v3 + 80);
  }
  if ( v2 == 24 )
    v3 = *(_QWORD *)(v3 + 88);
  if ( (*(_BYTE *)(a1 + 82) & 4) == 0 )
  {
    v4 = *(_BYTE *)(v3 + 80);
    if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
    {
      if ( (unsigned __int8)(v4 - 10) <= 1u && (*(_BYTE *)(*(_QWORD *)(v3 + 88) + 206LL) & 0x10) != 0 )
        sub_6E50A0();
    }
    else
    {
      if ( v4 > 0xBu )
      {
        if ( v4 == 17 )
          sub_721090(a1);
      }
      else if ( (v4 > 6u || v4 == 2) && (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
      {
        return sub_6DED10(v3, a2);
      }
      sub_8767A0(4, v3, a2, (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0);
    }
    return 0;
  }
  return 0;
}
