// Function: sub_70FCE0
// Address: 0x70fce0
//
_BOOL8 __fastcall sub_70FCE0(__int64 a1)
{
  char v1; // al
  _BOOL4 v2; // r8d
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // al

  v1 = *(_BYTE *)(a1 + 173);
  if ( v1 != 6 )
  {
    v2 = v1 != 12;
    if ( v1 == 8 )
    {
      v4 = *(_QWORD *)(a1 + 176);
      v2 = 0;
      if ( *(_BYTE *)(v4 + 173) == 6 && *(_BYTE *)(v4 + 176) == 6 )
      {
        v5 = *(_QWORD *)(a1 + 184);
        if ( *(_BYTE *)(v5 + 173) == 6 && *(_BYTE *)(v5 + 176) == 6 )
          return *(_QWORD *)(v4 + 184) == *(_QWORD *)(v5 + 184);
      }
    }
    return v2;
  }
  v6 = *(_BYTE *)(a1 + 176);
  if ( v6 == 1 )
    return sub_70FCC0(*(_QWORD *)(a1 + 184));
  v2 = 1;
  if ( v6 )
    return v2;
  return sub_70FCD0(*(_QWORD *)(a1 + 184));
}
