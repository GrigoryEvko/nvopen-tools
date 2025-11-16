// Function: sub_7AC1A0
// Address: 0x7ac1a0
//
__int64 __fastcall sub_7AC1A0(char a1, _DWORD *a2)
{
  char v2; // cl
  bool v3; // r9
  unsigned int v4; // r8d
  bool v5; // r10

  if ( (unk_4D04A10 & 1) == 0 )
  {
    v2 = unk_4D04A10 & 0x18;
    if ( (unk_4D04A10 & 2) != 0 )
    {
      v3 = 0;
      goto LABEL_4;
    }
    if ( v2 && (a1 & 0x10) != 0 )
    {
LABEL_12:
      sub_6851C0(0x1F6u, a2);
      return 1;
    }
    return 0;
  }
  v4 = a1 & 4;
  v2 = unk_4D04A10 & 0x18;
  v3 = v4 != 0;
  if ( (unk_4D04A10 & 2) == 0 )
  {
    if ( !v2 )
    {
      if ( (a1 & 4) == 0 )
        return v4;
      goto LABEL_19;
    }
    if ( (a1 & 0x10) != 0 )
    {
      if ( (a1 & 4) != 0 )
        goto LABEL_19;
      goto LABEL_12;
    }
    if ( (a1 & 4) != 0 )
      goto LABEL_19;
    return 0;
  }
LABEL_4:
  v4 = a1 & 8;
  v5 = 0;
  if ( v2 )
    v5 = (a1 & 0x10) != 0;
  if ( v3 && (a1 & 8) != 0 )
  {
    if ( (unk_4D04A10 & 4) != 0 )
    {
LABEL_8:
      sub_6851C0(0x119u, a2);
      return 1;
    }
    goto LABEL_19;
  }
  if ( !v3 )
  {
    if ( (a1 & 8) != 0 )
      goto LABEL_8;
    if ( !v5 )
      return v4;
    goto LABEL_12;
  }
LABEL_19:
  sub_6851C0(0x11Bu, a2);
  return 1;
}
