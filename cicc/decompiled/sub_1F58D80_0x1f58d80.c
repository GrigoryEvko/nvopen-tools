// Function: sub_1F58D80
// Address: 0x1f58d80
//
__int64 __fastcall sub_1F58D80(__int64 a1)
{
  unsigned int v1; // r12d
  char v3; // di
  unsigned int v4; // esi
  bool v5; // cc
  char v6; // dl

  v3 = *(_BYTE *)a1;
  if ( v3 )
  {
    v4 = sub_1F58BF0(v3);
    v5 = v4 <= 0x20;
    if ( v4 != 32 )
      goto LABEL_3;
LABEL_13:
    v6 = 5;
    goto LABEL_6;
  }
  v4 = sub_1F58D40(a1);
  v5 = v4 <= 0x20;
  if ( v4 == 32 )
    goto LABEL_13;
LABEL_3:
  if ( v5 )
  {
    if ( v4 == 8 )
    {
      v6 = 3;
    }
    else
    {
      v6 = 4;
      if ( v4 != 16 )
      {
        v6 = 2;
        if ( v4 != 1 )
          return (unsigned int)sub_1F58CC0(**(_QWORD ***)(a1 + 8), v4);
      }
    }
LABEL_6:
    LOBYTE(v1) = v6;
    return v1;
  }
  if ( v4 == 64 )
  {
    v6 = 6;
    goto LABEL_6;
  }
  if ( v4 == 128 )
  {
    v6 = 7;
    goto LABEL_6;
  }
  return (unsigned int)sub_1F58CC0(**(_QWORD ***)(a1 + 8), v4);
}
