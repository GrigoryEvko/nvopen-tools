// Function: sub_255B4C0
// Address: 0x255b4c0
//
__int64 __fastcall sub_255B4C0(_BYTE *a1, _BYTE *a2)
{
  char v2; // r9
  char v3; // r8
  char v4; // cl
  char v5; // dl
  char v6; // bl
  char v7; // al
  char v8; // r10
  char v9; // r11
  char v10; // si
  bool v11; // bl
  bool v12; // al
  bool v13; // r10
  bool v15; // zf

  v2 = a1[8];
  v3 = a1[9];
  v4 = a1[10];
  v5 = a1[11];
  v6 = a2[8];
  v7 = a2[9];
  v8 = a2[10];
  v9 = a2[11];
  v10 = v2 == v6;
  if ( v2 == 3 || v2 == v6 )
  {
    v2 = v6;
    v11 = v3 == v7;
    if ( v3 == 3 )
      goto LABEL_19;
  }
  else
  {
    v10 = 1;
    if ( v6 != 3 )
    {
      v15 = v2 == -1;
      v2 = -1;
      v10 = v15;
    }
    v11 = v3 == v7;
    if ( v3 == 3 )
      goto LABEL_19;
  }
  if ( !v11 )
  {
    if ( v7 != 3 )
    {
      v15 = v3 == -1;
      v3 = -1;
      v10 &= v15;
    }
    v12 = v4 == v8;
    if ( v4 == 3 )
      goto LABEL_20;
    goto LABEL_10;
  }
LABEL_19:
  v10 &= v11;
  v3 = v7;
  v12 = v4 == v8;
  if ( v4 == 3 )
    goto LABEL_20;
LABEL_10:
  if ( !v12 )
  {
    v12 = 1;
    if ( v8 != 3 )
    {
      v15 = v4 == -1;
      v4 = -1;
      v12 = v15;
    }
    v13 = v5 == v9;
    if ( v5 == 3 )
      goto LABEL_21;
    goto LABEL_14;
  }
LABEL_20:
  v4 = v8;
  v13 = v5 == v9;
  if ( v5 == 3 )
    goto LABEL_21;
LABEL_14:
  if ( v13 )
  {
LABEL_21:
    v5 = v9;
    goto LABEL_17;
  }
  v13 = 1;
  if ( v9 != 3 )
  {
    v15 = v5 == -1;
    v5 = -1;
    v13 = v15;
  }
LABEL_17:
  a1[8] = v2;
  a1[9] = v3;
  a1[10] = v4;
  a1[11] = v5;
  return v13 & (unsigned __int8)(v10 & v12);
}
