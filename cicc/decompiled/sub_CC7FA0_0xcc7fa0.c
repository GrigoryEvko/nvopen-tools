// Function: sub_CC7FA0
// Address: 0xcc7fa0
//
__int64 __fastcall sub_CC7FA0(_DWORD *a1, _DWORD *a2)
{
  int v2; // ecx
  bool v3; // r9
  int v4; // eax
  int v5; // edx
  int v6; // r10d
  unsigned int v7; // r8d
  int v9; // eax
  int v10; // r10d
  int v11; // edx

  v2 = a1[11];
  v3 = 0;
  if ( v2 == 14 )
    v3 = a1[12] == 1;
  v4 = a1[8];
  v5 = a2[8];
  switch ( v4 )
  {
    case 36:
      if ( v5 == 1 )
      {
        v9 = a1[10];
        v10 = a1[9];
        v11 = a2[9];
        if ( v9 != 1 )
          goto LABEL_12;
        goto LABEL_26;
      }
      goto LABEL_7;
    case 1:
      if ( v5 == 36 )
        goto LABEL_11;
      goto LABEL_7;
    case 37:
      if ( v5 != 2 )
        goto LABEL_7;
LABEL_11:
      v9 = a1[10];
      v10 = a1[9];
      v11 = a2[9];
      if ( v9 != 1 )
      {
LABEL_12:
        v7 = 0;
        if ( v10 != v11 )
          return v7;
        LOBYTE(v7) = v3 || a2[10] == v9;
        if ( !(_BYTE)v7 )
          return v7;
        v7 = 0;
        if ( v2 != a2[11] )
          return v7;
        goto LABEL_15;
      }
LABEL_26:
      v7 = 0;
      if ( v10 != v11 || a2[10] != 1 )
        return v7;
LABEL_28:
      LOBYTE(v7) = a2[11] == v2;
      return v7;
  }
  if ( v4 != 2 )
  {
LABEL_7:
    v6 = a1[10];
    if ( v6 != 1 )
      goto LABEL_8;
    goto LABEL_20;
  }
  if ( v5 == 37 )
    goto LABEL_11;
  v6 = a1[10];
  if ( v6 == 1 )
  {
LABEL_20:
    v7 = 0;
    if ( v4 != v5 )
      return v7;
    if ( a1[9] != a2[9] )
      return v7;
    LOBYTE(v7) = v3 || a2[10] == 1;
    if ( !(_BYTE)v7 )
      return v7;
    goto LABEL_28;
  }
LABEL_8:
  v7 = 0;
  if ( v4 != v5 )
    return v7;
  if ( a1[9] != a2[9] )
    return v7;
  LOBYTE(v7) = v3 || a2[10] == v6;
  if ( !(_BYTE)v7 )
    return v7;
  if ( a2[11] != v2 )
    return 0;
LABEL_15:
  v7 = 0;
  if ( a1[12] == a2[12] )
    LOBYTE(v7) = a1[13] == a2[13];
  return v7;
}
